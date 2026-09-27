"""Run MuVAP over an excerpt of an AVCC segment, one prediction per frame.

This is the offline twin of `demo/server.py`. The server holds a rolling window
that the camera fills; here the window is filled from a recording instead, and
the same `StreamingEncoders` cache runs the frozen pair over it frame by frame.
Every frame therefore sees exactly what a live listener would have seen at that
moment - ten seconds of history and nothing after it - which is the only reading
that a demo of a *streaming* model is allowed to show.

What comes out is the whole state of the model at each frame, not a summary:

* `gvap`   `[T, n_classes]` - the global head's full posterior over the
  role-relative codebook, so the renderer can rank it and show what the model
  thinks the conversation is doing rather than only the shift scalar;
* `p_shift` `[T]` - the hold/shift readout drawn from that posterior;
* `svap`   `[S, T, bins]` - each face's own future, one independent bin per
  span. Bin 0 covers the 0.2 s straddling now, which is the model's answer to
  "is this face talking";
* `bbox`, `visible`, `vad` - where each face was, whether it was tracked, and
  what the corpus says was said. The annotation travels along for comparison;
  nothing in the forward pass ever sees it.

The ten seconds before the excerpt are read too, and pushed through the cache
without being kept. Frame zero of the output then already has a full window
behind it, instead of the model warming up on screen.
"""

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.media import AudioSpec, apply_normalization, load_waveform, waveform_stats
from demo.streaming import StreamingEncoders
from models.muvap import frozen_pair, load_fusion
from preprocess.asd.video import crop_face
from preprocess.muvap.avcc import (
    read_boxes,
    read_rttm,
    speaker_key,
    vad_path,
)
from projection_window import ProjectionWindow

FPS = 25
SAMPLE_RATE = 16_000
SAMPLES_PER_FRAME = SAMPLE_RATE // FPS
FACE = 112
CONTEXT_SEC = 10.0
CONTEXT_FRAMES = int(CONTEXT_SEC * FPS)


def resolve(root: Path, video_id: str, segment_id: str):
    """The four files one segment is made of, in the AVCC layout."""
    for name in ("orig_video", "videos", "../orig_video", "../videos"):
        media = (root / name).resolve()
        if media.is_dir():
            break
    else:
        raise FileNotFoundError(f"no segment video beside {root}")
    rttm = vad_path(root, video_id, segment_id)
    if not rttm.exists():
        raise FileNotFoundError(f"no activity file {rttm}")
    return (
        media / video_id / f"{segment_id}.mp4",
        root / "bbox" / video_id / f"{segment_id}.csv",
        rttm,
    )


def decode(video_path, boxes, speakers, first, last):
    """Crops, boxes and visibility for one frame span, in one sequential read.

    Seeking to `first` and reading forward, rather than decoding the whole
    segment: an AVCC segment runs to half an hour and a demo wants a minute of
    it. The crop geometry is `preprocess.asd.video.crop_face`, the same function
    the training pack was written with - a demo cropped differently would be
    showing a different model.
    """
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"could not open {video_path}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.set(cv2.CAP_PROP_POS_FRAMES, first)

    count = last - first
    faces = np.full((len(speakers), count, FACE, FACE), 128, dtype=np.uint8)
    bbox = np.zeros((len(speakers), count, 4), dtype=np.float32)
    visible = np.zeros((len(speakers), count), dtype=bool)
    scale = np.array([1.0 / width, 1.0 / height] * 2, dtype=np.float32)

    for index in range(count):
        ok, frame = capture.read()
        if not ok:
            break
        present = boxes.get(first + index, {})
        for row, speaker in enumerate(speakers):
            box = present.get(speaker)
            if box is None:
                continue
            faces[row, index] = crop_face(frame, box)
            bbox[row, index] = np.asarray(box, dtype=np.float32) * scale
            visible[row, index] = True
    capture.release()
    return faces, bbox, visible


def load_model(vap_weights, asd_weights, muvap_weights, config, device):
    encoders = frozen_pair(vap_weights, asd_weights).to(device).eval()
    fusion, cfg = load_fusion(muvap_weights, config)
    return encoders, fusion.to(device).eval(), cfg


@torch.no_grad()
def run(args) -> None:
    root = Path(args.root)
    video_path, bbox_path, rttm_path = resolve(root, args.video, args.segment)

    capture = cv2.VideoCapture(str(video_path))
    source_fps = float(capture.get(cv2.CAP_PROP_FPS))
    source_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    if abs(source_fps - FPS) > 1e-6:
        raise SystemExit(
            f"{video_path} runs at {source_fps} fps; this tool assumes the model's "
            "own 25 Hz so that a source frame is a model frame"
        )

    boxes = read_boxes(bbox_path)
    activity = read_rttm(rttm_path, source_frames, source_fps)
    tracked = {speaker for frame in boxes.values() for speaker in frame}
    speakers = sorted(tracked | set(activity), key=speaker_key)

    show_first = round(args.start * FPS)
    show_last = show_first + round(args.seconds * FPS)
    if show_last > source_frames:
        raise SystemExit(f"{args.start}+{args.seconds}s runs past the segment")
    # Ten seconds of history in front of the excerpt, read but never shown.
    warm_first = max(0, show_first - CONTEXT_FRAMES)
    warm = show_first - warm_first

    print(f"  {args.video}/{args.segment}: {len(speakers)} speakers {speakers}")
    print(f"  decoding frames {warm_first}..{show_last} ({(show_last - warm_first) / FPS:.1f}s)")
    faces, bbox, visible = decode(video_path, boxes, speakers, warm_first, show_last)

    # The recording's own gain, measured over the whole recording, exactly as
    # the packed loader restores it. Measuring it on the excerpt would let the
    # level drift with what happens to be said in it.
    waveform, _ = load_waveform(video_path, AudioSpec(sample_rate=SAMPLE_RATE, mono=True))
    stats = waveform_stats(waveform)
    audio = apply_normalization(
        waveform[:, warm_first * SAMPLES_PER_FRAME : show_last * SAMPLES_PER_FRAME], stats
    )

    # The annotation is kept past the end of the excerpt. The renderer draws it
    # on both sides of the now-line, and the right-hand side is the future -
    # which for the last seconds of the clip lies beyond the clip itself.
    vad_last = min(source_frames, show_last + round(args.future_sec * FPS))
    vad = np.stack(
        [
            activity.get(speaker, np.zeros(source_frames, dtype=np.uint8))
            for speaker in speakers
        ]
    )[:, show_first:vad_last]

    device = args.device
    encoders, fusion, cfg = load_model(
        args.vap_weights, args.asd_weights, args.muvap_weights, args.config, device
    )
    gvap_window = ProjectionWindow(**cfg["gvap_projection_window"])
    svap_window = ProjectionWindow(**cfg["svap_projection_window"])

    total = faces.shape[1]
    stream = StreamingEncoders(encoders, len(speakers), CONTEXT_FRAMES, device)
    crops = torch.from_numpy(faces).to(device)
    present = torch.from_numpy(visible).to(device)
    audio = audio.to(device)

    shown = total - warm
    gvap = np.zeros((shown, gvap_window.n_classes), dtype=np.float32)
    shift = np.zeros(shown, dtype=np.float32)
    # The two rows p_shift is drawn from, kept alongside it. `shift_scale` is a
    # readout knob, not a weight, so storing these lets it be moved at render
    # time without running the model again.
    rows = np.zeros((shown, 2), dtype=np.float32)
    svap = np.zeros((len(speakers), shown, svap_window.n_bins), dtype=np.float32)

    for index in range(total):
        stream.push(crops[:, index])
        if index < warm:
            continue
        # The window the model reads: the newest `CONTEXT_FRAMES` frames and
        # nothing after them, which is what the live server holds.
        head = index + 1
        first = max(0, head - CONTEXT_FRAMES)
        block = audio[:, first * SAMPLES_PER_FRAME : head * SAMPLES_PER_FRAME]
        mask = present[:, first:head][None]
        with torch.autocast(device, torch.bfloat16):
            vap_stream, asd_stream = stream.run(block[None])
            global_logits, speaker_logits = fusion(vap_stream, asd_stream, mask)

        out = index - warm
        readout = gvap_window.get_shift_hold(
            global_logits[0, 0, -1].float().cpu(), shift_scale=args.shift_scale
        )
        gvap[out] = readout["probs"].numpy()
        rows[out] = readout["p_future"].numpy()
        shift[out] = float(readout["p_shift"])
        svap[:, out] = speaker_logits[0, :, -1].sigmoid().float().cpu().numpy()
        if out % (5 * FPS) == 0:
            print(f"    {out / FPS:5.1f}s / {shown / FPS:.1f}s", flush=True)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        speakers=np.array(speakers),
        gvap=gvap,
        p_shift=shift,
        p_future=rows,
        shift_scale=(
            args.shift_scale
            if args.shift_scale is not None
            else gvap_window.default_shift_scale
        ),
        svap=svap,
        bbox=bbox[:, warm:],
        visible=visible[:, warm:],
        vad=vad,
        codebook=gvap_window.codebook.numpy(),
        gvap_bin_sec=np.array(gvap_window.bin_sec, dtype=np.float32),
        gvap_hist_bins=gvap_window.num_hist_bins,
        svap_bin_sec=np.array(svap_window.bin_sec, dtype=np.float32),
        video=np.array(str(video_path)),
        start_frame=show_first,
        frames=shown,
        fps=FPS,
    )
    print(f"  {output}  ({shown} frames, {len(speakers)} speakers)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="the AVCC annotation root")
    parser.add_argument("--video", required=True)
    parser.add_argument("--segment", required=True)
    parser.add_argument("--start", type=float, required=True, help="seconds into the segment")
    parser.add_argument("--seconds", type=float, default=40.0)
    parser.add_argument("--vap-weights", required=True)
    parser.add_argument("--asd-weights", required=True)
    parser.add_argument(
        "--muvap-weights", required=True,
        help="published MuVAP release, or a training .ckpt",
    )
    parser.add_argument("--config", default="config/yaml/muvap.yaml")
    parser.add_argument("--shift-scale", type=float, default=None)
    parser.add_argument(
        "--future-sec", type=float, default=5.0,
        help="annotation kept past the excerpt, for the right of the now-line",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
