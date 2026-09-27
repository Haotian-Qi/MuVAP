"""Score every segment of one recording, frame by frame, to choose a demo clip.

Picking a clip by eye picks the clip that reads well, which is not the same as
the clip the model does well on. This runs the trained fusion over a whole
recording and writes what it cost at every frame - the task's own two losses,
nothing reweighted - so a clip can be chosen on the model's own terms and the
choice can be defended with a number.

Frames are scored in 30 s windows stepping 20 s, each carrying the 10 s before
it as unscored context. Every scored frame therefore has at least the ten
seconds of history the live demo runs on, and no frame is scored twice.

The `p_future` rows travel out too. They are what `shift_scale` reweights, so
sweeping that knob afterwards is arithmetic on this file rather than another
pass over the recording - see `tools/choose_clip.py`.
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.media import AudioSpec, apply_normalization, load_waveform, waveform_stats
from models.muvap import frozen_pair
from preprocess.muvap.avcc import read_boxes, read_rttm, segment_paths, speaker_key
from projection_window import ProjectionWindow
from tools.demo_infer import (
    CONTEXT_FRAMES,
    FPS,
    SAMPLES_PER_FRAME,
    SAMPLE_RATE,
    decode,
    load_model,
    resolve,
)

#: The training grid, reused here because it is what the numbers mean. A window
#: is scored over its last `WINDOW - CONTEXT_FRAMES` frames and reads the rest
#: as history, so consecutive windows tile the segment exactly once.
WINDOW = 750
STEP = WINDOW - CONTEXT_FRAMES


def segments_of(root: Path, video_id: str) -> list[str]:
    return [segment for _, segment in segment_paths(root, video_id)]


@torch.no_grad()
def score_segment(paths, encoders, fusion, gvap_window, svap_window, device):
    """Per-frame loss and hold/shift rows for one whole segment."""
    video_path, bbox_path, rttm_path = paths
    capture = cv2.VideoCapture(str(video_path))
    source_fps = float(capture.get(cv2.CAP_PROP_FPS))
    frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    if abs(source_fps - FPS) > 1e-6:
        raise ValueError(f"{video_path} is not at {FPS} fps")

    boxes = read_boxes(bbox_path)
    activity = read_rttm(rttm_path, frames, source_fps)
    tracked = {speaker for frame in boxes.values() for speaker in frame}
    speakers = sorted(tracked | set(activity), key=speaker_key)
    if len(speakers) < 2:
        return None

    faces, bbox, visible = decode(video_path, boxes, speakers, 0, frames)
    frames = faces.shape[1]

    waveform, _ = load_waveform(video_path, AudioSpec(sample_rate=SAMPLE_RATE, mono=True))
    audio = apply_normalization(waveform, waveform_stats(waveform))
    audio = audio[:, : frames * SAMPLES_PER_FRAME]
    if audio.shape[-1] < frames * SAMPLES_PER_FRAME:
        audio = F.pad(audio, (0, frames * SAMPLES_PER_FRAME - audio.shape[-1]))

    vad = np.stack(
        [activity.get(name, np.zeros(frames, dtype=np.uint8))[:frames] for name in speakers]
    )
    # Labels come off the whole segment, never off a window: the projection
    # reaches 1.4 s either side of a frame, and a window-local label would be
    # wrong at both of its edges.
    truth = torch.from_numpy(vad).float()[None]
    gvap_gt = gvap_window.get_labels(truth)[0]
    svap_gt = svap_window.get_labels(truth.transpose(0, 1))[:, :]

    gvap_loss = np.full(frames, np.nan, dtype=np.float32)
    svap_loss = np.full(frames, np.nan, dtype=np.float32)
    rows = np.zeros((frames, 2), dtype=np.float32)

    crops = torch.from_numpy(faces)
    present = torch.from_numpy(visible)
    for start in range(0, frames, STEP):
        stop = min(start + WINDOW, frames)
        if stop - start < FPS:
            break
        scored = min(CONTEXT_FRAMES, start)          # the unscored history prefix
        window_audio = audio[:, start * SAMPLES_PER_FRAME : stop * SAMPLES_PER_FRAME]
        visual = crops[:, start:stop].to(device)[None]
        mask = present[:, start:stop].to(device)[None]
        with torch.autocast(device, torch.bfloat16):
            vap_stream, asd_stream = encoders(window_audio.to(device)[None], visual)
            global_logits, speaker_logits = fusion(vap_stream, asd_stream, mask)

        global_logits = global_logits[0, 0].float().cpu()
        speaker_logits = speaker_logits[0].float().cpu()
        keep = slice(scored, stop - start)
        out = slice(start + scored, stop)

        gvap_loss[out] = F.cross_entropy(
            global_logits[keep], gvap_gt[out], reduction="none"
        ).numpy()
        # Per frame, averaged over the bins of every face actually on screen -
        # the same `valid` rule the task scores with, read one frame at a time.
        per_bin = F.binary_cross_entropy_with_logits(
            speaker_logits[:, keep], svap_gt[:, out].float(), reduction="none"
        )
        seen = present[:, out].float()[:, :, None].expand_as(per_bin)
        total = (per_bin * seen).sum(dim=(0, 2))
        count = seen.sum(dim=(0, 2)).clamp_min(1)
        svap_loss[out] = (total / count).numpy()
        rows[out] = gvap_window.get_probs(global_logits[keep])["p_future"].numpy()

    return {
        "speakers": np.array(speakers),
        "gvap_loss": gvap_loss,
        "svap_loss": svap_loss,
        "p_future": rows,
        "vad": vad.astype(np.uint8),
        "visible": visible,
        "frames": frames,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--video", required=True)
    parser.add_argument("--vap-weights", required=True)
    parser.add_argument("--asd-weights", required=True)
    parser.add_argument(
        "--muvap-weights", required=True,
        help="published MuVAP release, or a training .ckpt",
    )
    parser.add_argument("--config", default="config/yaml/muvap.yaml")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", required=True, help="directory for the per-segment npz")
    args = parser.parse_args()

    root = Path(args.root)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    encoders, fusion, cfg = load_model(
        args.vap_weights, args.asd_weights, args.muvap_weights, args.config, args.device
    )
    gvap_window = ProjectionWindow(**cfg["gvap_projection_window"])
    svap_window = ProjectionWindow(**cfg["svap_projection_window"])

    for segment in segments_of(root, args.video):
        target = out / f"{args.video}+{segment}.npz"
        if target.exists():
            print(f"  {segment}: already scored")
            continue
        try:
            paths = resolve(root, args.video, segment)
        except FileNotFoundError as error:
            print(f"  {segment}: skipped ({error})")
            continue
        if not paths[0].exists():
            print(f"  {segment}: no recording")
            continue
        result = score_segment(
            paths, encoders, fusion, gvap_window, svap_window, args.device
        )
        if result is None:
            print(f"  {segment}: fewer than two speakers")
            continue
        np.savez_compressed(target, **result)
        total = np.nanmean(result["gvap_loss"]) + np.nanmean(result["svap_loss"])
        print(
            f"  {segment}: {result['frames'] / FPS:6.1f}s  "
            f"gvap {np.nanmean(result['gvap_loss']):.3f}  "
            f"svap {np.nanmean(result['svap_loss']):.3f}  total {total:.3f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
