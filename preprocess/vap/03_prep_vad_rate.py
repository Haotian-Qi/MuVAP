"""Write the segment VAD again at a different frame rate.

`00_prep_fisher.py` rasterizes the word annotations at `VAD_HZ` and stores one
`.npy` per segment beside the audio. A projection window running at another rate
needs its labels on that grid: resampling the 25 Hz array would inherit its
quantisation, so this rasterizes the annotations again from their original
continuous times.

Only the VAD is rewritten. The audio is rate-independent and stays where it is,
so this touches ~1 GB rather than the 1.12 TB of segments, and it writes to
`seg/vad{rate}/` rather than over `seg/vad/`.

    python preprocess/vap/03_prep_vad_rate.py --frame-hz 12.5 \
        --fisher-path /path/to/fisher --splits train val test
"""

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

from preprocess.vap.utils import SAMPLE_RATE

SEG_SEC = 30
CONTEXT_SEC = 2.0


def rasterize(words_dir, conversation, frames, frame_hz):
    """`[2, frames]` activity for a whole conversation, from the word times."""
    vad = np.zeros((2, frames), dtype=np.float32)
    paths = sorted(words_dir.glob(f"{conversation}_*_words.txt"))
    if len(paths) != 2:
        raise FileNotFoundError(
            f"expected two *_words.txt for {conversation} in {words_dir}, found {len(paths)}"
        )
    for speaker, path in enumerate(paths):
        with open(path) as handle:
            for line in handle:
                if not line.strip():
                    continue
                begin, end, *_ = line.split()
                first = max(0, int(round(float(begin) * frame_hz)))
                last = min(frames, int(round(float(end) * frame_hz)))
                if last > first:
                    vad[speaker, first:last] = 1.0
    return vad


def segment_window(vad, start_sec, frame_hz, context_frames):
    """The stored slice for one segment: the segment plus context, zero-padded.

    Mirrors `extract_and_save_chunks`, so a reader can treat these files exactly
    as it treats the 25 Hz ones.
    """
    seg_frames = int(round(SEG_SEC * frame_hz))
    start = int(round(start_sec * frame_hz))
    end = start + seg_frames
    lo, hi = max(0, start - context_frames), min(vad.shape[-1], end + context_frames)
    chunk = vad[:, lo:hi]
    pad_left = max(0, context_frames - start)
    pad_right = max(0, (end + context_frames) - vad.shape[-1])
    if pad_left or pad_right:
        chunk = np.pad(chunk, ((0, 0), (pad_left, pad_right)))
    return chunk


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fisher-path", required=True)
    parser.add_argument("--frame-hz", type=float, required=True)
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    root = Path(args.fisher_path)
    label = f"{args.frame_hz:g}".replace(".", "_")
    context_frames = int(round(CONTEXT_SEC * args.frame_hz))

    # One rasterization per conversation, then slice every segment out of it.
    by_conversation = defaultdict(list)
    for split in args.splits:
        for line in open(root / f"{split}.txt"):
            if not line.strip():
                continue
            part, group, conversation, seg_id, start, *_ = line.split()
            by_conversation[(part, group, conversation)].append((int(seg_id), float(start)))

    written = skipped = 0
    for index, ((part, group, conversation), segments) in enumerate(sorted(by_conversation.items()), 1):
        out = root / part / "seg" / f"vad{label}" / group / conversation
        out.mkdir(parents=True, exist_ok=True)
        targets = [(s, p) for s, p in ((s, out / f"{s}.npy") for s, _ in segments)
                   if args.overwrite or not p.exists()]
        if not targets:
            skipped += len(segments)
            continue
        last_start = max(start for _, start in segments)
        frames = int(round((last_start + SEG_SEC + CONTEXT_SEC) * args.frame_hz)) + 1
        vad = rasterize(root / part / "regions_vap" / group, conversation, frames, args.frame_hz)
        starts = dict(segments)
        for seg_id, path in targets:
            np.save(path, segment_window(vad, starts[seg_id], args.frame_hz, context_frames))
            written += 1
        if index % 500 == 0:
            print(f"  {index}/{len(by_conversation)} conversations, {written} written", flush=True)
    print(f"done: {written} written, {skipped} already present, "
          f"{len(by_conversation)} conversations -> seg/vad{label}/")


if __name__ == "__main__":
    main()
