"""Pack AVCC conversations into the memory-mapped format MuVAP trains on.

Two pack kinds come out of the same source layout:

```bash
# whole segments, tiled into chunks by the loader at training time
python -m preprocess.muvap.prepare segments --root /data/AVCC --split train

# one window per annotated turn event, what the numbers are computed on
python -m preprocess.muvap.prepare events --root /data/AVCC --split test \\
    --events /data/AVCC/test_events.txt
```

A segment pack stores whole segments because a pack is a decoded mirror of the
corpus: choosing a window is a training decision, and
`data.dataloaders.conversation` makes it - cutting the grid to disk instead
would duplicate every overlapped frame and read no faster. An event pack is the
exception: its windows are anchored on the annotation rather than tiling, so
there is nothing for the loader to decide.
"""

import argparse
from pathlib import Path

from preprocess.muvap.avcc import (
    event_window,
    open_segment,
    read_events,
    read_splits,
    segment_paths,
)
from preprocess.muvap.schema import EVENT_KIND, SEGMENT_KIND
from preprocess.muvap.writer import ConversationShardWriter


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=[SEGMENT_KIND, EVENT_KIND])
    parser.add_argument("--root", type=Path, required=True, help="AVCC dataset root")
    parser.add_argument(
        "--media-root",
        type=Path,
        help="directory holding <video>/<segment>.mp4; found beside --root if omitted",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="defaults to <root>/packed/<kind>.<split>, beside the source media",
    )
    parser.add_argument("--split", default="train")
    parser.add_argument(
        "--only",
        action="append",
        default=[],
        metavar="VIDEO[/SEGMENT]",
        help="pack just these videos or segments, for inspecting one of them; "
        "repeatable",
    )
    parser.add_argument(
        "--splits",
        type=Path,
        help="split assignment file; defaults to <root>/splits.txt",
    )
    parser.add_argument(
        "--events",
        type=Path,
        help="turn-event file; required for an event pack",
    )
    parser.add_argument(
        "--window-sec",
        type=float,
        default=10.0,
        help="most history an event window carries; one closer than this to the "
        "start of its segment keeps what it has rather than being dropped",
    )
    parser.add_argument("--max-shard-gb", type=float, default=1.0)
    parser.add_argument(
        "--skip-missing",
        action="store_true",
        help="development subsets only; production preprocessing should fail on "
        "missing files",
    )
    return parser


def videos_in_split(root: Path, splits_path: Path | None, split: str) -> list[str]:
    splits = read_splits(splits_path or root / "splits.txt")
    chosen = [video for video, name in splits.items() if name == split]
    if not chosen:
        raise SystemExit(
            f"no video is assigned to split {split!r}; "
            f"the file names {sorted(set(splits.values()))}"
        )
    return sorted(chosen)


def wanted_segment(args, video_id: str, segment_id: str) -> bool:
    """Whether `--only` selects this segment; everything passes when unset."""
    if not args.only:
        return True
    return any(
        selector == video_id or selector == f"{video_id}/{segment_id}"
        for selector in args.only
    )


def pack_segments(args, writer, videos) -> int:
    written = 0
    for video_id in videos:
        for video, segment_id in segment_paths(args.root, video_id):
            if not wanted_segment(args, video, segment_id):
                continue
            segment = load(args, video, segment_id)
            if segment is None:
                continue
            writer.add(*segment.whole())
            segment.close()
            written += 1
            print(
                f"{video}/{segment_id}: {len(segment.speaker_ids)} speakers, "
                f"{segment.source_frames} frames at {segment.source_fps:g} fps"
            )
    return written


def pack_events(args, writer, videos) -> int:
    if args.events is None:
        raise SystemExit("an event pack needs --events")
    annotated = read_events(args.events)
    wanted = sorted(key for key in annotated if key[0] in set(videos))
    if not wanted:
        raise SystemExit(
            f"{args.events} holds no event for any video in split {args.split!r}"
        )

    written = dropped = 0
    shortest = 10**9
    for video_id, segment_id in wanted:
        if not wanted_segment(args, video_id, segment_id):
            continue
        segment = load(args, video_id, segment_id)
        if segment is None:
            continue
        for index, event in enumerate(annotated[(video_id, segment_id)]):
            span = event_window(
                event["gap_start"],
                segment.source_fps,
                segment.source_frames,
                args.window_sec,
            )
            annotation = event["annotation"]
            # An event naming a speaker this segment never rosters cannot be
            # scored: the model would have no row to answer with.
            unknown = {annotation.previous, annotation.following}.difference(
                segment.speaker_ids
            )
            if span is None or unknown:
                dropped += 1
                continue
            frames = span[1] - span[0]
            shortest = min(shortest, frames)
            sample_id = f"{video_id}+{segment_id}+{index:04d}"
            writer.add(
                *segment.window(span[0], span[1], sample_id), event=annotation
            )
            written += 1
        segment.close()
    if dropped:
        print(f"dropped {dropped} event(s) lying off their segment or naming an unknown speaker")
    if written:
        full = round(args.window_sec * 25)
        print(f"shortest window {shortest} frames against a full {full}")
    return written


def load(args, video_id: str, segment_id: str):
    try:
        # The scratch array for the decoded faces lands beside the pack, which
        # is the disk with room for it.
        return open_segment(
            args.root, video_id, segment_id, args.media_root, work_dir=args.output
        )
    except FileNotFoundError as error:
        if not args.skip_missing:
            raise
        print(f"skipping {video_id}/{segment_id}: {error}")
        return None


def main() -> None:
    args = build_parser().parse_args()
    if args.output is None:
        args.output = args.root / "packed" / f"{args.kind}.{args.split}"
        print(f"packing into {args.output}")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"output directory is not empty: {args.output}")

    videos = videos_in_split(args.root, args.splits, args.split)
    writer = ConversationShardWriter(
        args.output,
        kind=args.kind,
        max_shard_bytes=round(args.max_shard_gb * 1_000_000_000),
    )
    pack = pack_segments if args.kind == SEGMENT_KIND else pack_events
    count = pack(args, writer, videos)
    writer.close()
    print(f"Wrote {count} AVCC {args.kind} to {args.output}")


if __name__ == "__main__":
    main()
