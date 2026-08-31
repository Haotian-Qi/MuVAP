"""Build the canonical fast-training format for AVA, WASD, and AVCC."""

import argparse
from pathlib import Path

from preprocess.asd.adapters import clip_directory_records, manifest_records
from preprocess.asd.writer import write_records


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=["ava", "wasd", "msdwild", "avcc"])
    parser.add_argument(
        "--loader", type=Path, required=True, help="source TSV/manifest"
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="defaults to <dataset root>/packed/<split>, beside the source media",
    )
    parser.add_argument("--visual-root", type=Path)
    parser.add_argument("--audio-root", type=Path)
    parser.add_argument("--root", type=Path, help="base for AVCC manifest paths")
    parser.add_argument("--split", default="train")
    parser.add_argument("--audio-sample-rate", type=int, default=16_000)
    parser.add_argument("--max-shard-gb", type=float, default=1.0)
    parser.add_argument(
        "--skip-missing",
        action="store_true",
        help="development subsets only; production preprocessing should fail on missing files",
    )
    return parser


def require_roots(args: argparse.Namespace) -> None:
    if args.visual_root is None or args.audio_root is None:
        raise ValueError(f"{args.dataset} requires --visual-root and --audio-root")


def default_output(args: argparse.Namespace) -> Path:
    """Keep each corpus's pack beside that corpus.

    `<root>/clips_videos/<split>` becomes `<root>/packed/<split>`, so a dataset
    stays self-contained and can be copied to fast local storage on its own.
    """
    if args.visual_root is not None:
        return args.visual_root.parent / "packed" / args.split
    if args.root is not None:
        return args.root / "packed" / args.split
    raise ValueError("--output is required when no dataset root is given")


def main() -> None:
    args = build_parser().parse_args()
    if args.output is None:
        args.output = default_output(args)
        print(f"packing into {args.output}")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"output directory is not empty: {args.output}")

    if args.dataset in ("ava", "wasd"):
        require_roots(args)
        records = clip_directory_records(
            args.loader,
            args.visual_root,
            args.audio_root,
            args.split,
            dataset=args.dataset,
            skip_missing=args.skip_missing,
        )
    elif args.dataset == "msdwild":
        raise SystemExit(
            "MSDWild is packed from raw media by preprocess/asd/prep_msdwild.py, "
            "which needs all.rttm for labels; there is no loader TSV to read here"
        )
    else:
        records = manifest_records(
            args.loader,
            dataset="avcc",
            root=args.root,
            default_npy_audio_sample_rate=args.audio_sample_rate,
        )

    count = write_records(
        records,
        args.output,
        max_shard_bytes=round(args.max_shard_gb * 1_000_000_000),
    )
    print(f"Wrote {count} {args.dataset} samples to {args.output}")


if __name__ == "__main__":
    main()
