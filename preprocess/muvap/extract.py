"""Run the frozen VAP and ASD modules over a media pack, once.

The fusion module reads two streams that nothing in training changes, so
computing them every epoch is pure waste: the same segment goes through the
same frozen weights and comes back with the same numbers. This writes them out
as an embedding pack, which then trains several times faster and is several
times smaller than the media it came from.

```bash
python -m preprocess.muvap.extract \\
    --pack /data/AVCC/packed/segments.train \\
    --output /data/AVCC/packed/embeddings.segments.train \\
    --vap-weights weights/vap-role-mimi --asd-weights weights/asd-mimi
```

Extraction reads the media pack through the same loader training would and runs
the same `FrozenEncoders` the raw path runs, so the two paths differ in exactly
one way, and it is worth being precise about it. Every stage is causal, so a
frame's embedding depends on how much history was fed when it was encoded.
`--window 0` encodes each segment whole and every frame gets its full context;
a finite window encodes in pieces, each with a history prefix in front of it
that is dropped afterwards and never written. The prefix is what keeps the
difference small - it is there for the reason the training loader has one - but
a cache extracted whole is not bit-identical to a raw run chunked at training
time. It is the better-conditioned of the two, since the raw path never sees
more history than its own chunk's prefix.

The other thing to watch is staleness: the weights that produced a pack are
recorded in its `dataset.json`, and re-extracting is the only way to change
them.
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import yaml
from tqdm import tqdm

from config.setup import init_yaml_config
from data.dataloaders.conversation import PackedConversationDataset
from models.muvap import frozen_pair
from preprocess.muvap.schema import EVENT_KIND, KINDS, MEDIA_SOURCE
from preprocess.muvap.writer import EmbeddingShardWriter


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack", type=Path, required=True, help="media pack to read")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--vap-weights",
        required=True,
        help="published VAP release directory, or a weights.pt beside a config.yaml",
    )
    parser.add_argument("--asd-weights", required=True, help="published ASD release")
    parser.add_argument(
        "--vap-config",
        type=Path,
        help="architecture to rebuild, when the release ships no config.yaml",
    )
    parser.add_argument("--asd-config", type=Path)
    parser.add_argument(
        "--window",
        type=int,
        default=1000,
        help="model frames encoded at once; whole segments if 0",
    )
    parser.add_argument(
        "--context",
        type=int,
        default=250,
        help="history frames encoded in front of each window and then dropped",
    )
    parser.add_argument("--max-shard-gb", type=float, default=1.0)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser


def build_encoders(args):
    vap_cfg = init_yaml_config(args.vap_config)["vap"] if args.vap_config else None
    asd_cfg = init_yaml_config(args.asd_config)["asd"] if args.asd_config else None
    encoders = frozen_pair(args.vap_weights, args.asd_weights, vap_cfg, asd_cfg)
    provenance = {"vap": str(args.vap_weights), "asd": str(args.asd_weights)}
    return encoders.to(args.device).eval(), provenance


def chunks_by_sample(dataset: PackedConversationDataset) -> dict[int, list[int]]:
    """Chunk indices grouped by the sample they came from, in timeline order."""
    grouped: dict[int, list[int]] = {}
    for position, (entry_index, start, _) in enumerate(dataset.chunks):
        grouped.setdefault(entry_index, []).append((start, position))
    return {
        entry: [position for _, position in sorted(spans)]
        for entry, spans in grouped.items()
    }


def encode(encoders, item, device):
    """One chunk through the frozen pair, with the context prefix removed."""
    audio = item["audio"].unsqueeze(0).to(device)
    visual = item["visual"].unsqueeze(0).to(device)
    vap, asd = encoders(audio, visual)

    scored = item["mask"].to(device)
    # The prefix existed only to give the causal stacks their history; the
    # frames it produced are re-encoded as part of the previous window.
    return (
        vap[0, 0, scored].float().cpu().numpy(),
        asd[0, :, scored].float().cpu().numpy(),
    )


def main() -> None:
    args = build_parser().parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"output directory is not empty: {args.output}")

    with (args.pack / "dataset.json").open() as handle:
        pack = yaml.safe_load(handle)
    kind = pack.get("kind")
    if kind not in KINDS:
        raise SystemExit(f"{args.pack} is not a conversation pack")
    if pack.get("source") != MEDIA_SOURCE:
        raise SystemExit(
            f"{args.pack} already holds embeddings; extraction reads a media pack"
        )

    # An event window is short and anchored, and cutting it further would drop
    # the history its last frame is predicted from.
    window = None if kind == EVENT_KIND or args.window <= 0 else args.window
    dataset = PackedConversationDataset(
        args.pack, kind=kind, window=window, context=args.context
    )
    encoders, provenance = build_encoders(args)
    writer = EmbeddingShardWriter(
        args.output,
        kind=kind,
        max_shard_bytes=round(args.max_shard_gb * 1_000_000_000),
        provenance=provenance,
    )

    grouped = chunks_by_sample(dataset)
    for entry_index in tqdm(sorted(grouped), desc="segments"):
        entry = dataset.entries[entry_index]
        vap_parts, asd_parts, vad, visible, bbox = [], [], [], [], []
        for position in grouped[entry_index]:
            item = dataset[position]
            scored = item["mask"]
            vap_block, asd_block = encode(encoders, item, args.device)
            vap_parts.append(vap_block)
            asd_parts.append(asd_block)
            vad.append(item["vad"][:, scored].numpy())
            visible.append(item["visual_mask"][:, scored].numpy())
            bbox.append(item["bbox"][:, scored].numpy())

        writer.add(
            entry.metadata,
            np.concatenate(vap_parts, axis=0),
            np.concatenate(asd_parts, axis=1),
            np.concatenate(vad, axis=1).astype(np.uint8),
            np.concatenate(bbox, axis=1),
            np.concatenate(visible, axis=1),
        )
    writer.close()
    print(f"Wrote {writer.samples_written} {kind} embeddings to {args.output}")


if __name__ == "__main__":
    main()
