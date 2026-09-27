"""Score the VAP module alone on a MuVAP event pack.

The unimodal baseline the fusion has to beat: the same turn-taking question,
asked of audio only, decoded the same zero-shot way. The VAP release reads the
event's mixed recording, its own head produces the role-relative class at the
decision frame, and `get_shift_hold` turns that into hold or shift exactly as
`MuVAPTask` does with the global head.

The zero-shot readout is fitted to nothing. Any gap between it and the fusion
is what the visual channel and the multiparty structure are worth on this
corpus.

`--fit-pack` adds a second, fitted reading: a logistic regression over the same
decision frame's embedding, trained on held-out events and scored here. It is
the unimodal counterpart of `MuVAPTask`'s probe, so the two report the same
pair of numbers and the benchmark table has no empty cell.

```bash
python tools/vap_baseline.py --pack /data/AVCC/packed/events.test \\
    --weights weights/vap-role-mimi \\
    --fit-pack /data/AVCC/packed/events.train
```
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.dataloaders.conversation import (  # noqa: E402
    ConversationCollator,
    PackedConversationDataset,
    SpeakerFrameBudgetSampler,
)
from models.release import load_weights, module_config  # noqa: E402
from models.vap import build_vap  # noqa: E402
from projection_window import ProjectionWindow  # noqa: E402
from tasks.setup import LogisticProber  # noqa: E402
from tasks.vap_task import SHIFT_SCALES  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack", type=Path, required=True, help="an event media pack")
    parser.add_argument("--weights", required=True, help="published VAP release")
    parser.add_argument(
        "--fit-pack",
        type=Path,
        help="event pack the logistic probe fits on; the train split's, so "
        "nothing it is scored on was ever seen. Omitted, only the zero-shot "
        "readout is reported",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--dump",
        type=Path,
        help="write the per-event records here, for scoring done elsewhere",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    weights, cfg = module_config(args.weights, "vap")
    model = load_weights(build_vap(cfg), weights).to(args.device).eval()
    window = ProjectionWindow(**cfg["projection_window"])
    print(f"VAP release: {weights}\n  {window}\n")

    def run(pack):
        """Every event in one pack, read at its decision frame."""
        dataset = PackedConversationDataset(pack, kind="events")
        sampler = SpeakerFrameBudgetSampler(
            dataset.chunk_lengths(), dataset.chunk_speakers(),
            frame_budget=10**9, batch_size=args.batch_size, shuffle=False,
            drop_last=False,
        )
        collate = ConversationCollator(pad=True)

        collected = []
        for batch_indices in sampler:
            batch = collate([dataset[index] for index in batch_indices])
            audio = batch["audio"].to(args.device)
            with torch.no_grad(), torch.autocast(args.device, torch.bfloat16):
                logits, embedding = model(audio, return_embeddings=True)
            last = batch["mask"].bool().sum(dim=1) - 1
            for row, (event, metadata) in enumerate(
                zip(batch["event"], batch["metadata"])
            ):
                end = int(last[row])
                collected.append({
                    # Joins a prediction back to its row of the benchmark file.
                    "sample_id": metadata["sample_id"],
                    "logits": logits[row, end].float().cpu(),
                    # The frame the class is read from, so the probe is an
                    # upper bound on what that same representation carries.
                    "embedding": embedding[row, end].float().cpu().numpy(),
                    "label": int("SHIFT" in str(event["label"]).upper()),
                    # The same cell the fusion is reported in: speakers the
                    # recording tracks, so the two breakdowns line up.
                    "speakers": int(sum(metadata["tracked"])) or metadata["speakers"],
                })
        return collected

    def by_cell(collected):
        grouped = defaultdict(list)
        for output in collected:
            grouped["all"].append(output)
            grouped[f"{output['speakers']}spk"].append(output)
        return grouped

    outputs = run(args.pack)
    cells = by_cell(outputs)

    fitting = {}
    fit_outputs = []
    if args.fit_pack:
        print(f"fitting pool: {args.fit_pack}")
        fit_outputs = run(args.fit_pack)
        fitting = by_cell(fit_outputs)
    if args.dump:
        torch.save({"test": outputs, "fit": fit_outputs}, args.dump)
        print(f"wrote {len(outputs)} test and {len(fit_outputs)} fit records to {args.dump}")

    header = (f"{'cell':8}{'events':>8}{'shift':>8}{'f1_macro':>11}{'bacc':>9}"
              f"{'best scale':>12}{'f1 @ best':>11}")
    if fitting:
        header += f"{'probe f1':>11}{'probe bacc':>12}"
    print(header)
    for name in sorted(cells, key=lambda k: (k != "all", k)):
        samples = cells[name]
        stacked = torch.stack([s["logits"] for s in samples])
        labels = np.asarray([s["label"] for s in samples])
        prediction = window.get_shift_hold(stacked)["pred"].numpy()
        swept = {
            scale: f1_score(
                labels,
                window.get_shift_hold(stacked, shift_scale=scale)["pred"].numpy(),
                average="macro",
            )
            for scale in SHIFT_SCALES
        }
        best = max(swept, key=swept.get)
        line = (f"{name:8}{len(samples):>8}{labels.mean():>8.2f}"
                f"{f1_score(labels, prediction, average='macro'):>11.4f}"
                f"{balanced_accuracy_score(labels, prediction):>9.4f}"
                f"{best:>12.1f}{swept[best]:>11.4f}")
        pool = fitting.get(name)
        if pool:
            # class_weight="balanced" puts the boundary where a 50/50 test set
            # wants it, so a naturally skewed fitting pool needs no downsampling.
            probe = LogisticProber().fit_predict(
                np.stack([sample["embedding"] for sample in pool]),
                np.asarray([sample["label"] for sample in pool]),
                np.stack([sample["embedding"] for sample in samples]),
            )
            if probe is not None:
                line += (f"{f1_score(labels, probe, average='macro'):>11.4f}"
                         f"{balanced_accuracy_score(labels, probe):>12.4f}")
        print(line)

    # Per-class, for the pooled cell: the macro average hides which way the
    # decision is skewed, and on a 50/50 set that skew is the whole story.
    samples = cells["all"]
    stacked = torch.stack([s["logits"] for s in samples])
    labels = np.asarray([s["label"] for s in samples])
    names = ["hold", "shift"]
    zero_shot = window.get_shift_hold(stacked)["pred"].numpy()
    print("\n=== all: zero-shot, per class ===")
    print(classification_report(labels, zero_shot, target_names=names, digits=4))
    print("confusion (rows=true hold/shift, cols=pred):")
    print(confusion_matrix(labels, zero_shot))
    pool = fitting.get("all")
    if pool:
        probe = LogisticProber().fit_predict(
            np.stack([sample["embedding"] for sample in pool]),
            np.asarray([sample["label"] for sample in pool]),
            np.stack([sample["embedding"] for sample in samples]),
        )
        if probe is not None:
            print("=== all: probe, per class ===")
            print(classification_report(labels, probe, target_names=names, digits=4))
            print("confusion:")
            print(confusion_matrix(labels, probe))


if __name__ == "__main__":
    main()
