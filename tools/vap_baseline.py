"""Score the VAP module alone on a MuVAP event pack.

The unimodal baseline the fusion has to beat: the same turn-taking question,
asked of audio only, decoded the same zero-shot way. The VAP release reads the
event's mixed recording, its own head produces the role-relative class at the
decision frame, and `get_shift_hold` turns that into hold or shift exactly as
`MuVAPTask` does with the global head.

Nothing here is fitted. Any gap between this and the fusion is what the visual
channel and the multiparty structure are worth on this corpus.

```bash
python tools/vap_baseline.py --pack /data/AVCC/packed/events.test \\
    --weights weights/vap-role-cpc
```
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.dataloaders.conversation import (  # noqa: E402
    ConversationCollator,
    PackedConversationDataset,
    SpeakerFrameBudgetSampler,
)
from models.release import load_weights, module_config  # noqa: E402
from models.vap import build_vap  # noqa: E402
from projection_window import ProjectionWindow  # noqa: E402
from tasks.vap_task import SHIFT_SCALES  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack", type=Path, required=True, help="an event media pack")
    parser.add_argument("--weights", required=True, help="published VAP release")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    weights, cfg = module_config(args.weights, "vap")
    model = load_weights(build_vap(cfg), weights).to(args.device).eval()
    window = ProjectionWindow(**cfg["projection_window"])
    print(f"VAP release: {weights}\n  {window}\n")

    dataset = PackedConversationDataset(args.pack, kind="events")
    sampler = SpeakerFrameBudgetSampler(
        dataset.chunk_lengths(), dataset.chunk_speakers(),
        frame_budget=10**9, batch_size=args.batch_size, shuffle=False, drop_last=False,
    )
    collate = ConversationCollator(pad=True)

    outputs = []
    for batch_indices in sampler:
        batch = collate([dataset[index] for index in batch_indices])
        audio = batch["audio"].to(args.device)
        with torch.no_grad(), torch.autocast(args.device, torch.bfloat16):
            logits = model(audio)
        last = batch["mask"].bool().sum(dim=1) - 1
        for row, (event, metadata) in enumerate(zip(batch["event"], batch["metadata"])):
            outputs.append({
                "logits": logits[row, int(last[row])].float().cpu(),
                "label": int("SHIFT" in str(event["label"]).upper()),
                # The same cell the fusion is reported in: speakers the
                # recording tracks, so the two breakdowns line up.
                "speakers": int(sum(metadata["tracked"])) or metadata["speakers"],
            })

    cells = defaultdict(list)
    for output in outputs:
        cells["all"].append(output)
        cells[f"{output['speakers']}spk"].append(output)

    print(f"{'cell':8}{'events':>8}{'shift':>8}{'f1_macro':>11}{'bacc':>9}"
          f"{'best scale':>12}{'f1 @ best':>11}")
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
        print(f"{name:8}{len(samples):>8}{labels.mean():>8.2f}"
              f"{f1_score(labels, prediction, average='macro'):>11.4f}"
              f"{balanced_accuracy_score(labels, prediction):>9.4f}"
              f"{best:>12.1f}{swept[best]:>11.4f}")


if __name__ == "__main__":
    main()
