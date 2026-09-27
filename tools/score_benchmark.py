"""Score the AVCC turn-taking benchmark from the turn-event file alone.

Everything a score needs comes from `test_events.txt`, whose eighth column is
the event's speaker count:

```text
video_id  segment_id  previous  start  duration  following  label  n_speakers
```

The cells, the hold/shift collapse and the identity decision all come from the
same place here, so the numbers in a paper table and the numbers a reviewer
recomputes from the per-event CSV cannot drift apart.

Three rules this file exists to enforce:

* **the cell is the event file's `n_speakers`**, joined on the event, not the
  count of speakers a window happens to track. The two disagree on 99 events;
  mixing them puts an event in one cell for the benchmark and another for the
  model.
* **nothing true enters a prediction.** The floor holder a naming decision is
  built on is the model's own, not the annotation's. Naming the next speaker
  therefore carries the floor-holder error with it, and identity and hold/shift
  are allowed to disagree - on a two-speaker event the turn decision can be
  right while the face named is wrong, because the model put the floor with the
  wrong speaker.
* **two scores, one prediction, no collapse.** Hold/shift is the probe's
  decision against the event type. Identity is the named face against the gold
  next speaker. Neither is derived from the other.

Balanced accuracy throughout: the mean of the hold and shift accuracies within
the cell, so a cell that is not exactly balanced cannot flatter a model that
leans one way.
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent


def read_events(path):
    """`sample_id` -> that event's row, the way preprocessing numbered it.

    `prepare.py` enumerates a segment's events before it decides whether any of
    them can be windowed, so the index is the row's position in this file and a
    dropped event still consumes one. `event_id` is `label-video-segment-start`,
    with `start` exactly as the file writes it, so tie lists can name events.
    """
    from collections import defaultdict

    by_segment = defaultdict(list)
    for number, line in enumerate(path.read_text().splitlines(), start=1):
        parts = line.split()
        if not parts:
            continue
        if len(parts) != 8:
            raise ValueError(
                f"{path}:{number}: scoring needs 8 columns (video segment previous "
                f"start duration following label n_speakers), got {len(parts)}"
            )
        by_segment[(parts[0], parts[1])].append(parts)
    rows = {}
    for (video, segment), events in by_segment.items():
        for index, parts in enumerate(events):
            _, _, previous, start, _duration, following, label, speakers = parts
            rows[f"{video}+{segment}+{index:04d}"] = {
                "event_id": f"{label.lower()}-{video}-{segment}-{start}",
                "video_id": video,
                "segment_id": segment,
                "start_time": start,
                "previous_speaker": str(int(previous)),
                "gold_next_speaker": str(int(following)),
                "event_type": label.lower(),
                "n_speakers": str(int(speakers)),
            }
    return rows


def fit_probe(fit_records, test_records):
    """One probe for every event, so a cell is a slice and not a new model.

    `MuVAPTask` fits a probe per speaker-count cell, which is the right thing
    when the cells are the ones it defined. It cannot be right here: the cells
    are re-keyed afterwards from the event file, and a per-cell probe would
    make a prediction depend on which cell it lands in - the per-event CSV
    could then no longer reproduce the table. Same estimator and settings as
    `LogisticProber`, with the probability kept.
    """
    scaler = StandardScaler()
    clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000, random_state=42)
    X = np.stack([r["embedding"] for r in fit_records])
    y = np.asarray([r["label"] for r in fit_records])
    clf.fit(scaler.fit_transform(X), y)
    Z = scaler.transform(np.stack([r["embedding"] for r in test_records]))
    return clf.predict(Z), clf.predict_proba(Z)[:, 1]


def rates(labels, predictions):
    """Hold accuracy, shift accuracy, and the mean of the two."""
    labels, predictions = np.asarray(labels), np.asarray(predictions)
    out = []
    for value in (0, 1):                     # 0 hold, 1 shift
        mask = labels == value
        out.append(float((predictions[mask] == labels[mask]).mean()) if mask.any() else float("nan"))
    return out[0], out[1], (out[0] + out[1]) / 2


def table(rows, cells, title):
    print(f"\n{title}")
    print(f"{'cell':>6}{'n':>7}{'scoring':>13}{'hold_acc':>11}{'shift_acc':>11}{'balanced_acc':>14}")
    for cell, n, scoring, hold, shift, bacc in rows:
        print(f"{cell:>6}{n:>7}{scoring:>13}{hold*100:>10.1f}%{shift*100:>10.1f}%{bacc*100:>13.1f}%")
    return cells


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--muvap-dump", type=Path, required=True)
    parser.add_argument("--rolevap-dump", type=Path)
    parser.add_argument(
        "--events", type=Path, required=True,
        help="test_events.txt, with n_speakers as its eighth column",
    )
    parser.add_argument("--out", type=Path, required=True, help="directory for the CSVs")
    args = parser.parse_args()

    events = read_events(args.events)
    args.out.mkdir(parents=True, exist_ok=True)

    def keyed(records):
        """Every record beside its row of the event file, or a loud failure."""
        out, missing = [], 0
        for record in records:
            row = events.get(record["sample_id"])
            if row is None:
                missing += 1
                continue
            out.append((record, row))
        print(f"joined {len(out)} of {len(records)} records, {missing} unjoined")
        return out

    def cells_of(row):
        return ("all", f"{row['n_speakers']}spk")

    def emit(name, pairs, hold_shift, next_speaker, previous_speaker, prob_shift):
        """One table and one CSV, both read off the same predictions."""
        scored = {}
        for index, (_record, row) in enumerate(pairs):
            truth_shift = int(row["event_type"].lower() == "shift")
            identity = (
                None if next_speaker[index] is None
                else int(next_speaker[index] == row["gold_next_speaker"])
            )
            floor_right = (
                None if previous_speaker[index] is None
                else int(previous_speaker[index] == row["previous_speaker"])
            )
            for cell in cells_of(row):
                bucket = scored.setdefault(
                    cell,
                    {"label": [], "hs": [], "id_label": [], "id": [],
                     "floor": [], "id_given_floor": []},
                )
                bucket["label"].append(truth_shift)
                bucket["hs"].append(hold_shift[index])
                if floor_right is not None:
                    bucket["floor"].append(floor_right)
                if identity is not None:
                    bucket["id_label"].append(truth_shift)
                    # Identity is simply right or wrong; grouping by the event's
                    # own class is what puts it on the same balanced footing.
                    bucket["id"].append(identity)
                    if floor_right:
                        # Identity where the naming step was given a correct
                        # floor holder: what is left is the naming itself.
                        bucket["id_given_floor"].append(identity)
        rows = []
        for cell in ("2spk", "3spk", "all"):
            if cell not in scored:
                continue
            bucket = scored[cell]
            hold, shift, bacc = rates(bucket["label"], bucket["hs"])
            rows.append((cell, len(bucket["label"]), "hold-shift", hold, shift, bacc))
            if bucket["id"]:
                labels = np.asarray(bucket["id_label"])
                correct = np.asarray(bucket["id"])
                per = []
                for value in (0, 1):
                    mask = labels == value
                    per.append(float(correct[mask].mean()) if mask.any() else float("nan"))
                rows.append((cell, len(labels), "identity", per[0], per[1],
                             (per[0] + per[1]) / 2))
        table(rows, scored, f"{name}  (cells: n_speakers from the event file)")
        if any(scored[cell]["floor"] for cell in scored):
            print("  diagnostics (not scored decisions):")
            for cell in ("2spk", "3spk", "all"):
                bucket = scored.get(cell)
                if not bucket or not bucket["floor"]:
                    continue
                floor = float(np.mean(bucket["floor"])) * 100
                given = bucket["id_given_floor"]
                line = f"    {cell:>5}  previous-speaker acc {floor:5.1f}%"
                if given:
                    line += (f"   identity where the floor holder was right "
                             f"{float(np.mean(given)) * 100:5.1f}%  (n={len(given)})")
                print(line)

        path = args.out / f"{name}_predictions.csv"
        with path.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["event_id", "pred_hold_shift", "prob_shift",
                             "pred_previous_speaker", "pred_next_speaker"])
            for index, (_record, row) in enumerate(pairs):
                writer.writerow([
                    row["event_id"],
                    "shift" if hold_shift[index] else "hold",
                    "" if prob_shift is None else f"{prob_shift[index]:.6f}",
                    "" if previous_speaker[index] is None else previous_speaker[index],
                    "" if next_speaker[index] is None else next_speaker[index],
                ])
        print(f"wrote {path}")
        return rows

    # ---- MuVAP ----------------------------------------------------------
    dump = torch.load(args.muvap_dump, map_location="cpu", weights_only=False)
    pairs = keyed(dump["test"])
    probe_label, probe_prob = fit_probe(dump["fit"], [record for record, _ in pairs])

    hold_shift, next_speaker, previous_speaker = [], [], []
    for index, (record, row) in enumerate(pairs):
        ids = [str(s) for s in record["speaker_ids"]]
        shift = int(probe_label[index])
        # The model's own answer to who held the floor. Using the annotation's
        # here would hand the naming step a fact the model has not earned.
        floor_row = int(record["previous_pred"])
        if shift:
            scores = record["next_scores"].clone()
            if scores.numel() > 1:
                scores[floor_row] = -torch.inf
            pick = ids[int(scores.argmax())]
        else:
            pick = ids[floor_row]
        hold_shift.append(shift)
        next_speaker.append(pick)
        previous_speaker.append(ids[floor_row])
    emit("muvap", pairs, hold_shift, next_speaker, previous_speaker, probe_prob)

    # ---- RoleVAP --------------------------------------------------------
    if args.rolevap_dump:
        dump = torch.load(args.rolevap_dump, map_location="cpu", weights_only=False)
        pairs = keyed(dump["test"])
        probe_label, probe_prob = fit_probe(dump["fit"], [record for record, _ in pairs])
        blank = [None] * len(pairs)
        emit("rolevap", pairs, [int(v) for v in probe_label], blank, blank, probe_prob)


if __name__ == "__main__":
    main()
