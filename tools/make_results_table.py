"""Emit the AVCC results table from the per-event score CSVs.

The CSVs are the `<name>_predictions.csv` files `tools/score_benchmark.py`
writes, one row per event keyed on its `event_id`; the cells and the ground
truth come from the same `test_events.txt`, eighth column `n_speakers`:

```bash
python tools/make_results_table.py --events /data/AVCC/test_events.txt \
    --muvap scores/muvap_predictions.csv --rolevap scores/rolevap_predictions.csv
```

The cell is the event file's `n_speakers`, never the count of speakers a
window happens to track: the two disagree on 99 events, and mixing them puts an
event in one cell for the humans and the other for the model.

Balanced accuracy throughout - the mean of the hold and shift accuracies inside
the cell - so a cell that is not exactly balanced cannot flatter a model that
leans one way.

`--exclude` drops event ids (one per line, or the first column of a CSV), which
is how a model row is scored on the same events a human-consensus row keeps
after its ties are removed. Give it once per scoring, since the tie sets differ.
"""
import argparse, csv, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.score_benchmark import read_events  # noqa: E402


def read_rows(path):
    """The event file's rows, keyed on the `event_id` the score CSVs carry."""
    return {row["event_id"]: row for row in read_events(path).values()}


def read_scores(path):
    return {r["event_id"]: r for r in csv.DictReader(open(path))}


def read_exclude(path):
    if not path:
        return set()
    ids = set()
    for line in Path(path).read_text().splitlines():
        tok = line.split(",")[0].strip()
        if tok and tok.lower() != "event_id":
            ids.add(tok)
    return ids


def balanced(rows, correct):
    hold = [r for r in rows if r["_type"] == "hold"]
    shift = [r for r in rows if r["_type"] == "shift"]
    ah = sum(correct(r) for r in hold) / len(hold) if hold else float("nan")
    as_ = sum(correct(r) for r in shift) / len(shift) if shift else float("nan")
    return (ah + as_) / 2, ah, as_


def cells(rows):
    yield "2-spk", [r for r in rows if r["_cell"] == "2"]
    yield "3-spk", [r for r in rows if r["_cell"] == "3"]
    yield "All", rows


def score(rows, correct):
    got = {name: balanced(sub, correct) for name, sub in cells(rows)}
    return got


def fmt(x):
    return "--" if x != x else f"{100 * x:.1f}"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--events", type=Path, required=True,
        help="test_events.txt, with n_speakers as its eighth column",
    )
    p.add_argument("--muvap", type=Path, required=True)
    p.add_argument("--rolevap", type=Path)
    p.add_argument("--exclude-nsp", type=Path, help="tie events to drop under NSP scoring")
    p.add_argument("--exclude-hs", type=Path, help="tie events to drop under hold/shift scoring")
    p.add_argument("--label", default="", help="suffix for the row names")
    args = p.parse_args()

    truth = read_rows(args.events)
    drop_nsp, drop_hs = read_exclude(args.exclude_nsp), read_exclude(args.exclude_hs)

    def prep(scores, drop):
        rows = []
        for eid, k in truth.items():
            if eid in drop or eid not in scores:
                continue
            s = dict(scores[eid])
            s["_cell"] = k["n_speakers"]
            s["_type"] = k["event_type"]
            s["_gold_next"] = k["gold_next_speaker"]
            rows.append(s)
        return rows

    hs_ok = lambda r: r["pred_hold_shift"] == r["_type"]
    nsp_ok = lambda r: r["pred_next_speaker"] == r["_gold_next"]

    print(f"{'system':<22}{'scoring':<11}{'2-spk':>8}{'3-spk':>8}{'All':>8}"
          f"{'hold':>8}{'shift':>8}{'n':>7}")
    print("-" * 80)
    for name, path, both in (("RoleVAP", args.rolevap, False), ("MuVAP", args.muvap, True)):
        if path is None:
            continue
        scores = read_scores(path)
        rows = prep(scores, drop_hs)
        got = score(rows, hs_ok)
        print(f"{name + args.label:<22}{'hold/shift':<11}"
              f"{fmt(got['2-spk'][0]):>8}{fmt(got['3-spk'][0]):>8}{fmt(got['All'][0]):>8}"
              f"{fmt(got['All'][1]):>8}{fmt(got['All'][2]):>8}{len(rows):>7}")
        if both:
            rows = prep(scores, drop_nsp)
            got = score(rows, nsp_ok)
            print(f"{'':<22}{'NSP':<11}"
                  f"{fmt(got['2-spk'][0]):>8}{fmt(got['3-spk'][0]):>8}{fmt(got['All'][0]):>8}"
                  f"{fmt(got['All'][1]):>8}{fmt(got['All'][2]):>8}{len(rows):>7}")


if __name__ == "__main__":
    main()
