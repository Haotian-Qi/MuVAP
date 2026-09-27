"""Choose the demo clip, and the shift scale to read it at, from the scan.

Two decisions, both made on numbers rather than on how a clip looks:

**Which thirty seconds.** Every window of the scanned recording is ranked by
what the model actually paid on it - the task's two losses summed, averaged
over the window's frames - and windows that would make a poor demo whatever the
loss are struck out first: a window needs turn-taking in it, every speaker has
to be on screen throughout and every speaker has to say something. The cheapest
window that survives is the clip.

**Which shift scale.** `shift_scale` reweights the shift row of `p_future`
before the 0.5 threshold; it moves where the decision falls and touches nothing
else. It is swept here against the corpus's own turn events, scored the way the
paper scores them - balanced accuracy over hold and shift, which is the only
honest score when picking a threshold on a set where one class could be won by
answering it every time.

Nothing here runs the model. `tools/scan_segments.py` already did, and the rows
it wrote are what the sweep is arithmetic over.
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

FPS = 25


def read_events(path: Path, video_id: str):
    """`segment -> [(frame the model is judged on, is_shift)]` for one recording."""
    events = defaultdict(list)
    for line in Path(path).read_text().splitlines():
        parts = line.split()
        if len(parts) != 8 or parts[0] != video_id:
            continue
        # The model is given everything up to where the mutual silence starts,
        # and nothing of the silence itself; the last frame it saw is the frame
        # its answer is read off.
        frame = round(float(parts[3]) * FPS) - 1
        events[parts[1]].append((frame, parts[6] == "shift"))
    return events


#: Words that would make a clip awkward to show a conference room. The list is
#: deliberately broad and includes mild oaths: a false positive costs one
#: candidate window out of hundreds, and a false negative costs a talk.
BLOCKED_STEMS = (
    "fuck", "shit", "bitch", "cunt", "cock", "pussy", "dick", "prick", "wank",
    "bastard", "whore", "slut", "nigg", "fag", "retard", "spastic", "twat",
    "bollock", "bugger", "arse", "tit", "boob", "porn", "jerkoff", "jackass",
    "dumbass", "badass", "asshole", "asshat", "motherf", "goddam", "godammit",
)
#: Whole words only - these are harmless inside other words ("class", "hello").
BLOCKED_WORDS = (
    "ass", "damn", "damned", "hell", "crap", "crappy", "piss", "pissed",
    "pissing", "screwed", "sucks", "sucked", "freaking", "frickin", "friggin",
    "frick", "fricking", "jesus", "christ", "goddamn", "gosh", "shag", "horny",
    "sex", "sexy", "naked", "nude", "drunk", "stoned", "weed", "coke",
    "balls", "nuts", "butt", "fart", "puke", "vomit", "herpes", "std",
    "kill", "killed", "die", "died", "dead", "suicide", "racist", "nazi",
)


def blocked(word: str) -> bool:
    """Whether one transcript token is on the list, in any inflection."""
    token = "".join(c for c in word.lower() if c.isalpha() or c == "'")
    if not token:
        return False
    if token in BLOCKED_WORDS:
        return True
    return any(stem in token for stem in BLOCKED_STEMS)


def read_words(path: Path, segment: str):
    """`(start_sec, end_sec, word)` for one segment of the word-level layer."""
    out = []
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        parts = line.split("\t")
        if len(parts) < 6 or parts[1] != segment:
            continue
        start, duration = float(parts[3]), float(parts[4])
        out.append((start, start + duration, parts[5]))
    return out


def transcript_of(words, first, last, pad=0.0, fps=FPS):
    """The words spoken inside a frame span, in order.

    `pad` widens the span in seconds. A word ending a tenth of a second before
    the cut is still half audible at the cut, so the check is made on a slightly
    wider span than the clip itself.
    """
    lo, hi = first / fps - pad, last / fps + pad
    return [w for w in words if w[1] > lo and w[0] < hi]


def shift_at(rows, scale):
    """`p_future` rows reweighted and renormalised - `get_shift_hold`, inlined."""
    scaled = rows.copy()
    scaled[:, 1] *= scale
    return scaled[:, 1] / np.clip(scaled.sum(axis=1), 1e-8, None)


def sweep(rows_by_segment, events, scales):
    """Balanced accuracy over every event of the recording, per scale."""
    frames, labels = [], []
    for segment, rows in rows_by_segment.items():
        for frame, is_shift in events.get(segment, []):
            if 0 <= frame < len(rows) and np.isfinite(rows[frame]).all():
                frames.append(rows[frame])
                labels.append(is_shift)
    if not frames:
        return []
    stacked = np.stack(frames)
    truth = np.array(labels, dtype=bool)
    out = []
    for scale in scales:
        predicted = shift_at(stacked, scale) > 0.5
        shift_recall = predicted[truth].mean() if truth.any() else np.nan
        hold_recall = (~predicted[~truth]).mean() if (~truth).any() else np.nan
        out.append(
            {
                "scale": scale,
                "balanced": (shift_recall + hold_recall) / 2,
                "shift_recall": shift_recall,
                "hold_recall": hold_recall,
                "accuracy": (predicted == truth).mean(),
                "n": len(truth),
            }
        )
    return out


def alternations(vad):
    """How many times the floor changes hands, from the annotation alone.

    Counted on frames with exactly one speaker active, so an interjection over
    someone else is not a turn until the other stops. This is what separates a
    conversation from a monologue with agreement noises in it, and the annotated
    turn events do not measure it: they are only the clean transitions through
    mutual silence, and a window can have three of those and still be one person
    talking.
    """
    current, count = None, 0
    for frame in range(vad.shape[1]):
        speaking = np.flatnonzero(vad[:, frame])
        if len(speaking) == 1:
            if current is not None and speaking[0] != current:
                count += 1
            current = int(speaking[0])
    return count


def windows(data, events, length, step, min_shifts, min_holds, min_share, min_visible,
            words=None, word_pad=1.0, min_alternations=0, max_share=1.0):
    """Every window of one segment that could carry a demo, with its loss."""
    loss = data["gvap_loss"] + data["svap_loss"]
    visible, vad = data["visible"], data["vad"]
    frames = len(loss)
    out = []
    for start in range(0, frames - length + 1, step):
        span = slice(start, start + length)
        block = loss[span]
        if not np.isfinite(block).all():
            continue                                  # the unscored head of the segment
        inside = [e for e in events if start <= e[0] < start + length]
        shifts = sum(1 for _, is_shift in inside if is_shift)
        holds = len(inside) - shifts
        if shifts < min_shifts or holds < min_holds:
            continue
        if visible[:, span].mean(axis=1).min() < min_visible:
            continue                                  # a face that leaves the frame
        share = vad[:, span].mean(axis=1)
        if share.min() < min_share:
            continue                                  # a speaker who never speaks
        if share.max() > max_share:
            continue                                  # one speaker holding the floor
        turns = alternations(vad[:, span])
        if turns < min_alternations:
            continue                                  # a statement, not an exchange
        said = (
            transcript_of(words, start, start + length, word_pad)
            if words is not None else []
        )
        bad = sorted({w[2].lower() for w in said if blocked(w[2])})
        if words is not None and bad:
            continue                                  # not sayable from a podium
        out.append(
            {
                "start": start,
                "loss": float(block.mean()),
                "gvap": float(data["gvap_loss"][span].mean()),
                "svap": float(data["svap_loss"][span].mean()),
                "shifts": shifts,
                "holds": holds,
                "visible": float(visible[:, span].mean(axis=1).min()),
                "turns": turns,
                "share": share.round(2).tolist(),
                "words": said,
            }
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scan", required=True, help="directory written by scan_segments.py")
    parser.add_argument("--events", required=True, help="the corpus turn-event file")
    parser.add_argument("--video", required=True)
    parser.add_argument("--seconds", type=float, default=30.0)
    parser.add_argument("--step", type=int, default=25, help="frames between candidate windows")
    parser.add_argument("--min-shifts", type=int, default=2)
    parser.add_argument("--min-holds", type=int, default=1)
    parser.add_argument("--min-visible", type=float, default=0.95,
                        help="least of the window every face must be on screen for")
    parser.add_argument("--speakers", type=int, default=0,
                        help="require exactly this many speakers (0: any)")
    parser.add_argument("--min-share", type=float, default=0.05,
                        help="least of the window each speaker must be speaking")
    parser.add_argument("--max-share", type=float, default=1.0,
                        help="most of the window any one speaker may hold")
    parser.add_argument("--min-alternations", type=int, default=0,
                        help="least number of times the floor must change hands")
    parser.add_argument("--top", type=int, default=8)
    parser.add_argument(
        "--words", default=None,
        help="the corpus word-level transcript directory; given, windows "
             "containing anything on the blocklist are struck out",
    )
    parser.add_argument("--word-pad", type=float, default=1.0,
                        help="seconds either side of the clip the blocklist also covers")
    parser.add_argument("--print-transcript", action="store_true",
                        help="print what is said in each surviving window")
    args = parser.parse_args()

    events = read_events(Path(args.events), args.video)
    scan = sorted(Path(args.scan).glob(f"{args.video}+*.npz"))
    if not scan:
        raise SystemExit(f"no scan for {args.video} in {args.scan}")

    loaded, rows = {}, {}
    for path in scan:
        segment = path.stem.split("+")[1]
        data = np.load(path, allow_pickle=False)
        loaded[segment] = {key: data[key] for key in data.files}
        rows[segment] = loaded[segment]["p_future"]

    print(f"\n  {args.video}: {len(loaded)} segments, "
          f"{sum(len(v) for v in events.values())} annotated turn events\n")
    print(f"  {'segment':<9} {'length':>8} {'gvap':>7} {'svap':>7} {'total':>7} {'events':>7}")
    ranked = []
    for segment, data in loaded.items():
        gvap = float(np.nanmean(data["gvap_loss"]))
        svap = float(np.nanmean(data["svap_loss"]))
        ranked.append((gvap + svap, segment, gvap, svap, len(data["gvap_loss"])))
    for total, segment, gvap, svap, frames in sorted(ranked):
        print(f"  {segment:<9} {frames / FPS:7.1f}s {gvap:7.3f} {svap:7.3f} {total:7.3f} "
              f"{len(events.get(segment, [])):7d}")

    length = int(args.seconds * FPS)
    candidates = []
    for segment, data in loaded.items():
        if args.speakers and len(data["vad"]) != args.speakers:
            continue
        words = (
            read_words(Path(args.words) / args.video / f"{segment}.txt", segment)
            if args.words else None
        )
        for window in windows(
            data, events.get(segment, []), length, args.step,
            args.min_shifts, args.min_holds, args.min_share, args.min_visible,
            words, args.word_pad, args.min_alternations, args.max_share,
        ):
            candidates.append({**window, "segment": segment})
    candidates.sort(key=lambda w: w["loss"])

    print(f"\n  best {args.seconds:.0f} s windows with at least {args.min_shifts} shifts "
          f"and {args.min_holds} holds, every face on screen throughout:\n")
    print(f"  {'segment':<9} {'start':>8} {'total':>7} {'turns':>6} {'visible':>8}  "
          f"{'share':<20} events")
    for window in candidates[: args.top]:
        print(f"  {window['segment']:<9} {window['start'] / FPS:7.1f}s "
              f"{window['loss']:7.3f} {window['turns']:6d} {window['visible']:8.3f}  "
              f"{str(window['share']):<20} "
              f"{window['shifts']} shift / {window['holds']} hold")
    if not candidates:
        raise SystemExit("  none; loosen --min-shifts or --min-share")

    scales = np.round(np.arange(0.5, 6.01, 0.25), 2)
    print("\n  shift_scale, on every turn event of this recording:\n")
    print(f"  {'scale':>6} {'balanced':>9} {'shift rec':>10} {'hold rec':>9} {'accuracy':>9}")
    results = sweep(rows, events, scales)
    best = max(results, key=lambda r: r["balanced"])
    for result in results:
        mark = "  <-- best" if result is best else ""
        print(f"  {result['scale']:6.2f} {result['balanced']:9.3f} {result['shift_recall']:10.3f} "
              f"{result['hold_recall']:9.3f} {result['accuracy']:9.3f}{mark}")
    print(f"\n  n = {results[0]['n']} events\n")

    if args.print_transcript:
        for window in candidates[: args.top]:
            said = " ".join(w[2] for w in window["words"])
            print(f"\n  {window['segment']} @{window['start'] / FPS:.1f}s:\n    {said}")
        print()

    pick = candidates[0]
    print(f"  clip:        --segment {pick['segment']} --start {pick['start'] / FPS:.1f} "
          f"--seconds {args.seconds:.0f}")
    print(f"  shift scale: --shift-scale {best['scale']:g}\n")


if __name__ == "__main__":
    main()
