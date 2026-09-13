"""Run a command and report what the GPU actually did while it ran.

Training throughput questions are usually input questions, and the number that
answers them is the *distribution* of utilisation rather than its average. A run
that alternates 100% and 0% averages 50 and looks identical to a steady 50, but
the first is starving between batches and the second is simply slow. The
percentiles below separate those, and `idle` counts the samples where the card
had nothing to do at all.

```bash
python tools/gpu_watch.py -- python train_muvap.py --config config/yaml/muvap_media.yaml
```
"""

import argparse
import subprocess
import sys
import threading
import time
from pathlib import Path

QUERY = "utilization.gpu,utilization.memory,memory.used,power.draw"


def sample(stop, rows, index, interval):
    while not stop.is_set():
        try:
            out = subprocess.run(
                ["nvidia-smi", f"--query-gpu={QUERY}", "--format=csv,noheader,nounits",
                 "-i", str(index)],
                capture_output=True, text=True, timeout=5,
            ).stdout.strip()
            rows.append([float(value) for value in out.split(",")])
        except Exception:
            pass
        stop.wait(interval)


def report(rows, seconds, log=None):
    if not rows:
        print("no GPU samples were taken")
        return
    import numpy as np

    data = np.array(rows)
    use, _, memory, power = data.T
    lines = [
        f"samples            {len(use)} over {seconds:.0f}s",
        f"utilisation mean   {use.mean():.1f}%",
        "utilisation p5/25/50/75/95   "
        + " / ".join(f"{np.percentile(use, p):.0f}%" for p in (5, 25, 50, 75, 95)),
        f"at or above 95%    {(use >= 95).mean() * 100:.1f}% of samples",
        f"at or above 80%    {(use >= 80).mean() * 100:.1f}% of samples",
        f"idle (0%)          {(use == 0).mean() * 100:.1f}% of samples",
        f"memory peak        {memory.max() / 1024:.1f} GiB",
        f"power mean/peak    {power.mean():.0f} W / {power.max():.0f} W",
    ]
    # A long tail of low samples is the signature of an input pipeline that
    # cannot keep up; a flat high line means the card is the limit.
    verdict = (
        "the GPU is the bottleneck - the input pipeline keeps up"
        if (use >= 95).mean() > 0.9
        else "the GPU is waiting on input for a noticeable share of the run"
    )
    lines.append(f"verdict            {verdict}")
    text = "\n".join(lines)
    print("\n" + text)
    if log:
        Path(log).write_text(text + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interval", type=float, default=0.5)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--log", help="also write the summary here")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        raise SystemExit("give a command after --")

    rows, stop = [], threading.Event()
    watcher = threading.Thread(
        target=sample, args=(stop, rows, args.device, args.interval), daemon=True
    )
    started = time.time()
    watcher.start()
    try:
        code = subprocess.call(command)
    finally:
        stop.set()
        watcher.join(timeout=3)
    report(rows, time.time() - started, args.log)
    sys.exit(code)


if __name__ == "__main__":
    main()
