"""Download the frozen audio frontends ahead of a run.

Both frontends fetch their pretrained weights the first time a model is built,
which is fine interactively but not on a compute node with no outbound network,
and not when a typo in a repository id should fail in seconds rather than after
the dataset has been indexed. This does that fetch on its own.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.modules.audio import AUDIO_ENCODERS, build_audio_encoder  # noqa: E402


def describe(encoder):
    parameters = sum(p.numel() for p in encoder.parameters())
    return f"{parameters / 1e6:.1f}M parameters, {encoder.out_dim}-dim at {encoder.frame_hz} Hz"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "encoders",
        nargs="*",
        default=[],
        choices=sorted(AUDIO_ENCODERS),
        help="which frontends to fetch (default: all)",
    )
    args = parser.parse_args()

    for name in args.encoders or sorted(AUDIO_ENCODERS):
        print(f"{name}: fetching")
        try:
            encoder = build_audio_encoder({"audio_encoder": name})
        except Exception as error:  # network, missing extra, bad repository id
            raise SystemExit(f"{name}: {error}") from error
        print(f"{name}: ready - {describe(encoder)}")


if __name__ == "__main__":
    main()
