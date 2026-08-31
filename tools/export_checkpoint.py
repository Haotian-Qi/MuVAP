"""Turn a training checkpoint into something publishable.

A Lightning checkpoint is a resume artifact: most of it is optimizer moments,
scheduler and loop state, and - because the frozen frontend is an ordinary
submodule - a copy of pretrained weights that carry their own licence.

This writes the trained weights alone, keyed as the model expects them, beside
the config needed to rebuild it. The frozen frontend is dropped by default and
restored from its own pretrained source when the model is constructed.
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.release import CONFIG_NAME, PROVENANCE_NAME, WEIGHTS_NAME  # noqa: E402

MODULE_PREFIX = "model."


def strip_prefix(state_dict, prefix=MODULE_PREFIX):
    """Drop the LightningModule attribute name from every key."""
    return {
        key[len(prefix) :] if key.startswith(prefix) else key: value
        for key, value in state_dict.items()
    }


def build_model(cfg, module):
    if module == "asd":
        from models.asd import AudioVisualASD

        return AudioVisualASD(cfg["asd"])
    from models.vap import build_vap

    return build_vap(cfg["vap"])


def frozen_keys(model):
    return {name for name, parameter in model.named_parameters() if not parameter.requires_grad}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("output", help="destination directory")
    parser.add_argument("--module", choices=("vap", "asd"), default="vap")
    parser.add_argument(
        "--keep-frozen",
        action="store_true",
        help="also ship the frozen frontend, making the file self-contained "
        "but redistributing pretrained weights under their own licence",
    )
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if "state_dict" not in checkpoint:
        raise SystemExit(f"{args.checkpoint} is not a Lightning checkpoint")

    cfg = dict(checkpoint.get("hyper_parameters") or {})
    if args.module not in cfg:
        raise SystemExit(
            f"checkpoint carries no '{args.module}' config; was it trained by train_{args.module}.py?"
        )

    model = build_model(cfg, args.module)
    # Ship exactly what the model asks for. A checkpoint also carries buffers
    # the model derives at build time, and those are not weights.
    wanted = set(model.state_dict())
    weights = {k: v for k, v in strip_prefix(checkpoint["state_dict"]).items() if k in wanted}

    dropped = set()
    if not args.keep_frozen:
        dropped = frozen_keys(model) & set(weights)
        weights = {k: v for k, v in weights.items() if k not in dropped}

    # Loading into a freshly built model proves the export is usable: the frozen
    # frontend arrives from its own pretrained source, everything else from here.
    missing, _ = model.load_state_dict(weights, strict=False)
    if set(missing) - dropped:
        raise SystemExit(f"weights missing from the export: {sorted(set(missing) - dropped)[:5]}")

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(weights, out / WEIGHTS_NAME)
    with (out / CONFIG_NAME).open("w") as handle:
        yaml.safe_dump({args.module: cfg[args.module]}, handle, sort_keys=False)
    selection = next(
        (v for k, v in (checkpoint.get("callbacks") or {}).items() if "ModelCheckpoint" in k),
        {},
    )
    score = selection.get("best_model_score")
    provenance = {
        "module": args.module,
        "selected_on": selection.get("monitor"),
        "selected_score": float(score) if score is not None else None,
        "epoch": checkpoint.get("epoch"),
        "global_step": checkpoint.get("global_step"),
        "lightning_version": checkpoint.get("pytorch-lightning_version"),
        "parameters": sum(v.numel() for v in weights.values()),
        "frozen_frontend_omitted": sorted({k.split(".")[0] for k in dropped}) or None,
    }
    with (out / PROVENANCE_NAME).open("w") as handle:
        json.dump(provenance, handle, indent=2)

    source_mb = Path(args.checkpoint).stat().st_size / 1e6
    exported_mb = (out / WEIGHTS_NAME).stat().st_size / 1e6
    print(f"{args.checkpoint}  {source_mb:8.1f} MB")
    print(f"{out / WEIGHTS_NAME}  {exported_mb:8.1f} MB "
          f"({provenance['parameters'] / 1e6:.2f}M params, {exported_mb / source_mb:.0%} of the original)")
    if dropped:
        print(f"omitted frozen frontend: {provenance['frozen_frontend_omitted']} "
              f"({len(dropped)} tensors), restored from its pretrained source on load")


if __name__ == "__main__":
    main()
