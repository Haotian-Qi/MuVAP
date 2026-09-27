"""Reading a published weight release back into a model.

A release holds the trained weights alone: no optimizer state, and no copy of
the frozen frontend, which is restored from its own pretrained source when the
model is built. It also ships the model half of the config it was trained with,
so a caller supplies only where the data lives and where results go.
"""

from pathlib import Path

import torch
import yaml

WEIGHTS_NAME = "weights.pt"
CONFIG_NAME = "config.yaml"
PROVENANCE_NAME = "provenance.json"


def resolve(path):
    """Accept either a release directory or the weights file inside one."""
    path = Path(path)
    if path.is_dir():
        weights = path / WEIGHTS_NAME
        if not weights.exists():
            raise SystemExit(f"{path} is not a release: no {WEIGHTS_NAME}")
        return weights, path / CONFIG_NAME
    return path, path.parent / CONFIG_NAME


def merge_config(cfg, release_config, module):
    """Take the architecture from the release, keep local paths from `cfg`.

    A release must be rebuilt with the architecture it was trained with, but
    dataset roots and output directories belong to whoever is running it.
    """
    if not Path(release_config).exists():
        return cfg
    with open(release_config) as handle:
        published = yaml.safe_load(handle) or {}
    if module in published:
        cfg[module] = {**cfg.get(module, {}), **published[module]}
    return cfg


def module_config(weights, module, fallback=None):
    """The weights file, and the architecture to rebuild them into.

    A release ships the config it was trained with, which is the one that can be
    trusted. `fallback` is for weights that predate that convention.
    """
    weights_path, release_config = resolve(weights)
    if Path(release_config).exists():
        with open(release_config) as handle:
            published = yaml.safe_load(handle) or {}
        if module in published:
            return weights_path, published[module]
    if fallback is None:
        raise SystemExit(
            f"{weights} ships no '{module}' config, so there is nothing to say how "
            "to rebuild it; give the architecture it was trained with"
        )
    return weights_path, fallback


def load_weights(model, weights_path):
    """Load a release into `model`, allowing only the frozen frontend to be absent."""
    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected:
        raise SystemExit(
            f"{weights_path} holds weights this model does not want, so it was trained "
            f"with a different architecture: {sorted(unexpected)[:5]}"
        )
    frozen = {name for name, p in model.named_parameters() if not p.requires_grad}
    unexplained = set(missing) - frozen
    if unexplained:
        raise SystemExit(
            f"{weights_path} is missing trainable weights: {sorted(unexplained)[:5]}"
        )
    return model
