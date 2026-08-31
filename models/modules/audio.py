"""Selection and dimension-matching of the frozen audio frontend.

VAP and ASD both take a pretrained frozen encoder and run their own stack on
top, so which encoder that is belongs in one place rather than in each model.
"""

import torch.nn as nn

from .cpc_encoder import CPC
from .mimi_encoder import MimiEncoder

AUDIO_ENCODERS = {"cpc": CPC, "mimi": MimiEncoder}


def build_audio_encoder(cfg):
    """Frozen audio frontend named by `cfg['audio_encoder']` (default CPC).

    Accepts either a bare name or a block carrying the encoder's own options:

        audio_encoder: mimi
        audio_encoder: {name: mimi, tap: encoder}
    """
    spec = cfg.get("audio_encoder", "cpc")
    if isinstance(spec, str):
        spec = {"name": spec}
    options = dict(spec)
    name = str(options.pop("name", "cpc")).lower()
    if name not in AUDIO_ENCODERS:
        raise ValueError(
            f"audio_encoder must be one of {sorted(AUDIO_ENCODERS)}, got {name!r}"
        )
    return AUDIO_ENCODERS[name](**options)


def encoder_projection(encoder_dim, model_dim):
    """Bridge a frozen encoder's output onto the stack width, free when equal."""
    if encoder_dim == model_dim:
        return nn.Identity()
    return nn.Sequential(nn.LayerNorm(encoder_dim), nn.Linear(encoder_dim, model_dim))


def check_frame_rate(encoder, frame_hz, what):
    """Refuse a frontend whose feature rate does not match the target timeline."""
    encoder_hz = getattr(encoder, "frame_hz", frame_hz)
    if float(encoder_hz) != float(frame_hz):
        raise ValueError(
            f"audio encoder emits {encoder_hz} Hz features but {what} runs at "
            f"{frame_hz} Hz; they must agree frame for frame"
        )
