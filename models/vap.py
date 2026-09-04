"""Voice Activity Projection encoders.

Three architectures share one head contract: a `[batch, frames, classes]` logit
stream over whatever codebook the configured `ProjectionWindow` defines.

* `AudioVAP` reads one mixed-mono channel. Speaker identity is not given to
  the model, so it only works with the `role_relative` codebook, which names
  speakers by their role in the conversation.
* `StereoVAP` reproduces the original VAP model: one channel per speaker,
  a shared encoder and self-attention stack, cross-attention between the two
  channels, and one head over their concatenation. Channels are ordered, so it
  pairs with the `speaker_based` codebook that projects one channel per row.
* `AnchoredVAP` also uses the `speaker_based` codebook, but from one mixed
  channel: the identity a second channel would carry comes instead from the
  voice activity strictly before the frame being labelled, conditioned in.
"""

import torch
import torch.nn as nn

from projection_window import ProjectionWindow

from .modules.attention import TransformerModel
from .modules.audio import build_audio_encoder, check_frame_rate, encoder_projection
from .modules.projection import LinearHead


class AudioVAP(nn.Module):
    """Single-channel VAP over a mixed recording."""

    def __init__(self, cfg):
        super().__init__()
        model_dim = cfg["temporal"]["d_model"]
        self.audio_encoder = build_audio_encoder(cfg)
        self.encoder_proj = encoder_projection(self.audio_encoder.out_dim, model_dim)
        self.transformer = TransformerModel(cfg["temporal"])
        output_dim = ProjectionWindow(**cfg["projection_window"]).n_classes
        self.vap_head = LinearHead(model_dim, output_dim)

    def forward(self, audio, return_embeddings=False):
        if audio.ndim == 3 and audio.shape[1] != 1:
            raise ValueError(
                f"AudioVAP expects one audio channel, got {audio.shape[1]}; "
                "use arch: stereo for two-channel input"
            )
        embedding = self.transformer(self.encoder_proj(self.audio_encoder(audio)))
        logits = self.vap_head(embedding)
        if return_embeddings:
            return logits, embedding
        return logits


class StereoVAP(nn.Module):
    """Two-channel VAP, one channel per speaker, as in the original VAP paper."""

    def __init__(self, cfg):
        super().__init__()
        model_dim = cfg["temporal"]["d_model"]
        cross_cfg = cfg.get("cross", cfg["temporal"])
        if cross_cfg["d_model"] != model_dim:
            raise ValueError("StereoVAP temporal and cross d_model must match")

        self.audio_encoder = build_audio_encoder(cfg)
        self.encoder_proj = encoder_projection(self.audio_encoder.out_dim, model_dim)
        # One set of weights sees both channels: which speaker sits on which
        # channel is arbitrary, so the encoder must not learn a per-channel bias.
        self.transformer = TransformerModel(cfg["temporal"])
        self.cross = TransformerModel(cross_cfg, cross_attention=True)
        self.channel_embed = nn.Embedding(2, model_dim)
        nn.init.zeros_(self.channel_embed.weight)

        output_dim = ProjectionWindow(**cfg["projection_window"]).n_classes
        self.vap_head = LinearHead(model_dim * 2, output_dim)

    def encode_channel(self, waveform, channel):
        index = torch.tensor(channel, device=waveform.device)
        encoded = self.encoder_proj(self.audio_encoder(waveform))
        return encoded + self.channel_embed(index)

    def forward(self, audio, return_embeddings=False):
        if audio.ndim != 3 or audio.shape[1] != 2:
            raise ValueError(
                f"StereoVAP expects [batch, 2, samples] audio, got {tuple(audio.shape)}"
            )

        first = self.transformer(self.encode_channel(audio[:, 0:1], 0))
        second = self.transformer(self.encode_channel(audio[:, 1:2], 1))

        first_cross = self.cross(first, src=second)
        second_cross = self.cross(second, src=first)

        embedding = torch.cat([first_cross, second_cross], dim=-1)
        logits = self.vap_head(embedding)
        if return_embeddings:
            return logits, embedding
        return logits


class AnchoredVAP(nn.Module):
    """One mixed channel, anchored by the voice activity behind each frame.

    This is `VapGPTMono` from the original VAP repository, on this codebase's
    parts. `speaker_based` classes name channels, which mono audio cannot
    identify - which is why `AudioVAP` is barred from that codebook. The voice
    activity supplies the identity instead: a `Linear(2, d_model)`, layer
    normalised, added to the encoder output, then the same two causal stacks the
    stereo model uses, run back to back on the single channel.

    One departure, forced by the label convention. The reference conditions on
    the activity at frame `t` because its target starts at `t + 1`
    (`ObjectiveVAP.projection` shifts with `va[..., 1:, :]`). `speaker_based`
    here bins the future from `t` inclusive, so the anchor is delayed by one
    frame to keep the same property: the model is never shown any part of the
    window it is being asked to predict.
    """

    def __init__(self, cfg):
        super().__init__()
        model_dim = cfg["temporal"]["d_model"]
        cross_cfg = cfg.get("cross", cfg["temporal"])
        if cross_cfg["d_model"] != model_dim:
            raise ValueError("AnchoredVAP temporal and cross d_model must match")

        self.audio_encoder = build_audio_encoder(cfg)
        self.encoder_proj = encoder_projection(self.audio_encoder.out_dim, model_dim)
        # `init_va_conditioning` in the reference: orthogonal weights, and a
        # layer norm on the conditioning before it is added to the audio.
        self.anchor = nn.Linear(2, model_dim)
        nn.init.orthogonal_(self.anchor.weight)
        self.anchor_norm = nn.LayerNorm(model_dim)
        # `ar_channel` then `ar`: both causal self-attention over one channel.
        self.transformer = TransformerModel(cfg["temporal"])
        self.cross = TransformerModel(cross_cfg)
        self.vap_head = LinearHead(model_dim, ProjectionWindow(**cfg["projection_window"]).n_classes)

    @staticmethod
    def past_activity(vad):
        """`[batch, 2, frames]` voice activity delayed by one frame."""
        return torch.cat([torch.zeros_like(vad[:, :, :1]), vad[:, :, :-1]], dim=2)

    def encode_activity(self, vad, dtype):
        return self.anchor_norm(self.anchor(self.past_activity(vad).transpose(1, 2).to(dtype)))

    def forward(self, audio, vad, return_embeddings=False):
        if audio.ndim == 3 and audio.shape[1] != 1:
            raise ValueError(
                f"AnchoredVAP expects one audio channel, got {audio.shape[1]}"
            )
        features = self.encoder_proj(self.audio_encoder(audio))
        if vad.shape[-1] != features.shape[1]:
            raise ValueError(
                f"anchor has {vad.shape[-1]} frames, audio has {features.shape[1]}"
            )
        embedding = self.cross(self.transformer(features + self.encode_activity(vad, features.dtype)))
        logits = self.vap_head(embedding)
        if return_embeddings:
            return logits, embedding
        return logits


ARCHITECTURES = {"mono": AudioVAP, "stereo": StereoVAP, "anchored": AnchoredVAP}


def build_vap(cfg):
    """Instantiate the VAP architecture named by `cfg['arch']` (default mono)."""
    arch = str(cfg.get("arch", "mono")).lower()
    if arch not in ARCHITECTURES:
        raise ValueError(f"arch must be one of {sorted(ARCHITECTURES)}, got {arch!r}")

    projection = ProjectionWindow(**cfg["projection_window"])
    if arch == "anchored" and projection.mode != "speaker_based":
        raise ValueError(
            "anchored conditioning names channels, so it pairs with speaker_based; "
            "a role codebook needs no anchor"
        )
    if arch == "mono" and projection.mode == "speaker_based":
        raise ValueError(
            "speaker_based labels name a channel, which mono audio cannot identify; "
            "use arch: stereo or a role-based projection mode"
        )
    model = ARCHITECTURES[arch](cfg)
    check_frame_rate(model.audio_encoder, projection.frame_hz, "the projection window")
    return model


def audio_channels(cfg):
    """Channels the configured architecture reads from disk."""
    return 2 if str(cfg.get("arch", "mono")).lower() == "stereo" else 1


def needs_vad(cfg):
    """Whether the architecture is conditioned on voice activity as well as audio."""
    return str(cfg.get("arch", "mono")).lower() == "anchored"
