"""Kyutai Mimi as a frozen causal audio frontend.

Mimi is the neural codec from Kyutai's Moshi. It is causal by construction -
its SEANet convolutions left-pad, and its encoder transformer uses a sliding
causal mask - which is what makes it usable here at all: a VAP model that saw
even a few milliseconds of future audio would not be a turn-taking predictor.

The encoder is tapped *before* Mimi's own 2x downsample, which puts its
features on a 25 Hz grid - exactly the rate the projection window labels. The
12.5 Hz `latent` tap is available for completeness but does not line up with a
25 Hz label grid, so `build_vap` refuses the mismatch rather than let the loss
fail on a shape error deep in a run.

Weights are the pretrained `kyutai/mimi` checkpoint and stay frozen; only the
projection onto `d_model` is learned.
"""

import torch.nn as nn
import torch.nn.functional as F
import torchaudio

SOURCE_RATE = 16_000
MIMI_RATE = 24_000
#: Mimi consumes 1920 samples (80 ms) per latent at 24 kHz; the encoder emits
#: one feature per 960 samples, which is 25 Hz.
MIMI_ENCODER_STRIDE = 960
#: Left delay that buys back the resampler's lookahead, with margin. 1 ms.
RESAMPLE_DELAY = 16


def causal_resample_16k_to_24k(waveform):
    """16 -> 24 kHz without reading the future.

    `torchaudio.functional.resample` convolves with a symmetric sinc, so output
    sample m depends on input past m*2/3 - measured, 9 output samples of
    lookahead at this rate pair. Left-padding alone does not help, because
    dropping the same number of output samples undoes the shift exactly.
    Instead the signal is *delayed* by `RESAMPLE_DELAY` input samples while the
    nominal output length is kept, so every output sample depends only on input
    strictly behind it. The price is a 1 ms delay against a 40 ms frame.
    """
    length = round(waveform.shape[-1] * MIMI_RATE / SOURCE_RATE)
    delayed = F.pad(waveform, (RESAMPLE_DELAY, 0))
    return torchaudio.functional.resample(delayed, SOURCE_RATE, MIMI_RATE)[..., :length]


class MimiEncoder(nn.Module):
    """Frozen Mimi encoder producing `[batch, frames, 512]` at 25 Hz."""

    TAPS = ("encoder", "latent")

    def __init__(self, repo_id="kyutai/mimi", tap="encoder"):
        super().__init__()
        if tap not in self.TAPS:
            raise ValueError(f"tap must be one of {self.TAPS}, got {tap!r}")
        try:
            from transformers import MimiModel
        except ImportError as error:  # pragma: no cover - dependency guard
            raise ImportError(
                "vap.audio_encoder: mimi needs transformers: pip install 'transformers>=4.45'"
            ) from error

        mimi = MimiModel.from_pretrained(repo_id)
        if not getattr(mimi.config, "use_causal_conv", False):
            raise ValueError(f"{repo_id} was built non-causal; it cannot front a VAP model")

        # Nothing downstream reconstructs audio or uses discrete codes, and the
        # decoder is over half the checkpoint.
        for unused in ("decoder", "decoder_transformer", "upsample", "quantizer"):
            if hasattr(mimi, unused):
                setattr(mimi, unused, None)
        if tap == "encoder":
            mimi.downsample = None

        self.mimi = mimi
        self.tap = tap
        self._out_dim = int(mimi.config.hidden_size)
        self.freeze()

    @property
    def out_dim(self):
        return self._out_dim

    @property
    def frame_hz(self):
        return 25.0 if self.tap == "encoder" else float(self.mimi.config.frame_rate)

    def freeze(self):
        for parameter in self.mimi.parameters():
            parameter.requires_grad_(False)

    def train(self, mode=True):
        """Keep the frozen frontend in eval mode.

        Its transformer has dropout; leaving it active would inject noise into
        features that never adapt to it.
        """
        super().train(mode)
        self.mimi.eval()
        return self

    def forward(self, waveform):
        if waveform.ndim == 2:
            waveform = waveform.unsqueeze(1)
        if waveform.shape[1] != 1:
            raise ValueError(
                f"MimiEncoder takes one channel at a time, got {waveform.shape[1]}"
            )

        audio = causal_resample_16k_to_24k(waveform)
        # Right-padding to a whole frame only ever adds future silence.
        remainder = (-audio.shape[-1]) % MIMI_ENCODER_STRIDE
        if remainder:
            audio = F.pad(audio, (0, remainder))

        features = self.mimi.encoder(audio)
        encoded = self.mimi.encoder_transformer(features.transpose(1, 2))
        if hasattr(encoded, "last_hidden_state"):
            encoded = encoded.last_hidden_state
        elif isinstance(encoded, (tuple, list)):
            encoded = encoded[0]

        if self.tap == "latent":
            encoded = self.mimi.downsample(encoded.transpose(1, 2)).transpose(1, 2)
        return encoded
