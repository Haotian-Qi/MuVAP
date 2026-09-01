"""Canonical media conversion shared by every MuVAP+ dataset.

Dataset adapters resolve paths and metadata only. Signal conversion belongs
here so AVA, WASD, MSDWild, Fisher, and AVCC cannot silently diverge.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
import torchaudio.functional as AF


@dataclass(frozen=True)
class AudioSpec:
    sample_rate: int = 16_000
    mono: bool = True
    peak_limit: float = 1.0 + 1e-4
    # Out-of-range float PCM is a scaling bug in most corpora, so it raises by
    # default. Set this only for a source known to store real audio that
    # overshoots, where clipping is the same thing any encoder would do.
    clamp_to_peak: bool = False


@dataclass(frozen=True)
class VisualSpec:
    height: int = 112
    width: int = 112


CANONICAL_AUDIO = AudioSpec()
CANONICAL_VISUAL = VisualSpec()
ASD_TARGET_RMS_DBFS = -25.0
ASD_MAX_GAIN_DB = 20.0
ASD_PEAK_LIMIT = 0.99


def _integer_audio_to_float(array: np.ndarray) -> np.ndarray:
    info = np.iinfo(array.dtype)
    scale = float(max(abs(info.min), abs(info.max)))
    return array.astype(np.float32) / scale


def _channels_first(array: np.ndarray) -> np.ndarray:
    """Canonicalize common `[N]`, `[C,N]`, and legacy `[N,C]` layouts."""
    if array.ndim == 1:
        return array[None, :]
    if array.ndim != 2:
        raise ValueError(f"NumPy audio must have 1 or 2 dimensions, got {array.shape}")
    if array.shape[0] <= 8:
        return array
    if array.shape[1] <= 8:
        return array.T
    raise ValueError(
        f"cannot infer audio channel axis for shape {array.shape}; expected [C,N] or [N,C]"
    )


def canonicalize_waveform(
    waveform: torch.Tensor, spec: AudioSpec = CANONICAL_AUDIO
) -> torch.Tensor:
    """Return `[channels, samples]` float32 PCM without changing its gain."""
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    if waveform.ndim != 2:
        raise ValueError(
            f"audio must have 1 or 2 dimensions, got {tuple(waveform.shape)}"
        )

    waveform = waveform.to(torch.float32)
    if not torch.isfinite(waveform).all():
        raise ValueError("audio contains NaN or infinite values")
    if waveform.numel() and waveform.abs().max().item() > spec.peak_limit:
        if not spec.clamp_to_peak:
            raise ValueError(
                "floating-point audio exceeds [-1, 1]; convert integer PCM to normalized "
                "float during preprocessing instead of applying per-clip normalization"
            )
        waveform = waveform.clamp(-1.0, 1.0)
    if spec.mono and waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    return waveform.contiguous()


@dataclass(frozen=True)
class WaveformStats:
    """Level of a whole recording, measured after DC removal.

    Packed shards hold unnormalized audio, so these travel in the index and let
    a reader reproduce the exact per-recording gain from any slice of it. Gain
    derived from a slice would drift with the content of that slice.
    """

    mean: float
    rms: float
    peak: float


def waveform_stats(waveform: torch.Tensor) -> WaveformStats:
    mean = waveform.mean().item() if waveform.numel() else 0.0
    centred = waveform - mean
    return WaveformStats(
        mean=mean,
        rms=centred.square().mean().sqrt().item() if centred.numel() else 0.0,
        peak=centred.abs().max().item() if centred.numel() else 0.0,
    )


def normalization_gain(
    stats: WaveformStats,
    target_rms_dbfs: float = ASD_TARGET_RMS_DBFS,
    max_gain_db: float = ASD_MAX_GAIN_DB,
    peak_limit: float = ASD_PEAK_LIMIT,
) -> float:
    if stats.rms <= 1e-8 or stats.peak <= 1e-8:
        return 1.0
    return min(
        10 ** (target_rms_dbfs / 20) / stats.rms,
        10 ** (max_gain_db / 20),
        peak_limit / stats.peak,
    )


def apply_normalization(
    waveform: torch.Tensor,
    stats: WaveformStats,
    target_rms_dbfs: float = ASD_TARGET_RMS_DBFS,
    max_gain_db: float = ASD_MAX_GAIN_DB,
    peak_limit: float = ASD_PEAK_LIMIT,
) -> torch.Tensor:
    """Centre and scale a slice using its whole recording's statistics."""
    gain = normalization_gain(stats, target_rms_dbfs, max_gain_db, peak_limit)
    return ((waveform - stats.mean) * gain).contiguous()


def normalize_waveform(
    waveform: torch.Tensor,
    target_rms_dbfs: float = ASD_TARGET_RMS_DBFS,
    max_gain_db: float = ASD_MAX_GAIN_DB,
    peak_limit: float = ASD_PEAK_LIMIT,
) -> tuple[torch.Tensor, float]:
    stats = waveform_stats(waveform)
    gain = normalization_gain(stats, target_rms_dbfs, max_gain_db, peak_limit)
    return ((waveform - stats.mean) * gain).contiguous(), gain


def load_waveform(
    path: str | Path,
    spec: AudioSpec = CANONICAL_AUDIO,
    *,
    npy_sample_rate: int | None = None,
) -> tuple[torch.Tensor, int]:
    """Load file or NumPy PCM into the canonical amplitude representation."""
    path = Path(path)
    if path.suffix.lower() == ".npy":
        array = _channels_first(np.load(path, allow_pickle=False))
        if np.issubdtype(array.dtype, np.integer):
            array = _integer_audio_to_float(array)
        waveform = torch.from_numpy(np.asarray(array))
        source_rate = npy_sample_rate
        if source_rate is None:
            raise ValueError(f"sample rate is required for NumPy audio: {path}")
    else:
        # torchaudio returns normalized float PCM for integer WAV/FLAC sources.
        waveform, source_rate = torchaudio.load(path, normalize=True)
        # Lossy codecs such as AAC can decode slightly outside nominal PCM bounds.
        waveform = waveform.clamp(-1.0, 1.0)

    waveform = canonicalize_waveform(waveform, spec)
    if source_rate != spec.sample_rate:
        waveform = AF.resample(waveform, source_rate, spec.sample_rate)
    return waveform.contiguous(), spec.sample_rate


def slice_audio(
    waveform: torch.Tensor,
    sample_rate: int,
    *,
    start_time: float | None = None,
    end_time: float | None = None,
    target_samples: int | None = None,
) -> torch.Tensor:
    """Keep timeline origin: crop and zero-pad only at the right boundary."""
    start = round((start_time or 0.0) * sample_rate)
    end = round(end_time * sample_rate) if end_time is not None else waveform.shape[-1]
    if start < 0 or end < start:
        raise ValueError("audio time range is invalid")
    waveform = waveform[..., start:end]

    if target_samples is not None:
        if target_samples < 0:
            raise ValueError("target_samples cannot be negative")
        waveform = waveform[..., :target_samples]
        waveform = F.pad(waveform, (0, max(0, target_samples - waveform.shape[-1])))
    return waveform.contiguous()


def validate_visual(
    array: np.ndarray, spec: VisualSpec = CANONICAL_VISUAL
) -> np.ndarray:
    """Validate the tensor contract expected by `VisualEncoder`."""
    if array.ndim != 3 or tuple(array.shape[1:]) != (spec.height, spec.width):
        raise ValueError(
            f"visual must be [T, {spec.height}, {spec.width}], got {array.shape}"
        )
    if array.dtype != np.uint8:
        raise ValueError(f"visual must be uint8 in [0, 255], got {array.dtype}")
    return array
