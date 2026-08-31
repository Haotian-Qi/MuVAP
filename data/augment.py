"""Train-time augmentation for packed ASD samples.

Every transform here is label-preserving except `substitute_audio`, which
rewrites the activity labels to silence on purpose. Visual parameters are drawn
once per chunk and applied to every frame, because the model is temporal and a
per-frame jitter would destroy the motion it reads.
"""

import math
import random
from dataclasses import dataclass

import cv2
import numpy as np

VISUAL_MODES = ("orig", "flip", "crop", "rotate")


@dataclass(frozen=True)
class AugmentConfig:
    """How aggressively to augment; all probabilities are per chunk."""

    audio_mix_prob: float = 0.5
    audio_snr_db: tuple[float, float] = (-5.0, 5.0)
    visual: bool = True
    crop_scale: tuple[float, float] = (0.7, 1.0)
    rotate_degrees: float = 15.0
    # Ablation only: replace the audio outright and call the sample silent.
    # Stronger than mixing for teaching rejection of off-screen speech, but it
    # also lets the model key on audio plausibility instead of AV synchrony.
    substitute_audio_prob: float = 0.0

    @classmethod
    def from_cfg(cls, cfg: dict | None) -> "AugmentConfig | None":
        if not cfg:
            return None
        snr = cfg.get("audio_snr_db", (-5.0, 5.0))
        return cls(
            audio_mix_prob=float(cfg.get("audio_mix_prob", 0.5)),
            audio_snr_db=(float(snr[0]), float(snr[1])),
            visual=bool(cfg.get("visual", True)),
            crop_scale=tuple(cfg.get("crop_scale", (0.7, 1.0))),
            rotate_degrees=float(cfg.get("rotate_degrees", 15.0)),
            substitute_audio_prob=float(cfg.get("substitute_audio_prob", 0.0)),
        )


def _power_db(samples: np.ndarray) -> float:
    return 10.0 * math.log10(float(np.mean(samples.astype(np.float64) ** 2)) + 1e-4)


def fit_length(donor: np.ndarray, samples: int) -> np.ndarray:
    """Wrap or truncate donor audio to the target length."""
    if len(donor) == 0:
        return np.zeros(samples, dtype=np.float32)
    if len(donor) < samples:
        donor = np.tile(donor, math.ceil(samples / len(donor)))
    return donor[:samples]


def mix_audio(
    clean: np.ndarray, donor: np.ndarray, snr_db: float
) -> np.ndarray:
    """Add donor audio at a target signal-to-noise ratio.

    The label is deliberately unchanged. A negative face with foreign speech
    added must stay negative, which is what teaches the model to reject speech
    it cannot see; a positive face stays positive when a competing voice
    arrives.
    """
    donor = fit_length(donor, len(clean))
    scale = 10.0 ** ((_power_db(clean) - _power_db(donor) - snr_db) / 20.0)
    return np.clip(clean + donor * scale, -1.0, 1.0)


def augment_visual(
    faces: np.ndarray, config: AugmentConfig, rng: random.Random
) -> np.ndarray:
    """Apply one geometric transform to every frame of a chunk."""
    mode = rng.choice(VISUAL_MODES)
    if mode == "orig":
        return faces
    size = faces.shape[-1]
    if mode == "flip":
        return np.ascontiguousarray(faces[:, :, ::-1])
    if mode == "crop":
        low, high = config.crop_scale
        side = max(8, int(size * rng.uniform(low, high)))
        if side >= size:
            return faces
        top = rng.randint(0, size - side)
        left = rng.randint(0, size - side)
        window = faces[:, top : top + side, left : left + side]
        return np.stack(
            [
                cv2.resize(frame, (size, size), interpolation=cv2.INTER_LINEAR)
                for frame in window
            ]
        )
    angle = rng.uniform(-config.rotate_degrees, config.rotate_degrees)
    matrix = cv2.getRotationMatrix2D((size / 2, size / 2), angle, 1.0)
    return np.stack(
        [cv2.warpAffine(frame, matrix, (size, size)) for frame in faces]
    )
