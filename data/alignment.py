"""Time-based alignment utilities for audio, video frames, and frame labels.

The model runs on a fixed 25 Hz timeline. Source videos remain at their native
frame rate; frames and labels are sampled onto the model timeline at load time.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class AlignmentConfig:
    model_fps: float = 25.0
    audio_sample_rate: int = 16_000

    def __post_init__(self) -> None:
        if self.model_fps <= 0:
            raise ValueError("model_fps must be positive")
        if self.audio_sample_rate <= 0:
            raise ValueError("audio_sample_rate must be positive")


@dataclass(frozen=True)
class AVSample:
    sample_id: str
    audio_path: Path
    visual_path: Path
    source_fps: float
    labels: Sequence[int]
    start_sec: float = 0.0

    def __post_init__(self) -> None:
        if self.source_fps <= 0:
            raise ValueError(f"{self.sample_id}: source_fps must be positive")
        if self.start_sec < 0:
            raise ValueError(f"{self.sample_id}: start_sec cannot be negative")


def model_frame_count(source_frames: int, source_fps: float, model_fps: float) -> int:
    """Return the number of complete model intervals in a source-frame span."""
    if source_frames < 0 or source_fps <= 0 or model_fps <= 0:
        raise ValueError("frame counts and frame rates must be valid")
    return int(np.floor((source_frames / source_fps) * model_fps + 1e-9))


def source_indices(
    source_frames: int, source_fps: float, model_fps: float
) -> np.ndarray:
    """Map model-frame centres to the source intervals that contain them."""
    count = model_frame_count(source_frames, source_fps, model_fps)
    model_centres = (np.arange(count, dtype=np.float64) + 0.5) / model_fps
    indices = np.floor(model_centres * source_fps).astype(np.int64)
    return np.clip(indices, 0, max(source_frames - 1, 0))


def model_indices(
    source_frames: int, source_fps: float, model_fps: float
) -> np.ndarray:
    """Map source-frame centres to the model intervals that contain them.

    The inverse direction of `source_indices`, used to lift model-rate
    predictions back onto the native annotation timeline for evaluation.
    Always returns exactly `source_frames` indices, so a caller can score one
    prediction per native frame regardless of the source frame rate.
    """
    if source_frames < 0:
        raise ValueError("source_frames cannot be negative")
    count = model_frame_count(source_frames, source_fps, model_fps)
    source_centres = (np.arange(source_frames, dtype=np.float64) + 0.5) / source_fps
    indices = np.floor(source_centres * model_fps).astype(np.int64)
    return np.clip(indices, 0, max(count - 1, 0))


def align_visual_and_labels(
    visual: np.ndarray,
    labels: Sequence[int],
    source_fps: float,
    model_fps: float,
    strict: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample faces and their labels with one shared timestamp mapping.

    A length disagreement means the face track and the annotation disagree
    about how long the clip is, which silently shifts every later label.
    `strict` reports that instead of trimming to the shorter of the two.
    """
    labels_array = np.asarray(labels, dtype=np.float32)
    if strict and len(visual) != len(labels_array):
        raise ValueError(
            f"visual has {len(visual)} frames but labels have {len(labels_array)}"
        )
    usable_frames = min(len(visual), len(labels_array))
    if usable_frames == 0:
        raise ValueError("visual and labels must contain at least one frame")

    indices = source_indices(usable_frames, source_fps, model_fps)
    return visual[:usable_frames][indices], labels_array[:usable_frames][indices]


def fit_audio_to_frames(
    waveform: torch.Tensor,
    frame_count: int,
    sample_rate: int,
    model_fps: float,
) -> torch.Tensor:
    """Pad/crop audio to the exact duration represented by model frames."""
    target_samples = round(frame_count * sample_rate / model_fps)
    waveform = waveform[..., :target_samples]
    return F.pad(waveform, (0, max(0, target_samples - waveform.shape[-1])))
