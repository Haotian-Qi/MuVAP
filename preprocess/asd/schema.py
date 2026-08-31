"""Shared source record and packed-shard schema.

The packed format is a *decoded mirror* of the source corpus, not a
model-ready tensor set. Faces, labels, and audio are stored on the native
timeline with no gain applied, so any codebase can consume a pack without
inheriting MuVAP+'s 25 Hz model timeline or its loudness convention. Every
model-specific transform lives in `data.dataloaders.packed` instead.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

FORMAT_NAME = "muvap-asd-mmap"
FORMAT_VERSION = 4
SUPPORTED_VERSIONS = (4,)

# The model timeline. It belongs to the loader, not to the stored data, and is
# repeated here only because preprocessing reports frame counts against it.
MODEL_FPS = 25.0

SAMPLE_RATE = 16_000
FACE_SIZE = 112


@dataclass(frozen=True)
class SourceRecord:
    dataset: str
    sample_id: str
    visual_path: Path
    audio_path: Path
    source_fps: float
    labels: Sequence[int]
    start_sec: float = 0.0
    npy_audio_sample_rate: int | None = None
    visual_kind: str = "npy"

    def __post_init__(self) -> None:
        if self.source_fps <= 0:
            raise ValueError(f"{self.sample_id}: source_fps must be positive")
        if self.start_sec < 0:
            raise ValueError(f"{self.sample_id}: start_sec cannot be negative")
        if self.visual_kind not in {"npy", "jpeg_directory"}:
            raise ValueError(f"unsupported visual_kind: {self.visual_kind}")
