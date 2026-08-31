"""Pack decoded native-rate samples into memory-mapped NumPy shards.

Nothing here is specific to the model that will read the shards. Faces and
labels keep the source frame rate, audio keeps its full duration and its
original gain, and the per-recording level statistics needed to reproduce a
normalization travel in the index. Resampling onto a model timeline and
applying loudness normalization are the reader's job.
"""

import json
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import torch

from data.alignment import model_frame_count
from data.media import (
    AudioSpec,
    load_waveform,
    slice_audio,
    validate_visual,
    waveform_stats,
)
from preprocess.asd.schema import (
    FACE_SIZE,
    FORMAT_NAME,
    FORMAT_VERSION,
    MODEL_FPS,
    SAMPLE_RATE,
    SourceRecord,
)


def _load_faces(record: SourceRecord, face_size: int = FACE_SIZE) -> np.ndarray:
    if record.visual_kind == "npy":
        return validate_visual(np.load(record.visual_path, allow_pickle=False))

    paths = sorted(record.visual_path.glob("*.jpg"), key=lambda path: float(path.stem))
    if not paths:
        raise FileNotFoundError(f"{record.sample_id}: no JPEG faces")
    frames = []
    for path in paths:
        frame = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if frame is None:
            raise ValueError(f"{record.sample_id}: could not decode {path}")
        frames.append(
            cv2.resize(frame, (face_size, face_size), interpolation=cv2.INTER_LINEAR)
        )
    return np.stack(frames).astype(np.uint8, copy=False)


def _pcm16(waveform: torch.Tensor) -> np.ndarray:
    values = waveform.squeeze(0).clamp(-1.0, 1.0)
    return (values * 32767.0).round().to(torch.int16).cpu().numpy()


def native_audio_samples(source_frames: int, source_fps: float) -> int:
    """Audio length covering exactly the span of the stored face frames."""
    return round(source_frames / source_fps * SAMPLE_RATE)


class ShardWriter:
    def __init__(self, output_dir: Path, max_shard_bytes: int = 1_000_000_000):
        self.output_dir = output_dir
        self.max_shard_bytes = max_shard_bytes
        self.shard_index = 0
        self.samples_written = 0
        self._reset()
        output_dir.mkdir(parents=True, exist_ok=True)

    def _reset(self) -> None:
        self.faces: list[np.ndarray] = []
        self.audio: list[np.ndarray] = []
        self.labels: list[np.ndarray] = []
        self.entries: list[dict] = []
        self.byte_count = 0
        self.face_count = 0
        self.audio_count = 0
        self.label_count = 0

    def add_arrays(
        self,
        *,
        dataset: str,
        sample_id: str,
        faces: np.ndarray,
        waveform: torch.Tensor,
        labels,
        source_fps: float,
        start_sec: float = 0.0,
    ) -> None:
        faces = validate_visual(faces)
        labels = np.asarray(labels, dtype=np.uint8)
        if len(faces) != len(labels):
            raise ValueError(
                f"{sample_id}: {len(faces)} faces but {len(labels)} labels"
            )
        source_frames = len(labels)
        if source_frames == 0:
            raise ValueError(f"{sample_id}: sample is empty")

        waveform = slice_audio(
            waveform,
            SAMPLE_RATE,
            target_samples=native_audio_samples(source_frames, source_fps),
        )
        stats = waveform_stats(waveform)
        audio = _pcm16(waveform)

        sample_bytes = faces.nbytes + audio.nbytes + labels.nbytes
        if self.entries and self.byte_count + sample_bytes > self.max_shard_bytes:
            self.flush()

        self.entries.append(
            {
                "dataset": dataset,
                "sample_id": sample_id,
                "source_fps": source_fps,
                "source_frames": source_frames,
                "start_sec": start_sec,
                # Reported for convenience; the loader recomputes it.
                "model_frames": model_frame_count(source_frames, source_fps, MODEL_FPS),
                "face_offset": self.face_count,
                "audio_offset": self.audio_count,
                "audio_samples": len(audio),
                "label_offset": self.label_count,
                "audio_mean": stats.mean,
                "audio_rms": stats.rms,
                "audio_peak": stats.peak,
                "speaking_ratio": float(labels.mean()),
            }
        )
        self.faces.append(faces)
        self.audio.append(audio)
        self.labels.append(labels)
        self.byte_count += sample_bytes
        self.face_count += len(faces)
        self.audio_count += len(audio)
        self.label_count += len(labels)

    def add(self, record: SourceRecord) -> None:
        faces = _load_faces(record)
        waveform, sample_rate = load_waveform(
            record.audio_path,
            AudioSpec(sample_rate=SAMPLE_RATE, mono=True),
            npy_sample_rate=record.npy_audio_sample_rate,
        )
        waveform = slice_audio(waveform, sample_rate, start_time=record.start_sec)
        self.add_arrays(
            dataset=record.dataset,
            sample_id=record.sample_id,
            faces=faces,
            waveform=waveform,
            labels=record.labels,
            source_fps=record.source_fps,
            start_sec=record.start_sec,
        )

    def flush(self) -> None:
        if not self.entries:
            return
        shard_dir = self.output_dir / f"shard-{self.shard_index:06d}"
        shard_dir.mkdir(parents=False, exist_ok=False)
        np.save(shard_dir / "faces.npy", np.concatenate(self.faces), allow_pickle=False)
        np.save(shard_dir / "audio.npy", np.concatenate(self.audio), allow_pickle=False)
        np.save(
            shard_dir / "labels.npy", np.concatenate(self.labels), allow_pickle=False
        )
        with (shard_dir / "index.json").open("w") as handle:
            json.dump(self.entries, handle, separators=(",", ":"))
        self.samples_written += len(self.entries)
        self.shard_index += 1
        self._reset()

    def close(self) -> None:
        self.flush()
        manifest = {
            "format": FORMAT_NAME,
            "version": FORMAT_VERSION,
            "timeline": "native",
            "audio_sample_rate": SAMPLE_RATE,
            "audio_normalization": None,
            "face_shape": [FACE_SIZE, FACE_SIZE],
            "face_color": "grayscale",
            "face_dtype": "uint8",
            "audio_dtype": "int16",
            "label_dtype": "uint8",
            "shards": self.shard_index,
            "samples": self.samples_written,
        }
        with (self.output_dir / "dataset.json").open("w") as handle:
            json.dump(manifest, handle, indent=2)


def write_records(
    records: Iterable[SourceRecord], output_dir: Path, max_shard_bytes: int
) -> int:
    writer = ShardWriter(output_dir, max_shard_bytes)
    for record in records:
        writer.add(record)
    writer.close()
    return writer.samples_written
