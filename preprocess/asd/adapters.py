"""Dataset-specific metadata adapters.

Adapters do not normalize signals. They only convert source layouts into
`SourceRecord`; the shared writer owns every media transformation.
"""

import ast
import csv
import json
from pathlib import Path
from typing import Iterable

from preprocess.asd.schema import SAMPLE_RATE, SourceRecord


def _labels(text: str) -> list[int]:
    values = ast.literal_eval(text)
    if not isinstance(values, (list, tuple)):
        raise ValueError("labels must be a list")
    result = [int(value) for value in values]
    if any(value not in (0, 1) for value in result):
        raise ValueError("labels must contain only 0 and 1")
    return result


def _existing(path: Path, sample_id: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"{sample_id}: missing {path}")
    return path


def clip_directory_records(
    loader_path: Path,
    visual_root: Path,
    audio_root: Path,
    split: str,
    dataset: str = "ava",
    skip_missing: bool = False,
) -> Iterable[SourceRecord]:
    """Read the AVA-style loader TSV and nested JPEG/WAV layout.

    AVA and WASD ship the identical layout - `clips_videos/<split>/<clip>/
    <entity>/<timestamp>.jpg` beside `clips_audios/<split>/<clip>/<entity>.wav`
    - and the identical five-column loader TSV, so one reader serves both.
    """
    face_by_id = {
        path.name: path for path in (visual_root / split).glob("*/*") if path.is_dir()
    }
    audio_by_id = {path.stem: path for path in (audio_root / split).glob("*/*.wav")}
    with loader_path.open(newline="") as handle:
        for row_number, row in enumerate(csv.reader(handle, delimiter="\t"), start=1):
            if len(row) < 4:
                raise ValueError(
                    f"{loader_path}:{row_number}: expected at least 4 columns"
                )
            sample_id, _, fps, label_text = row[:4]
            face_path = face_by_id.get(sample_id)
            audio_path = audio_by_id.get(sample_id)
            if face_path is None or audio_path is None:
                if skip_missing:
                    continue
                raise FileNotFoundError(
                    f"{sample_id}: missing face directory or clip-local WAV"
                )
            yield SourceRecord(
                dataset=dataset,
                sample_id=sample_id,
                visual_path=face_path,
                audio_path=audio_path,
                source_fps=float(fps),
                labels=_labels(label_text),
                visual_kind="jpeg_directory",
            )


def manifest_records(
    manifest_path: Path,
    dataset: str,
    root: Path | None = None,
    default_npy_audio_sample_rate: int = SAMPLE_RATE,
) -> Iterable[SourceRecord]:
    """Read the canonical TSV adapter used for AVCC and future datasets.

    Required header fields: sample_id, visual_path, audio_path, source_fps,
    labels. Optional: start_sec, visual_kind, audio_sample_rate.
    """
    root = root or manifest_path.parent
    with manifest_path.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"sample_id", "visual_path", "audio_path", "source_fps", "labels"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{manifest_path}: missing columns {sorted(missing)}")
        for row_number, row in enumerate(reader, start=2):
            try:
                yield SourceRecord(
                    dataset=dataset,
                    sample_id=row["sample_id"],
                    visual_path=_existing(root / row["visual_path"], row["sample_id"]),
                    audio_path=_existing(root / row["audio_path"], row["sample_id"]),
                    source_fps=float(row["source_fps"]),
                    labels=json.loads(row["labels"]),
                    start_sec=float(row.get("start_sec") or 0.0),
                    npy_audio_sample_rate=int(
                        row.get("audio_sample_rate") or default_npy_audio_sample_rate
                    ),
                    visual_kind=row.get("visual_kind") or "npy",
                )
            except (TypeError, ValueError, json.JSONDecodeError) as error:
                raise ValueError(
                    f"{manifest_path}:{row_number}: invalid row"
                ) from error
