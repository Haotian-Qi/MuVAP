import csv
import logging
from collections import Counter, defaultdict
from pathlib import Path

import torch

from data.alignment import model_frame_count, model_indices
from tasks.asd_metrics import ava_active_speaker_map

logger = logging.getLogger(__name__)

MODEL_FPS = 25.0

PREDICTION_FIELDS = [
    "video_id",
    "frame_timestamp",
    "entity_box_x1",
    "entity_box_y1",
    "entity_box_x2",
    "entity_box_y2",
    "label",
    "entity_id",
    "score",
]


def _lift_to_native(entity_id, scores, native_frames, source_fps, model_fps):
    """Expand model-rate scores to one score per native groundtruth row."""
    expected_frames = model_frame_count(native_frames, source_fps, model_fps)
    if len(scores) != expected_frames:
        raise ValueError(
            f"{entity_id}: {native_frames} groundtruth rows at {source_fps} fps imply "
            f"{expected_frames} model frames, got {len(scores)} scores"
        )
    indices = model_indices(native_frames, source_fps, model_fps)
    return scores[torch.as_tensor(indices, dtype=torch.long)]


def _count_drifting_entities(rows, fps_by_entity, model_fps):
    """Count entities whose groundtruth timestamps are not uniformly spaced.

    The lift assumes native frame `j` sits at `t0 + j / source_fps`, matching
    the positional mapping used when the sample was packed. A few AVA entities
    have gaps in their annotation timeline; they are scored positionally
    anyway, but the count belongs in the log so the number stays auditable.
    """
    timestamps = defaultdict(list)
    for row in rows:
        timestamps[row["entity_id"]].append(float(row["frame_timestamp"]))
    tolerance = 1.0 / model_fps
    drifting = 0
    for entity_id, values in timestamps.items():
        source_fps = fps_by_entity.get(entity_id)
        if not source_fps:
            continue
        origin = values[0]
        if any(
            abs((value - origin) - index / source_fps) > tolerance
            for index, value in enumerate(values)
        ):
            drifting += 1
    return drifting


def export_ava_predictions(
    groundtruth_path,
    output_path,
    scores_by_entity,
    fps_by_entity=None,
    model_fps=MODEL_FPS,
):
    """Write official AVA predictions and return the mAP over every row.

    `scores_by_entity` holds one score per model frame. When `fps_by_entity` is
    given, scores are lifted onto each entity's native annotation timeline
    first, so the exported file has exactly one row per groundtruth row.
    """
    groundtruth_path = Path(groundtruth_path)
    output_path = Path(output_path)
    rows = list(csv.DictReader(groundtruth_path.open()))
    expected = Counter(row["entity_id"] for row in rows)
    missing = expected.keys() - scores_by_entity.keys()
    extra = scores_by_entity.keys() - expected.keys()
    if missing or extra:
        raise ValueError(f"AVA entity mismatch: {len(missing)} missing, {len(extra)} extra")

    if fps_by_entity is not None:
        absent = expected.keys() - fps_by_entity.keys()
        if absent:
            raise ValueError(f"AVA source fps missing for {len(absent)} entities")
        scores_by_entity = {
            entity_id: _lift_to_native(
                entity_id,
                scores_by_entity[entity_id],
                expected[entity_id],
                fps_by_entity[entity_id],
                model_fps,
            )
            for entity_id in expected
        }
        drifting = _count_drifting_entities(rows, fps_by_entity, model_fps)
        if drifting:
            logger.info(
                "%d of %d AVA entities have non-uniform groundtruth timestamps; "
                "scored positionally",
                drifting,
                len(expected),
            )

    for entity_id, count in expected.items():
        if len(scores_by_entity[entity_id]) != count:
            raise ValueError(f"{entity_id}: expected {count} scores, got {len(scores_by_entity[entity_id])}")

    positions = defaultdict(int)
    labels = []
    scores = []
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PREDICTION_FIELDS)
        writer.writeheader()
        for row in rows:
            entity_id = row["entity_id"]
            score = float(scores_by_entity[entity_id][positions[entity_id]])
            positions[entity_id] += 1
            prediction = {field: row[field] for field in PREDICTION_FIELDS[:-2]}
            writer.writerow({**prediction, "label": "SPEAKING_AUDIBLE", "entity_id": entity_id, "score": score})
            labels.append(row["label"] == "SPEAKING_AUDIBLE")
            scores.append(score)
    return ava_active_speaker_map(torch.tensor(scores), torch.tensor(labels))
