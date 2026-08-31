"""Crop raw MSDWild videos into aligned 30-second speaker-track samples."""

import argparse
import csv
import shutil
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.utils import load_audio  # noqa: E402
from preprocess.asd.video import crop_face, load_visual  # noqa: E402
from preprocess.asd.writer import ShardWriter  # noqa: E402


@dataclass(frozen=True)
class Detection:
    bbox: tuple[float, float, float, float]


def read_annotations(path: Path) -> dict[int, dict[str, Detection]]:
    annotations: dict[int, dict[str, Detection]] = defaultdict(dict)
    with path.open(newline="") as handle:
        for row_number, row in enumerate(csv.reader(handle), start=1):
            if len(row) < 8:
                raise ValueError(f"{path}:{row_number}: expected 8 columns")
            try:
                frame = int(row[0])
                speaker_id = row[2]
                bbox = tuple(float(value) for value in row[3:7])
                int(row[7])
            except ValueError as error:
                raise ValueError(f"{path}:{row_number}: invalid annotation") from error
            annotations[frame][speaker_id] = Detection(bbox)
    return dict(annotations)


def read_rttm(path: Path) -> dict[str, dict[str, list[tuple[float, float]]]]:
    activity: dict[str, dict[str, list[tuple[float, float]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            parts = line.split()
            if not parts:
                continue
            if len(parts) < 8 or parts[0] != "SPEAKER":
                raise ValueError(f"{path}:{line_number}: invalid RTTM row")
            try:
                start = float(parts[3])
                duration = float(parts[4])
            except ValueError as error:
                raise ValueError(f"{path}:{line_number}: invalid time") from error
            activity[parts[1]][parts[7]].append((start, start + duration))
    return {
        video_id: {speaker: sorted(intervals) for speaker, intervals in speakers.items()}
        for video_id, speakers in activity.items()
    }


def retained_speakers(
    annotations: dict[int, dict[str, Detection]],
    start_frame: int,
    frame_count: int,
    min_face_ratio: float,
) -> list[str]:
    counts: dict[str, int] = defaultdict(int)
    for frame in range(start_frame, start_frame + frame_count):
        for speaker_id in annotations.get(frame, {}):
            counts[speaker_id] += 1
    minimum = min_face_ratio * frame_count
    return sorted(speaker_id for speaker_id, count in counts.items() if count >= minimum)


def clip_starts(first_face_frame: int, last_face_frame: int, stride_frames: int):
    return range(first_face_frame, last_face_frame + 1, stride_frames)


def video_metadata(path: Path) -> tuple[float, int]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"could not open video: {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if fps <= 0 or frame_count <= 0:
        raise ValueError(f"invalid video metadata: {path}")
    return fps, frame_count


def process_video(
    video_path: Path,
    annotation_path: Path,
    activity: dict[str, list[tuple[float, float]]],
    writer: ShardWriter,
    clip_seconds: float,
    overlap_seconds: float,
    min_face_ratio: float,
    min_clip_seconds: float,
) -> int:
    annotations = read_annotations(annotation_path)
    if not annotations:
        return 0

    fps, video_frames = video_metadata(video_path)
    clip_frames = round(clip_seconds * fps)
    stride_frames = round((clip_seconds - overlap_seconds) * fps)
    min_clip_frames = max(1, round(min_clip_seconds * fps))
    if clip_frames <= 0 or stride_frames <= 0:
        raise ValueError("clip and stride must each contain at least one frame")
    waveform, _ = load_audio(video_path)
    first_face = min(annotations)
    last_face = min(max(annotations), video_frames - 1)
    written = 0

    for segment_id, start_frame in enumerate(
        clip_starts(first_face, last_face, stride_frames)
    ):
        # Take whatever is left at the tail. Windows used to be dropped unless
        # they were complete, which discarded every video shorter than one clip
        # - about a third of MSDWild. The loader pads and chunks, so a short
        # sample costs nothing.
        window_frames = min(clip_frames, video_frames - start_frame)
        if window_frames < min_clip_frames:
            break
        speakers = retained_speakers(
            annotations, start_frame, window_frames, min_face_ratio
        )
        if not speakers:
            continue

        frames, decoded_fps = load_visual(video_path, start_frame, window_frames)
        if not np.isclose(decoded_fps, fps):
            raise ValueError(f"frame-rate changed while decoding {video_path}")
        if len(frames) < min_clip_frames:
            continue
        window_frames = len(frames)

        start_time = start_frame / fps
        start_sample = round(start_time * 16_000)
        target_samples = round(window_frames / fps * 16_000)
        audio = waveform[:, start_sample : start_sample + target_samples]

        for speaker_id in speakers:
            faces = []
            labels = np.zeros(window_frames, dtype=np.uint8)
            for offset, frame in enumerate(frames):
                detection = annotations.get(start_frame + offset, {}).get(speaker_id)
                faces.append(crop_face(frame, detection.bbox if detection else None))
            intervals = activity.get(speaker_id, [])
            for offset in range(window_frames):
                timestamp = start_time + (offset + 0.5) / fps
                labels[offset] = any(start <= timestamp < end for start, end in intervals)
            faces = np.stack(faces).astype(np.uint8, copy=False)
            sample_id = f"{video_path.stem}_{segment_id:05d}_{speaker_id}"
            writer.add_arrays(
                dataset="msdwild",
                sample_id=sample_id,
                faces=faces,
                waveform=audio,
                labels=labels,
                source_fps=fps,
                start_sec=start_time,
            )
            written += 1
    return written


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--rttm", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        help="defaults to <dataset root>/packed/<split>, beside the source media",
    )
    parser.add_argument(
        "--split",
        default="all",
        help="official split to pack: all, few.train, few.val, many.val",
    )
    parser.add_argument("--clip-seconds", type=float, default=30.0)
    # Windows no longer overlap: the loader computes projection targets over a
    # whole track before chunking, so duplicated frames buy nothing and would
    # both inflate the pack and over-weight this corpus in an epoch.
    parser.add_argument("--overlap-seconds", type=float, default=0.0)
    parser.add_argument("--min-clip-seconds", type=float, default=2.0)
    parser.add_argument("--min-face-ratio", type=float, default=0.9)
    parser.add_argument("--max-shard-gb", type=float, default=1.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.clip_seconds <= 0:
        raise ValueError("clip-seconds must be positive")
    if not 0 <= args.overlap_seconds < args.clip_seconds:
        raise ValueError("overlap-seconds must be in [0, clip-seconds)")
    if not 0 < args.min_face_ratio <= 1:
        raise ValueError("min-face-ratio must be in (0, 1]")
    if args.max_shard_gb <= 0:
        raise ValueError("max-shard-gb must be positive")
    if not 0 < args.min_clip_seconds <= args.clip_seconds:
        raise ValueError("min-clip-seconds must be in (0, clip-seconds]")
    if args.output is None:
        args.output = args.input.parent / "packed" / args.split
        print(f"packing into {args.output}")
    if args.output.exists():
        if not args.overwrite and any(args.output.iterdir()):
            raise FileExistsError(f"output directory is not empty: {args.output}")
        if args.overwrite:
            shutil.rmtree(args.output)

    args.output.mkdir(parents=True)
    videos = sorted(args.input.glob("*.mp4"))
    if not videos:
        raise FileNotFoundError(f"no MP4 files found in {args.input}")

    rttm_path = args.rttm or args.input.parent / "all.rttm"
    if not rttm_path.exists():
        raise FileNotFoundError(f"missing RTTM labels: {rttm_path}")
    activity = read_rttm(rttm_path)

    if args.split != "all":
        # The official split lists are themselves RTTMs; their video ids are
        # the split membership.
        split_path = args.input.parent / f"{args.split}.rttm"
        if not split_path.exists():
            raise FileNotFoundError(f"missing split list: {split_path}")
        wanted = set(read_rttm(split_path))
        videos = [video for video in videos if video.stem in wanted]
        if not videos:
            raise ValueError(f"no videos matched split {args.split}")
        print(f"{args.split}: {len(videos)} videos")
    total = 0
    writer = ShardWriter(
        args.output, max_shard_bytes=round(args.max_shard_gb * 1_000_000_000)
    )
    try:
        for index, video_path in enumerate(videos, start=1):
            annotation_path = video_path.with_suffix(".csv")
            if not annotation_path.exists():
                raise FileNotFoundError(f"missing annotations: {annotation_path}")
            total += process_video(
                video_path,
                annotation_path,
                activity.get(video_path.stem, {}),
                writer,
                args.clip_seconds,
                args.overlap_seconds,
                args.min_face_ratio,
                args.min_clip_seconds,
            )
            print(f"[{index}/{len(videos)}] {video_path.stem}: {total} samples")
    finally:
        writer.close()
    print(f"Wrote {total} samples to {args.output}")


if __name__ == "__main__":
    main()
