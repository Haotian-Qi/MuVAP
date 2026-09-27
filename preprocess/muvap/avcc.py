"""AVCC source layout to conversation records.

AVCC ships annotation layers beside the recordings rather than pre-cut media:
per-frame boxes in `bbox/<video>/<segment>.csv`, speaker activity in
`rttm/<video>/<segment>.rttm`, and the turn events the module is scored on in
one flat file. This module resolves that layout and decodes it; the writer owns
every media transformation, and nothing here normalizes a signal.

A segment is a segment if `rttm/` names it.

The speaker roster of a segment is the union of its boxes and its activity
file, not either one alone. A speaker who is heard but never tracked still has to occupy
a row, because the global label is defined over everyone in the conversation -
the model simply sees a blank face there and the visibility flag says why.
"""

import csv
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from data.media import AudioSpec, load_waveform
from preprocess.asd.video import crop_face
from preprocess.muvap.schema import (
    FACE_SIZE,
    SAMPLE_RATE,
    ConversationRecord,
    EventAnnotation,
)

#: Face crops for a speaker with no box in a frame. The frontend normalizes to
#: its own mean, so mid-grey is the least informative thing to hand it.
BLANK_FACE = 128

#: The canonical media layout, `orig_video/<video>/<segment>.mp4` beside
#: `orig_audio/<video>/<segment>.wav`. AVCC keeps recordings outside the
#: annotation repository, so the root is resolved rather than assumed and
#: `--media-root` overrides it; `videos/` is accepted as the older spelling.
VIDEO_DIRS = ("orig_video", "videos", "../orig_video", "../videos")
AUDIO_DIR = "orig_audio"

#: Where speaker activity is read from: `rttm/<video>/<segment>.rttm`.
VAD_DIR = "rttm"


def speaker_key(speaker: str):
    """Order speakers numerically, and put any non-numeric label after them.

    Most speakers are numbered and line up with the bounding-box ids, but the
    annotation also uses bare labels - `B` for an off-screen voice - which have
    no box and never take a turn. They still occupy a row, because the global
    label is defined over everyone who is audible.
    """
    return (0, int(speaker), "") if speaker.isdigit() else (1, 0, speaker)


def read_boxes(path: Path) -> dict[int, dict[str, tuple[float, float, float, float]]]:
    """`frame_idx -> speaker -> pixel xyxy`, straight off the annotation CSV."""
    boxes: dict[int, dict[str, tuple[float, float, float, float]]] = defaultdict(dict)
    with Path(path).open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"frame_idx", "speaker_id", "x1", "y1", "x2", "y2"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path}: missing columns {sorted(missing)}")
        for row in reader:
            speaker = str(int(row["speaker_id"]))
            boxes[int(row["frame_idx"])][speaker] = (
                float(row["x1"]),
                float(row["y1"]),
                float(row["x2"]),
                float(row["y2"]),
            )
    return boxes


def read_rttm(path: Path, source_frames: int, source_fps: float) -> dict[str, np.ndarray]:
    """`speaker -> [frames]` activity, rasterized onto the native timeline."""
    activity: dict[str, np.ndarray] = {}
    with Path(path).open() as handle:
        for line in handle:
            parts = line.split()
            if not parts:
                continue
            if len(parts) < 8:
                raise ValueError(f"{path}: malformed RTTM line: {line.strip()!r}")
            # The speaker field is a label, not a number: most are numbered to
            # match the boxes, but `B` and friends are not, and coercing them
            # would drop a voice the global label has to account for.
            start, duration, speaker = float(parts[3]), float(parts[4]), parts[7]
            row = activity.setdefault(
                speaker, np.zeros(source_frames, dtype=np.uint8)
            )
            first = max(0, min(round(start * source_fps), source_frames))
            last = max(0, min(round((start + duration) * source_fps), source_frames))
            row[first:last] = 1
    return activity


def read_events(path: Path) -> dict[tuple[str, str], list[dict]]:
    """Group the flat turn-event file by the segment each event belongs to.

    Eight whitespace-separated columns:

    ```text
    video_id  segment_id  previous  start  duration  following  label  n_speakers
    ```

    `start` and `duration` describe the mutual silence between the two turns,
    and `previous` and `following` are the speakers on either side of it - so a
    hold is exactly an event whose two speakers are the same one, which
    `preprocess.muvap.validate` checks against the label. `n_speakers` is the
    event's speaker count; only the benchmark scorer reads it.
    """
    events: dict[tuple[str, str], list[dict]] = defaultdict(list)
    with Path(path).open() as handle:
        for number, line in enumerate(handle, start=1):
            parts = line.split()
            if not parts:
                continue
            if len(parts) != 8:
                raise ValueError(
                    f"{path}:{number}: expected 8 columns "
                    "(video segment previous start duration following label "
                    f"n_speakers), got {len(parts)}"
                )
            video, segment, previous, start, duration, following, label = parts[:7]
            events[(video, segment)].append(
                {
                    "gap_start": float(start),
                    "gap_duration": float(duration),
                    "annotation": EventAnnotation(
                        label=label,
                        previous=str(int(previous)),
                        following=str(int(following)),
                        pause=float(duration),
                    ),
                }
            )
    return events


def event_window(
    gap_start: float,
    source_fps: float,
    source_frames: int,
    window_sec: float,
) -> tuple[int, int] | None:
    """The window a turn event is judged on, or None if it lies off the segment.

    It ends where the mutual silence begins - the corpus already trims 0.1 s of
    padding off each side when it extracts an event, so this is the padded edge
    the VAP module's `event_crop` arrives at by adding that padding itself - and
    reaches `window_sec` back from there. The last frame the model is given is
    the last frame of the previous speaker's turn, and none of the silence is
    shown: how long the pause runs is part of the answer.

    An event closer to the start of its segment than `window_sec` keeps the
    history it does have rather than being dropped. That truncation is genuine -
    the recording really does start there - and it is the same thing the loader
    does at a track's first frames; padding it would fabricate speech instead.
    Every annotated event is therefore scored.
    """
    last = round(gap_start * source_fps)
    if last <= 0 or last > source_frames:
        return None
    return max(0, last - round(window_sec * source_fps)), last


class AVCCSegments:
    """One AVCC segment, decoded once and sliced on demand."""

    def __init__(
        self,
        video_id: str,
        segment_id: str,
        video_path: Path,
        audio_path: Path,
        bbox_path: Path,
        rttm_path: Path,
        work_dir: Path | None = None,
    ):
        self.video_id = video_id
        self.segment_id = segment_id
        self.video_path = Path(video_path)
        self.audio_path = Path(audio_path)
        self.boxes = read_boxes(bbox_path)

        capture = cv2.VideoCapture(str(self.video_path))
        if not capture.isOpened():
            raise FileNotFoundError(f"could not open video: {self.video_path}")
        self.source_fps = float(capture.get(cv2.CAP_PROP_FPS))
        self.source_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self.height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        if self.source_fps <= 0 or self.source_frames <= 0:
            capture.release()
            raise ValueError(f"video has no usable timeline: {self.video_path}")

        self.activity = read_rttm(rttm_path, self.source_frames, self.source_fps)
        tracked = {speaker for frame in self.boxes.values() for speaker in frame}
        self.speaker_ids = sorted(tracked | set(self.activity), key=speaker_key)
        if not self.speaker_ids:
            capture.release()
            raise ValueError(
                f"{video_id}/{segment_id}: neither layer names a speaker"
            )

        self._waveform = None
        self._scratch = None
        self.faces, self.bbox, self.visible = self._decode(capture, work_dir)
        # Segment-level, so a face merely absent from one window still counts as
        # trackable. Only a speaker the recording never boxes is not.
        self.tracked = self.visible.any(axis=1)
        capture.release()
        self.vad = np.stack(
            [
                self.activity.get(speaker, np.zeros(self.source_frames, dtype=np.uint8))
                for speaker in self.speaker_ids
            ]
        )

    def _decode(self, capture, work_dir: Path | None = None):
        """Crop every tracked face in every frame, in one sequential pass.

        Decoding is sequential because seeking a long recording per window costs
        more than reading it straight through. The crops do not fit in memory
        for the segments that matter - this corpus runs to half an hour, which
        is 1.7 GB of faces at three speakers - so they are written to a scratch
        array on disk and read back a window at a time. Nothing downstream can
        tell the difference: a memory-mapped array indexes exactly like a
        resident one, and the writer streams it out in blocks.
        """
        speakers = len(self.speaker_ids)
        shape = (speakers, self.source_frames, FACE_SIZE, FACE_SIZE)
        if work_dir is not None:
            Path(work_dir).mkdir(parents=True, exist_ok=True)
            self._scratch = Path(work_dir) / f".decode-{self.video_id}-{self.segment_id}.npy"
            faces = np.lib.format.open_memmap(
                self._scratch, mode="w+", dtype=np.uint8, shape=shape
            )
            faces[:] = BLANK_FACE
        else:
            faces = np.full(shape, BLANK_FACE, dtype=np.uint8)
        bbox = np.zeros((speakers, self.source_frames, 4), dtype=np.float32)
        visible = np.zeros((speakers, self.source_frames), dtype=bool)
        scale = np.array(
            [1.0 / self.width, 1.0 / self.height] * 2, dtype=np.float32
        )

        for index in range(self.source_frames):
            ok, frame = capture.read()
            if not ok:
                # The container's frame count can overstate what decodes. The
                # remaining frames keep their blank faces and unset visibility,
                # which is the same thing an untracked speaker gets.
                break
            present = self.boxes.get(index, {})
            for row, speaker in enumerate(self.speaker_ids):
                box = present.get(speaker)
                if box is None:
                    continue
                faces[row, index] = crop_face(frame, box)
                bbox[row, index] = np.asarray(box, dtype=np.float32) * scale
                visible[row, index] = True
        return faces, bbox, visible

    def close(self) -> None:
        """Release the decoded faces and remove the scratch array."""
        self.faces = None
        if self._scratch is not None and self._scratch.exists():
            self._scratch.unlink()
            self._scratch = None

    def waveform(self):
        """The segment's mixed recording, at its own gain, read once."""
        if self._waveform is None:
            self._waveform, _ = load_waveform(
                self.audio_path, AudioSpec(sample_rate=SAMPLE_RATE, mono=True)
            )
        return self._waveform

    def record(self, sample_id: str, start_sec: float = 0.0):
        return ConversationRecord(
            dataset="avcc",
            sample_id=sample_id,
            video_id=self.video_id,
            segment_id=self.segment_id,
            speaker_ids=self.speaker_ids,
            source_fps=self.source_fps,
            start_sec=start_sec,
            tracked=self.tracked.tolist(),
        )

    def whole(self):
        """The segment as one sample; the loader chunks it at training time."""
        sample_id = f"{self.video_id}+{self.segment_id}"
        return (
            self.record(sample_id),
            self.faces,
            self.waveform(),
            self.vad,
            self.bbox,
            self.visible,
        )

    def window(self, first: int, last: int, sample_id: str):
        """One anchored span of the segment, with its own slice of the audio."""
        start_sec = first / self.source_fps
        samples_first = round(start_sec * SAMPLE_RATE)
        samples_last = round(last / self.source_fps * SAMPLE_RATE)
        return (
            self.record(sample_id, start_sec=start_sec),
            self.faces[:, first:last],
            self.waveform()[:, samples_first:samples_last],
            self.vad[:, first:last],
            self.bbox[:, first:last],
            self.visible[:, first:last],
        )


def media_root(root: Path, override: Path | None = None) -> Path:
    """The directory holding `<video>/<segment>.mp4`, resolved or given."""
    if override is not None:
        return Path(override)
    root = Path(root)
    for name in VIDEO_DIRS:
        candidate = (root / name).resolve()
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        f"no segment video beside {root}; looked for {list(VIDEO_DIRS)}, "
        "pass --media-root"
    )


def audio_path(root: Path, video_id: str, segment_id: str, recording: Path) -> Path:
    """The segment's audio: its own WAV where one exists, else the video itself.

    A separate `orig_audio/` WAV is the canonical layout and is what a corpus
    with cleaned or separately mastered audio should ship. Falling back to the
    MP4 is not a lesser path - the shared loader brings either to 16 kHz mono
    the same way - it just saves extracting a track that is already there.
    """
    wav = Path(root) / AUDIO_DIR / video_id / f"{segment_id}.wav"
    return wav if wav.exists() else recording


def vad_path(root: Path, video_id: str, segment_id: str) -> Path:
    """The activity file for one segment."""
    return Path(root) / VAD_DIR / video_id / f"{segment_id}.rttm"


def segment_paths(root: Path, video_id: str) -> list[tuple[str, str]]:
    """Every `(video, segment)` pair the activity files name for one video."""
    directory = Path(root) / VAD_DIR / video_id
    segments = directory.glob("*.rttm") if directory.is_dir() else []
    return [(video_id, segment) for segment in sorted(path.stem for path in segments)]


def read_splits(path: Path) -> dict[str, str]:
    """`video_id -> split` from the headered two-column split file."""
    splits = {}
    with Path(path).open() as handle:
        for number, line in enumerate(handle, start=1):
            parts = line.split()
            if not parts or parts[0] == "videoID":
                continue
            if len(parts) != 2:
                raise ValueError(f"{path}:{number}: expected 'video_id split'")
            splits[parts[0]] = parts[1]
    return splits


def open_segment(
    root: Path,
    video_id: str,
    segment_id: str,
    media: Path | None = None,
    work_dir: Path | None = None,
) -> AVCCSegments:
    """Resolve a segment's four inputs, naming every one that is absent.

    Checked up front rather than left to whichever reader runs first: a segment
    annotated in one layer and not another is a gap in the corpus, and the
    useful message says which file, not which line of which parser.
    """
    root = Path(root)
    recording = media_root(root, media) / video_id / f"{segment_id}.mp4"
    paths = {
        "video_path": recording,
        "audio_path": audio_path(root, video_id, segment_id, recording),
        "bbox_path": root / "bbox" / video_id / f"{segment_id}.csv",
        "rttm_path": vad_path(root, video_id, segment_id),
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"{video_id}/{segment_id} is missing {len(missing)} of its inputs: "
            + ", ".join(missing)
        )
    return AVCCSegments(
        video_id=video_id,
        segment_id=segment_id,
        work_dir=work_dir,
        **paths,
    )
