"""Pack decoded multiparty segments into memory-mapped NumPy shards.

The layout mirrors `preprocess.asd.writer`, with one difference that follows
from the data: a sample has a speaker axis. Faces, labels, and boxes are stored
speaker-major - all of speaker 0's frames, then all of speaker 1's - so a
reader that wants one speaker's window reads one contiguous run rather than a
stride over the whole segment. Audio has no speaker axis at all: the segment is
one mixed recording, which is what both frozen modules were trained to read.

Nothing here is specific to the model that will consume the shards, and nothing
here normalizes a signal. The per-recording level statistics needed to
reproduce a normalization travel in the index instead.

Faces never sit in memory as a whole shard. They are appended to a scratch file
block by block as each segment arrives and the `.npy` is written out of it at
flush, so the crops cost a fixed buffer instead of growing with the shard - a
half-hour segment's are 1.1 GB at two speakers, and a shard holds several.

That bounds the faces, not the whole job. Decoding a half-hour of 44.1 kHz
stereo down to 16 kHz mono peaks around 2.6 GB in the audio loader, which is the
larger cost by far and is left alone deliberately: every module packs audio
through that one path, and resampling it in pieces would make MuVAP's audio
differ from the ASD and VAP packs in the last decimal place. Packing the longest
segment in AVCC peaks near 3.6 GB all told.
"""

import json
from pathlib import Path

import numpy as np
import torch
from numpy.lib import format as npy_format

from data.alignment import model_frame_count
from data.media import slice_audio, validate_visual, waveform_stats
from preprocess.muvap.schema import (
    EMBEDDING_SOURCE,
    EVENT_KIND,
    FACE_SIZE,
    MEDIA_SOURCE,
    MODEL_FPS,
    SAMPLE_RATE,
    SEGMENT_KIND,
    ConversationRecord,
    EventAnnotation,
    manifest,
)


def _pcm16(waveform: torch.Tensor) -> np.ndarray:
    values = waveform.squeeze(0).clamp(-1.0, 1.0)
    return (values * 32767.0).round().to(torch.int16).cpu().numpy()


#: Faces are copied to and from disk in blocks of this many frames, which is
#: what bounds packing memory regardless of how long a segment runs.
FACE_BLOCK = 2000


def native_audio_samples(source_frames: int, source_fps: float) -> int:
    """Audio length covering exactly the span of the stored face frames."""
    return round(source_frames / source_fps * SAMPLE_RATE)


class ConversationShardWriter:
    def __init__(
        self,
        output_dir: Path,
        kind: str = SEGMENT_KIND,
        max_shard_bytes: int = 1_000_000_000,
    ):
        if kind not in (SEGMENT_KIND, EVENT_KIND):
            raise ValueError(f"unknown pack kind: {kind!r}")
        self.output_dir = Path(output_dir)
        self.kind = kind
        self.max_shard_bytes = max_shard_bytes
        self.shard_index = 0
        self.samples_written = 0
        self._reset()
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _reset(self) -> None:
        self.audio: list[np.ndarray] = []
        self.labels: list[np.ndarray] = []
        self.boxes: list[np.ndarray] = []
        self.entries: list[dict] = []
        self.byte_count = 0
        self.face_count = 0
        self.audio_count = 0
        self.label_count = 0
        self._faces_path = self.output_dir / f".faces-{self.shard_index:06d}.raw"
        self._faces_file = None

    def _append_faces(self, faces: np.ndarray) -> None:
        """Stream one sample's crops to the shard's scratch file, block by block.

        `faces` may be memory-mapped, so it is never asked for all at once.
        """
        if self._faces_file is None:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self._faces_file = self._faces_path.open("wb")
        for track in faces:
            for start in range(0, len(track), FACE_BLOCK):
                block = np.ascontiguousarray(track[start : start + FACE_BLOCK])
                self._faces_file.write(block.tobytes())

    def _write_faces(self, shard_dir: Path) -> None:
        """Turn the scratch file into a real `.npy`, without loading it."""
        self._faces_file.close()
        self._faces_file = None
        header = {
            "descr": npy_format.dtype_to_descr(np.dtype(np.uint8)),
            "fortran_order": False,
            "shape": (self.face_count, FACE_SIZE, FACE_SIZE),
        }
        with (shard_dir / "faces.npy").open("wb") as out:
            npy_format.write_array_header_2_0(out, header)
            with self._faces_path.open("rb") as scratch:
                while block := scratch.read(FACE_BLOCK * FACE_SIZE * FACE_SIZE):
                    out.write(block)
        self._faces_path.unlink()

    def add(
        self,
        record: ConversationRecord,
        faces: np.ndarray,
        waveform: torch.Tensor,
        vad: np.ndarray,
        bbox: np.ndarray,
        visible: np.ndarray,
        event: EventAnnotation | None = None,
    ) -> None:
        """Append one segment.

        `faces` is `[speakers, frames, 112, 112]` uint8, `vad` and `visible` are
        `[speakers, frames]`, and `bbox` is `[speakers, frames, 4]` normalized
        xyxy. `waveform` is the segment's mixed recording at its original gain.
        """
        speakers = len(record.speaker_ids)
        faces = np.asarray(faces)
        if faces.ndim != 4 or faces.shape[0] != speakers:
            raise ValueError(
                f"{record.sample_id}: faces must be [speakers, frames, {FACE_SIZE}, "
                f"{FACE_SIZE}] for {speakers} speakers, got {faces.shape}"
            )
        source_frames = faces.shape[1]
        if source_frames == 0:
            raise ValueError(f"{record.sample_id}: sample is empty")
        # One validation per speaker: the shared checker owns the geometry and
        # the dtype, and it only speaks about a single track.
        for track in faces:
            validate_visual(track)

        vad = np.asarray(vad, dtype=np.uint8)
        visible = np.asarray(visible).astype(bool)
        bbox = np.asarray(bbox, dtype=np.float32)
        for name, array, shape in (
            ("vad", vad, (speakers, source_frames)),
            ("visible", visible, (speakers, source_frames)),
            ("bbox", bbox, (speakers, source_frames, 4)),
        ):
            if array.shape != shape:
                raise ValueError(
                    f"{record.sample_id}: {name} must be {shape}, got {array.shape}"
                )

        waveform = slice_audio(
            waveform,
            SAMPLE_RATE,
            target_samples=native_audio_samples(source_frames, record.source_fps),
        )
        stats = waveform_stats(waveform)
        audio = _pcm16(waveform)

        # Visibility rides along in the spare bit of the label byte rather than
        # in an array of its own: it is one flag per speaker-frame, and keeping
        # it beside the activity it qualifies means a reader cannot pick up one
        # without the other.
        packed_labels = (vad & 1) | (visible.astype(np.uint8) << 1)

        sample_bytes = faces.nbytes + audio.nbytes + packed_labels.nbytes + bbox.nbytes
        if self.entries and self.byte_count + sample_bytes > self.max_shard_bytes:
            self.flush()

        entry = {
            "dataset": record.dataset,
            "sample_id": record.sample_id,
            "video_id": record.video_id,
            "segment_id": record.segment_id,
            "speaker_ids": list(record.speaker_ids),
            "speakers": speakers,
            "tracked": [bool(flag) for flag in (
                record.tracked if record.tracked is not None else visible.any(axis=1)
            )],
            "source_fps": record.source_fps,
            "source_frames": source_frames,
            "start_sec": record.start_sec,
            # Reported for convenience; the loader recomputes it.
            "model_frames": model_frame_count(
                source_frames, record.source_fps, MODEL_FPS
            ),
            "face_offset": self.face_count,
            "audio_offset": self.audio_count,
            "audio_samples": len(audio),
            "label_offset": self.label_count,
            "audio_mean": stats.mean,
            "audio_rms": stats.rms,
            "audio_peak": stats.peak,
            "speaking_ratio": float(vad.mean()),
            "visible_ratio": float(visible.mean()),
        }
        if (event is None) != (self.kind == SEGMENT_KIND):
            raise ValueError(
                f"{record.sample_id}: a {self.kind!r} pack "
                f"{'must not' if event is None else 'must'} carry turn-event annotation"
            )
        if event is not None:
            for role, speaker in (("previous", event.previous), ("following", event.following)):
                if speaker not in record.speaker_ids:
                    raise ValueError(
                        f"{record.sample_id}: {role} speaker {speaker!r} is not one of "
                        f"the segment's speakers {list(record.speaker_ids)}"
                    )
            entry["event"] = {
                "label": event.label,
                "previous": event.previous,
                "following": event.following,
                "pause": event.pause,
            }

        self.entries.append(entry)
        self._append_faces(faces)
        self.audio.append(audio)
        self.labels.append(packed_labels.reshape(-1))
        self.boxes.append(bbox.reshape(-1, 4))
        self.byte_count += sample_bytes
        self.face_count += speakers * source_frames
        self.audio_count += len(audio)
        self.label_count += speakers * source_frames

    def flush(self) -> None:
        if not self.entries:
            return
        shard_dir = self.output_dir / f"shard-{self.shard_index:06d}"
        shard_dir.mkdir(parents=False, exist_ok=False)
        self._write_faces(shard_dir)
        np.save(shard_dir / "audio.npy", np.concatenate(self.audio), allow_pickle=False)
        np.save(
            shard_dir / "labels.npy", np.concatenate(self.labels), allow_pickle=False
        )
        np.save(shard_dir / "bbox.npy", np.concatenate(self.boxes), allow_pickle=False)
        with (shard_dir / "index.json").open("w") as handle:
            json.dump(self.entries, handle, separators=(",", ":"))
        self.samples_written += len(self.entries)
        self.shard_index += 1
        self._reset()

    def close(self) -> None:
        self.flush()
        with (self.output_dir / "dataset.json").open("w") as handle:
            json.dump(
                manifest(
                    self.kind, MEDIA_SOURCE, self.shard_index, self.samples_written
                ),
                handle,
                indent=2,
            )


class EmbeddingShardWriter:
    """Write what the frozen modules produced, in the layout the reader expects.

    Deliberately parallel to `ConversationShardWriter`: same sharding, same
    speaker-major order, same packed label byte, same entry keys where they
    mean the same thing. What differs is forced by the content. Features are on
    the model timeline rather than the corpus's, so `model_frames` is the only
    frame count there is; and the global stream has no speaker axis, so it gets
    its own offset rather than sharing the per-speaker one.

    An embedding pack carries its labels, boxes, and visibility so it stands on
    its own. They cost one byte and sixteen per speaker-frame against several
    kilobytes of features, and needing the media pack beside it to train would
    defeat the point of having one.
    """

    def __init__(
        self,
        output_dir: Path,
        kind: str = SEGMENT_KIND,
        max_shard_bytes: int = 1_000_000_000,
        provenance: dict | None = None,
    ):
        if kind not in (SEGMENT_KIND, EVENT_KIND):
            raise ValueError(f"unknown pack kind: {kind!r}")
        self.output_dir = Path(output_dir)
        self.kind = kind
        self.max_shard_bytes = max_shard_bytes
        self.provenance = provenance or {}
        self.shard_index = 0
        self.samples_written = 0
        self._reset()
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _reset(self) -> None:
        self.vap: list[np.ndarray] = []
        self.asd: list[np.ndarray] = []
        self.labels: list[np.ndarray] = []
        self.boxes: list[np.ndarray] = []
        self.entries: list[dict] = []
        self.byte_count = 0
        self.vap_count = 0
        self.asd_count = 0

    def add(
        self,
        entry: dict,
        vap: np.ndarray,
        asd: np.ndarray,
        vad: np.ndarray,
        bbox: np.ndarray,
        visible: np.ndarray,
    ) -> None:
        """Append one sample's features beside the media entry they came from.

        `vap` is `[frames, vap_dim]` and `asd` is `[speakers, frames, asd_dim]`,
        both on the model timeline. `vad`, `visible`, and `bbox` are the media
        pack's own labels already sampled onto that timeline, so the two packs
        answer identically and a run can switch between them.
        """
        speakers = entry["speakers"]
        vap = np.asarray(vap, dtype=np.float32)
        asd = np.asarray(asd, dtype=np.float32)
        if vap.ndim != 2:
            raise ValueError(f"{entry['sample_id']}: vap must be [frames, dim]")
        frames = vap.shape[0]
        if asd.shape[:2] != (speakers, frames):
            raise ValueError(
                f"{entry['sample_id']}: asd must be [{speakers}, {frames}, dim], "
                f"got {asd.shape}"
            )

        vad = np.asarray(vad, dtype=np.uint8)
        visible = np.asarray(visible).astype(bool)
        bbox = np.asarray(bbox, dtype=np.float32)
        for name, array, shape in (
            ("vad", vad, (speakers, frames)),
            ("visible", visible, (speakers, frames)),
            ("bbox", bbox, (speakers, frames, 4)),
        ):
            if array.shape != shape:
                raise ValueError(
                    f"{entry['sample_id']}: {name} must be {shape}, got {array.shape}"
                )
        packed_labels = (vad & 1) | (visible.astype(np.uint8) << 1)

        sample_bytes = vap.nbytes + asd.nbytes + packed_labels.nbytes + bbox.nbytes
        if self.entries and self.byte_count + sample_bytes > self.max_shard_bytes:
            self.flush()

        # Carry the media entry forward so a sample is traceable to the segment
        # it was cut from, and drop the offsets, which describe the other pack.
        carried = {
            key: value
            for key, value in entry.items()
            if key not in {"face_offset", "audio_offset", "label_offset", "audio_samples"}
        }
        self.entries.append(
            carried
            | {
                "model_frames": frames,
                "vap_offset": self.vap_count,
                "asd_offset": self.asd_count,
                "label_offset": self.asd_count,
                "vap_dim": vap.shape[-1],
                "asd_dim": asd.shape[-1],
            }
        )
        self.vap.append(vap)
        self.asd.append(asd.reshape(-1, asd.shape[-1]))
        self.labels.append(packed_labels.reshape(-1))
        self.boxes.append(bbox.reshape(-1, 4))
        self.byte_count += sample_bytes
        self.vap_count += frames
        self.asd_count += speakers * frames

    def flush(self) -> None:
        if not self.entries:
            return
        shard_dir = self.output_dir / f"shard-{self.shard_index:06d}"
        shard_dir.mkdir(parents=False, exist_ok=False)
        np.save(shard_dir / "vap.npy", np.concatenate(self.vap), allow_pickle=False)
        np.save(shard_dir / "asd.npy", np.concatenate(self.asd), allow_pickle=False)
        np.save(
            shard_dir / "labels.npy", np.concatenate(self.labels), allow_pickle=False
        )
        np.save(shard_dir / "bbox.npy", np.concatenate(self.boxes), allow_pickle=False)
        with (shard_dir / "index.json").open("w") as handle:
            json.dump(self.entries, handle, separators=(",", ":"))
        self.samples_written += len(self.entries)
        self.shard_index += 1
        self._reset()

    def close(self) -> None:
        self.flush()
        with (self.output_dir / "dataset.json").open("w") as handle:
            json.dump(
                manifest(
                    self.kind,
                    EMBEDDING_SOURCE,
                    self.shard_index,
                    self.samples_written,
                    # Which weights produced this, so a stale cache is findable
                    # rather than merely wrong.
                    extracted_by=self.provenance,
                ),
                handle,
                indent=2,
            )
