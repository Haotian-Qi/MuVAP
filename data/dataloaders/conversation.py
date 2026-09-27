"""Memory-mapped reader for multiparty conversation packs.

The multiparty counterpart of `data.dataloaders.packed`, and it works the same
way: shards hold decoded source material, and everything MuVAP-specific -
sampling onto the 25 Hz model timeline, loudness normalization, chunking,
labelling - happens here, so one pack can serve a model whose conventions
differ from this one.

One dataset class reads two kinds of pack, and which one it is changes nothing
downstream of `__getitem__`:

* a **media** pack is raw material - faces and one mixed recording - and the
  frozen VAP and ASD modules are run on it by the task, batch by batch;
* an **embedding** pack is what those modules already produced, on the 25 Hz
  timeline, with the labels alongside so it stands on its own.

The second is a cache of the first, not a different experiment: the extraction
tool runs the same `FrozenEncoders` the raw path runs. Reach for the media pack
to change or fine-tune a frozen module, and the embedding pack otherwise - it
is several times smaller and skips both encoders on every step.
"""

import json
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, Sampler

from data.alignment import model_frame_count, source_indices
from data.media import WaveformStats, apply_normalization
from preprocess.muvap.schema import (
    EVENT_KIND,
    MEDIA_SOURCE,
    MODEL_FPS,
    SAMPLE_RATE,
    SEGMENT_KIND,
    check_manifest,
)

#: How the two label bits are packed by the writer.
VAD_BIT = 1
VISIBLE_BIT = 2

#: What a frame with no face holds, and therefore what padding a short track
#: should look like to the visual frontend.
BLANK_FACE = 128


@dataclass(frozen=True)
class _Entry:
    root: Path
    shard: str
    metadata: dict
    native: bool


class PackedConversationDataset(Dataset):
    """Random-access dataset over conversation packs, raw or precomputed."""

    def __init__(
        self,
        roots: str | Path | Sequence[str | Path],
        gvap_projection=None,
        svap_projection=None,
        kind: str = SEGMENT_KIND,
        window: int | None = None,
        overlap: int = 0,
        context: int = 0,
        model_fps: float = MODEL_FPS,
        normalize: bool = True,
    ):
        """Read packed shards, optionally as fixed-length training chunks.

        `window` splits a segment into spans of that length. `overlap` is how
        much of the previous span each one repeats: the paper's setup is 30 s
        windows overlapping by 10 s, which is `window=750, overlap=250` at
        25 Hz. With no overlap the spans instead tile the segment near-equally,
        so one epoch still covers every frame exactly once.

        `context` frames are read in front of each span and marked unscored,
        which is the alternative to overlapping: the model gets the same history
        without the overlapped frames contributing to the loss twice.

        The grid is fixed. A randomised start would decorrelate a frame from the
        amount of history it is seen with, which is worth having when windows
        tile without overlap - but it also turns an epoch from a pass over the
        segment into sampling with replacement, and an overlapping grid already
        shows every frame at two different offsets for free.

        An event pack ignores all of it. Its windows are anchored on the
        annotation, and moving one would change the question being asked.
        """
        if isinstance(roots, (str, Path)):
            roots = [roots]
        self.kind = kind
        self.gvap_projection = gvap_projection
        self.svap_projection = svap_projection
        self.window = None if kind == EVENT_KIND else window
        self.overlap = 0 if kind == EVENT_KIND else max(0, overlap)
        self.context = 0 if kind == EVENT_KIND else max(0, context)
        self.model_fps = model_fps
        self.normalize = normalize
        self.samples_per_frame = round(SAMPLE_RATE / model_fps)
        self.entries: list[_Entry] = []
        self._arrays: dict[tuple[Path, str], dict[str, np.ndarray]] = {}

        sources = set()
        for raw_root in roots:
            root = Path(raw_root)
            with (root / "dataset.json").open() as handle:
                pack = json.load(handle)
            check_manifest(pack, root, kind)
            sources.add(pack["source"])
            native = pack["timeline"] == "native"
            for shard_dir in sorted(root.glob("shard-*")):
                with (shard_dir / "index.json").open() as handle:
                    for metadata in json.load(handle):
                        self.entries.append(
                            _Entry(root, shard_dir.name, metadata, native)
                        )
        if len(sources) > 1:
            raise ValueError(
                f"cannot mix {sorted(sources)} packs in one dataset: one is raw media "
                "and the other is what a frozen encoder made of it"
            )
        self.source = sources.pop() if sources else MEDIA_SOURCE

        self.model_frames = np.array(
            [self._frames(entry) for entry in self.entries], dtype=np.int64
        )
        self.speakers = np.array(
            [entry.metadata["speakers"] for entry in self.entries], dtype=np.int64
        )
        self.chunks = self._build_chunks()

    # ------------------------------------------------------------------ layout

    def _frames(self, entry: _Entry) -> int:
        if not entry.native:
            return int(entry.metadata["model_frames"])
        return model_frame_count(
            entry.metadata["source_frames"],
            entry.metadata["source_fps"],
            self.model_fps,
        )

    @property
    def predict_window(self) -> int | None:
        """Frames scored per chunk; the context prefix is read on top of this."""
        if self.window is None:
            return None
        return max(1, self.window - self.context)

    def _build_chunks(self) -> list[tuple[int, int, int]]:
        chunks: list[tuple[int, int, int]] = []
        span = self.predict_window
        for index in range(len(self.entries)):
            frames = int(self.model_frames[index])
            if frames == 0:
                continue
            if span is None or frames <= span:
                chunks.append((index, 0, frames))
                continue
            if self.overlap:
                # A fixed grid stepping by span - overlap, and a short final
                # window for whatever is left. Overlapped frames are seen twice
                # in an epoch, which is what an overlapping grid means.
                step = max(1, span - self.overlap)
                start = covered = 0
                while start + span <= frames:
                    chunks.append((index, start, span))
                    covered = start + span
                    start += step
                # A tail only earns its place if it reaches frames no full
                # window did; where the grid already lands on the end of the
                # segment, one would be a duplicate of the window before it.
                if covered < frames:
                    chunks.append((index, start, frames - start))
                continue
            edges = (
                np.linspace(0, frames, math.ceil(frames / span) + 1)
                .round()
                .astype(int)
            )
            for start, end in zip(edges[:-1], edges[1:]):
                chunks.append((index, int(start), int(end - start)))
        return chunks

    def _fed_length(self, start: int, length: int) -> int:
        return length + min(self.context, start)

    def chunk_lengths(self) -> np.ndarray:
        """Frames actually fed per chunk - what bounds memory and batch size."""
        return np.fromiter(
            (self._fed_length(start, length) for _, start, length in self.chunks),
            dtype=np.int64,
            count=len(self.chunks),
        )

    def chunk_speakers(self) -> np.ndarray:
        """Speakers per chunk. A batch has to agree on this, so a sampler groups on it."""
        return self.speakers[[index for index, _, _ in self.chunks]]

    def __len__(self) -> int:
        return len(self.chunks)

    # ------------------------------------------------------------------- reads

    def _open(self, entry: _Entry) -> dict[str, np.ndarray]:
        key = (entry.root, entry.shard)
        arrays = self._arrays.get(key)
        if arrays is None:
            shard_dir = entry.root / entry.shard
            names = (
                ("faces", "audio", "labels", "bbox")
                if entry.native
                else ("vap", "asd", "labels", "bbox")
            )
            arrays = {
                name: np.load(
                    shard_dir / f"{name}.npy", mmap_mode="r", allow_pickle=False
                )
                for name in names
            }
            self._arrays[key] = arrays
        return arrays

    def _rows(self, entry: _Entry, frames: int) -> np.ndarray:
        """Per-speaker row indices onto the model timeline.

        A media pack stores the corpus's own frame rate and is sampled here; an
        embedding pack was written on the model timeline by an encoder that runs
        at that rate, so its rows are already one per model frame.
        """
        if not entry.native:
            return np.arange(frames, dtype=np.int64)
        return source_indices(
            entry.metadata["source_frames"],
            entry.metadata["source_fps"],
            self.model_fps,
        )

    def _stats(self, entry: _Entry) -> WaveformStats:
        return WaveformStats(
            mean=entry.metadata["audio_mean"],
            rms=entry.metadata["audio_rms"],
            peak=entry.metadata["audio_peak"],
        )

    def _read_audio(self, entry: _Entry, start: int, length: int) -> torch.Tensor:
        """Read one model-frame span and restore the recording's own gain."""
        audio = self._open(entry)["audio"]
        offset = entry.metadata["audio_offset"] + start * self.samples_per_frame
        available = entry.metadata["audio_samples"] - start * self.samples_per_frame
        wanted = length * self.samples_per_frame
        block = np.array(
            audio[offset : offset + min(wanted, max(available, 0))], copy=True
        )
        waveform = torch.from_numpy(block).float().div_(32768.0).unsqueeze(0)
        if waveform.shape[-1] < wanted:
            waveform = F.pad(waveform, (0, wanted - waveform.shape[-1]))
        if self.normalize:
            waveform = apply_normalization(waveform, self._stats(entry))
        return waveform

    def _speaker_rows(self, entry: _Entry, rows: np.ndarray, span: slice) -> np.ndarray:
        """Flat indices of one span, for every speaker, into a speaker-major array.

        The writer stores all of speaker 0's frames, then all of speaker 1's, so
        a speaker's stride is the segment's own length rather than the speaker
        count.
        """
        stride = (
            entry.metadata["source_frames"]
            if entry.native
            else entry.metadata["model_frames"]
        )
        offsets = np.arange(entry.metadata["speakers"], dtype=np.int64) * stride
        return offsets[:, None] + rows[span][None, :]

    # ------------------------------------------------------------------- items

    def __getitem__(self, index: int):
        entry_index, predict_start, predict_length = self.chunks[index]
        entry = self.entries[entry_index]
        arrays = self._open(entry)
        frames = int(self.model_frames[entry_index])
        speakers = entry.metadata["speakers"]

        # Read a history prefix where one exists; at a segment's real start
        # there is none, and that truncation is genuine rather than an artefact.
        prefix = min(self.context, predict_start)
        start = predict_start - prefix
        length = prefix + predict_length
        scored = torch.zeros(length, dtype=torch.bool)
        scored[prefix:] = True

        rows = self._rows(entry, frames)
        span = slice(start, start + length)
        flat = self._speaker_rows(entry, rows, span)

        label_base = entry.metadata["label_offset"]
        packed = np.asarray(arrays["labels"][label_base + flat], dtype=np.uint8)
        vad = torch.from_numpy((packed & VAD_BIT).astype(np.float32))
        visible = torch.from_numpy(((packed & VISIBLE_BIT) > 0))
        bbox = torch.from_numpy(
            np.asarray(arrays["bbox"][label_base + flat], dtype=np.float32)
        )

        item = {
            "vad": vad,
            "visual_mask": visible,
            "bbox": bbox,
            "mask": scored,
            "speakers": speakers,
            "metadata": entry.metadata,
        }
        item.update(self._features(entry, arrays, rows, span, flat, start, length))
        item.update(self._labels(entry, arrays, rows, frames, start, length))
        return item

    def _features(self, entry, arrays, rows, span, flat, start, length):
        if entry.native:
            faces = np.asarray(arrays["faces"][entry.metadata["face_offset"] + flat])
            # Kept as bytes. The visual frontend divides by 255 on its way in, so
            # widening here would only quadruple what every worker copies and
            # what crosses the bus - on the raw path that is the difference
            # between feeding the GPU and starving it.
            return {
                "audio": self._read_audio(entry, start, length),
                "visual": torch.from_numpy(np.ascontiguousarray(faces)),
            }
        vap = np.asarray(
            arrays["vap"][entry.metadata["vap_offset"] + rows[span]], dtype=np.float32
        )
        asd = np.asarray(
            arrays["asd"][entry.metadata["asd_offset"] + flat], dtype=np.float32
        )
        return {
            "vap_emb": torch.from_numpy(vap).unsqueeze(0),
            "asd_emb": torch.from_numpy(asd),
        }

    def _labels(self, entry, arrays, rows, frames, start, length):
        """Project the *whole* segment's activity, then take this chunk's slice.

        Labels are one byte a frame, so reading the segment is cheap, and taking
        the window from the segment rather than from the chunk keeps a chunk
        boundary from fabricating silence in the targets.

        An event pack gets none of this. Its ground truth is the annotation the
        event carries, and its window ends inside the silence being predicted,
        so a projection over it would be reading a future that was deliberately
        cut off.
        """
        if self.kind == EVENT_KIND:
            return {"event": entry.metadata["event"]}

        segment = self._speaker_rows(entry, rows, slice(0, frames))
        packed = np.asarray(
            arrays["labels"][entry.metadata["label_offset"] + segment], dtype=np.uint8
        )
        activity = torch.from_numpy((packed & VAD_BIT).astype(np.float32))

        labels = {}
        window = slice(start, start + length)
        with torch.no_grad():
            if self.gvap_projection is not None:
                labels["gvap_gt"] = self.gvap_projection.get_labels(
                    activity.unsqueeze(0)
                )[0, window]
            if self.svap_projection is not None:
                # `independent` scores one speaker at a time, so the speaker
                # axis becomes the batch axis rather than a second row.
                labels["svap_gt"] = self.svap_projection.get_labels(
                    activity.unsqueeze(1)
                )[:, window]
        return labels


class ConversationCollator:
    """Bring a batch to one length.

    `target` pins that length, which is what training uses: every batch comes
    out `[batch, speakers, train_window, ...]` whatever the windows in it were.
    The alternative, cropping to the shortest member the way `ASDCollator`
    does, throws away the difference - a real batch here was
    `[505, 504, 430, 403, 353, 353]`, cropped to 353 - and leaves the shape
    changing from step to step, which keeps the allocator churning and the
    card's throughput uneven.

    Padding costs nothing in correctness. The filler is marked unscored in
    `mask`, so neither loss sees it, and every stage of both frozen encoders is
    causal, so frames appended at the end cannot change any earlier output.

    Without `target`, `pad` chooses between padding to the longest member and
    cropping to the shortest.

    The speaker axis is never padded. A batch whose members disagree on it is a
    sampler bug, not something to paper over: a fabricated speaker would be
    invisible for every frame, and the frame encoder would have to open its mask
    to a face that does not exist.
    """

    def __init__(
        self,
        samples_per_frame: int = SAMPLE_RATE // int(MODEL_FPS),
        pad: bool = False,
        target: int | None = None,
    ):
        self.samples_per_frame = samples_per_frame
        self.pad = pad
        self.target = target

    @staticmethod
    def _to_length(tensor: torch.Tensor, length: int, axis: int, value: float = 0.0):
        """Crop or right-pad one stream along `axis` (counted from the end)."""
        current = tensor.shape[-axis]
        if current == length:
            return tensor
        if current > length:
            index = [slice(None)] * tensor.ndim
            index[-axis] = slice(0, length)
            return tensor[tuple(index)]
        padding = [0, 0] * (axis - 1) + [0, length - current]
        return F.pad(tensor, padding, value=value)

    #: Every batched stream, and which axis from the end holds time.
    TIME_AXIS = {
        "vap_emb": 2,
        "asd_emb": 2,
        "vad": 1,
        "visual_mask": 1,
        "bbox": 2,
        "mask": 1,
        "gvap_gt": 1,
        "svap_gt": 2,
    }

    def __call__(self, batch):
        speakers = {item["speakers"] for item in batch}
        if len(speakers) > 1:
            raise ValueError(
                f"a batch must agree on its speaker count, got {sorted(speakers)}; "
                "use SpeakerFrameBudgetSampler to group conversations by size"
            )

        lengths = [item["mask"].shape[0] for item in batch]
        frames = self.target or (max(lengths) if self.pad else min(lengths))

        collated = {
            key: torch.stack([self._to_length(item[key], frames, axis) for item in batch])
            for key, axis in self.TIME_AXIS.items()
            if key in batch[0]
        }
        if "visual" in batch[0]:
            # Mid-grey, the value a missing face already carries, so padding a
            # short track looks to the frontend like the blanks it knows.
            collated["visual"] = torch.stack(
                [self._to_length(item["visual"], frames, 3, BLANK_FACE) for item in batch]
            )
        if "audio" in batch[0]:
            samples = frames * self.samples_per_frame
            collated["audio"] = torch.stack(
                [self._to_length(item["audio"], samples, 1) for item in batch]
            )
        collated["speakers"] = speakers.pop()
        collated["metadata"] = [item["metadata"] for item in batch]
        if "event" in batch[0]:
            collated["event"] = [item["event"] for item in batch]
        return collated


class SpeakerFrameBudgetSampler(Sampler):
    """Group chunks by speaker count first, then to a roughly constant frame budget.

    A batch *must* agree on its speaker count, because the speaker axis is a
    real tensor dimension and no padding of it would be honest. Within that,
    `batch_size` fixes the size where a run needs one - which is what the paper
    reports - and leaving it unset falls back to holding `batch * frames` near
    `frame_budget` instead, which keeps activation memory and the per-frame
    gradient noise of the mean-reduced losses stable in the way a fixed size
    does not. That is the same reasoning as `FrameBudgetBatchSampler`, which
    this reduces to when every conversation has the same number of speakers.
    """

    def __init__(
        self,
        lengths,
        speakers,
        frame_budget: int,
        batch_size: int | None = None,
        min_batch: int = 1,
        max_batch: int = 16,
        shuffle: bool = True,
        drop_last: bool = False,
        seed: int = 0,
        num_replicas: int = 1,
        rank: int = 0,
    ):
        if min_batch < 1 or max_batch < min_batch:
            raise ValueError("min_batch must be positive and not exceed max_batch")
        self.lengths = np.asarray(lengths, dtype=np.int64)
        self.speakers = np.asarray(speakers, dtype=np.int64)
        if len(self.lengths) != len(self.speakers):
            raise ValueError("lengths and speakers must describe the same chunks")
        self.frame_budget = frame_budget
        self.batch_size = batch_size
        self.min_batch = min_batch
        self.max_batch = max_batch
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self._batches = self._build()

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch
        self._batches = self._build()

    def _build(self) -> list[list[int]]:
        buckets = defaultdict(list)
        for index, count in enumerate(self.speakers):
            buckets[int(count)].append(index)

        # Seeded by epoch alone, so every rank draws the same batches and then
        # takes its own slice of them.
        rng = random.Random(self.seed + self.epoch)
        batches: list[list[int]] = []
        for indices in buckets.values():
            # Chunks are indexed in segment order, so neighbouring indices are
            # overlapping windows of one conversation. Shuffling before the
            # stable length sort keeps equal-length chunks - nearly every full
            # training window - from always batching with the same neighbours;
            # the sort still groups short tails for the frame budget.
            if self.shuffle:
                rng.shuffle(indices)
            order = sorted(indices, key=lambda index: -self.lengths[index])
            cursor = 0
            while cursor < len(order):
                longest = int(self.lengths[order[cursor]])
                size = self.batch_size or min(
                    max(self.frame_budget // max(longest, 1), self.min_batch),
                    self.max_batch,
                )
                batch = order[cursor : cursor + size]
                cursor += size
                if self.drop_last and len(batch) < self.min_batch:
                    continue
                batches.append(batch)

        if self.shuffle:
            rng.shuffle(batches)
        if self.num_replicas > 1:
            # Every rank must run the same number of steps or DDP deadlocks.
            remainder = len(batches) % self.num_replicas
            if remainder:
                batches += batches[: self.num_replicas - remainder]
            batches = batches[self.rank :: self.num_replicas]
        return batches

    def __iter__(self):
        return iter(self._batches)

    def __len__(self) -> int:
        return len(self._batches)
