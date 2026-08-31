"""Memory-mapped reader that turns native-rate packs into model-ready batches.

Shards hold decoded source media: faces and labels at the native frame rate,
unnormalized 16 kHz audio. Everything MuVAP+ specific happens here - sampling
onto the 25 Hz model timeline, loudness normalization, chunking, augmentation -
so the same shards can serve a model with different conventions.
"""

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, Sampler

from data.alignment import model_frame_count, source_indices
from data.augment import AugmentConfig, augment_visual, mix_audio
from data.media import WaveformStats, apply_normalization
from preprocess.asd.schema import FORMAT_NAME, MODEL_FPS, SAMPLE_RATE, SUPPORTED_VERSIONS


def _clip_id(sample_id: str) -> str:
    """Strip the entity suffix so same-recording samples share a key.

    AVA and WASD both name tracks `<clip>_<start>_<end>:<entity>`, where every
    entity of a clip may share one audio recording.
    """
    return sample_id.rsplit(":", 1)[0]


@dataclass(frozen=True)
class _Entry:
    root: Path
    shard: str
    metadata: dict


class PackedASDDataset(Dataset):
    """Random-access dataset backed by OS-page-cached NumPy memmaps."""

    def __init__(
        self,
        roots: str | Path | Sequence[str | Path],
        projection_window=None,
        window: int | None = None,
        context: int = 0,
        jitter: bool = False,
        augment: AugmentConfig | None = None,
        model_fps: float = MODEL_FPS,
        normalize: bool = True,
    ):
        """Read packed shards, optionally as fixed-length training chunks.

        `window` splits every track longer than it into near-equal chunks, so
        one epoch still covers every frame while batches stay uniform enough to
        pack tightly.

        `context` frames are read *before* each chunk and marked invalid in the
        returned mask. The model is causal, so a frame near the start of a chunk
        would otherwise be asked to predict from history the encoder was never
        shown. Feeding that history as unscored context costs a little compute
        and no storage, where overlapping the chunks would duplicate the data
        and still train on the truncated copies. A track's real first frames get
        no prefix: there the missing history is genuine, and the model has to
        cope with it at inference too.

        `jitter` slides the chunk grid within its track on every access, which
        varies the boundaries across epochs at no extra cost. All spans are
        counted in model frames, not source frames.
        """
        if isinstance(roots, (str, Path)):
            roots = [roots]
        self.projection_window = projection_window
        self.window = window
        self.context = max(0, context)
        self.jitter = jitter
        self.augment = augment
        self.model_fps = model_fps
        self.normalize = normalize
        self.samples_per_frame = round(SAMPLE_RATE / model_fps)
        self.entries: list[_Entry] = []
        self._arrays: dict[
            tuple[Path, str], tuple[np.ndarray, np.ndarray, np.ndarray]
        ] = {}

        for raw_root in roots:
            root = Path(raw_root)
            with (root / "dataset.json").open() as handle:
                manifest = json.load(handle)
            if (
                manifest.get("format") != FORMAT_NAME
                or manifest.get("version") not in SUPPORTED_VERSIONS
            ):
                raise ValueError(f"unsupported packed dataset format: {root}")
            if manifest.get("timeline") != "native":
                raise ValueError(f"pack is not on the native timeline: {root}")
            if manifest.get("audio_sample_rate") != SAMPLE_RATE:
                raise ValueError(f"unexpected audio sample rate: {root}")
            if manifest.get("audio_normalization") is not None:
                raise ValueError(f"pack has baked-in audio normalization: {root}")

            for shard_dir in sorted(root.glob("shard-*")):
                with (shard_dir / "index.json").open() as handle:
                    for metadata in json.load(handle):
                        self.entries.append(_Entry(root, shard_dir.name, metadata))

        self.model_frames = np.array(
            [
                model_frame_count(
                    entry.metadata["source_frames"],
                    entry.metadata["source_fps"],
                    model_fps,
                )
                for entry in self.entries
            ],
            dtype=np.int64,
        )
        self.chunks = self._build_chunks()
        self._clips = [_clip_id(entry.metadata["sample_id"]) for entry in self.entries]

    @property
    def predict_window(self) -> int | None:
        """Frames scored per chunk; the context prefix is read on top of this."""
        if self.window is None:
            return None
        return max(1, self.window - self.context)

    def _build_chunks(self) -> list[tuple[int, int, int]]:
        """Split tracks into near-equal *scored* spans.

        A chunk is identified by the span it predicts. Whatever context the
        loader prepends is read on top and never scored, so the spans still tile
        the track exactly and every frame is predicted once per epoch.
        """
        chunks: list[tuple[int, int, int]] = []
        span = self.predict_window
        for index in range(len(self.entries)):
            frames = int(self.model_frames[index])
            if frames == 0:
                continue
            if span is None or frames <= span:
                chunks.append((index, 0, frames))
                continue
            edges = np.linspace(
                0, frames, math.ceil(frames / span) + 1
            ).round().astype(int)
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

    def __len__(self) -> int:
        return len(self.chunks)

    def _open(self, entry: _Entry):
        key = (entry.root, entry.shard)
        arrays = self._arrays.get(key)
        if arrays is None:
            shard_dir = entry.root / entry.shard
            faces = np.load(shard_dir / "faces.npy", mmap_mode="r", allow_pickle=False)
            audio = np.load(shard_dir / "audio.npy", mmap_mode="r", allow_pickle=False)
            labels = np.load(shard_dir / "labels.npy", mmap_mode="r", allow_pickle=False)
            arrays = faces, audio, labels
            self._arrays[key] = arrays
        return arrays

    def _stats(self, entry: _Entry) -> WaveformStats:
        return WaveformStats(
            mean=entry.metadata["audio_mean"],
            rms=entry.metadata["audio_rms"],
            peak=entry.metadata["audio_peak"],
        )

    def _read_audio(self, entry: _Entry, start: int, length: int) -> torch.Tensor:
        """Read one model-frame span and restore the recording's own gain."""
        _, audio, _ = self._open(entry)
        offset = entry.metadata["audio_offset"] + start * self.samples_per_frame
        available = entry.metadata["audio_samples"] - start * self.samples_per_frame
        wanted = length * self.samples_per_frame
        block = np.array(audio[offset : offset + min(wanted, max(available, 0))], copy=True)
        waveform = torch.from_numpy(block).float().div_(32768.0).unsqueeze(0)
        if waveform.shape[-1] < wanted:
            waveform = F.pad(waveform, (0, wanted - waveform.shape[-1]))
        if self.normalize:
            waveform = apply_normalization(waveform, self._stats(entry))
        return waveform

    def _donor_audio(self, clip: str, frames: int, rng) -> np.ndarray:
        """Read audio from a different clip than `clip`.

        Drawing from the whole dataset rather than the mini-batch matters here:
        every face track in a WASD clip shares one clip-level recording, and the
        batch sampler groups by length, which puts those tracks together. A
        batch-local donor would often be the very same waveform.
        """
        for _ in range(8):
            index = rng.randrange(len(self.entries))
            if self._clips[index] == clip:
                continue
            entry = self.entries[index]
            span = int(self.model_frames[index])
            start = rng.randrange(max(span - frames, 0) + 1) if span > frames else 0
            return self._read_audio(entry, start, min(frames, max(span, 1)))[0].numpy()
        return np.zeros(frames * self.samples_per_frame, dtype=np.float32)

    def __getitem__(self, index: int):
        entry_index, predict_start, predict_length = self.chunks[index]
        entry = self.entries[entry_index]
        faces, _, labels = self._open(entry)
        frames = int(self.model_frames[entry_index])
        if self.jitter and predict_length < frames:
            predict_start = random.randint(0, frames - predict_length)

        # Read a history prefix where one exists; at a track's real start there
        # is none, and that truncation is genuine rather than an artefact.
        prefix = min(self.context, predict_start)
        start = predict_start - prefix
        length = prefix + predict_length
        scored = torch.zeros(length, dtype=torch.bool)
        scored[prefix:] = True

        # One shared mapping puts faces and labels on the model timeline, which
        # is what keeps a label attached to the face it describes.
        rows = source_indices(
            entry.metadata["source_frames"], entry.metadata["source_fps"], self.model_fps
        )
        span = rows[start : start + length]
        face_block = np.asarray(faces[entry.metadata["face_offset"] + span])
        # Labels are one byte a frame, so the whole track is cheap to read. The
        # projection window needs 50 frames of history and 15 of future; taking
        # them from the track rather than the chunk keeps a chunk boundary from
        # fabricating silence in the targets.
        track_labels = np.asarray(
            labels[entry.metadata["label_offset"] + rows], dtype=np.uint8
        ).astype(np.float32)
        waveform = self._read_audio(entry, start, length)

        if self.augment is not None:
            # The module RNG is what Lightning seeds per worker, so augmentation
            # is reproducible from the run seed.
            rng = random
            clip = self._clips[entry_index]
            audio_block = waveform[0].numpy()
            if rng.random() < self.augment.substitute_audio_prob:
                audio_block = self._donor_audio(clip, length, rng)
                track_labels = np.zeros_like(track_labels)
            elif rng.random() < self.augment.audio_mix_prob:
                audio_block = mix_audio(
                    audio_block,
                    self._donor_audio(clip, length, rng),
                    rng.uniform(*self.augment.audio_snr_db),
                )
            waveform = torch.from_numpy(np.ascontiguousarray(audio_block)).unsqueeze(0)
            if self.augment.visual:
                face_block = augment_visual(face_block, self.augment, rng)

        visual = torch.from_numpy(np.ascontiguousarray(face_block)).float()
        track_vad = torch.from_numpy(track_labels)
        vad = track_vad[start : start + length]
        if self.projection_window is None:
            return waveform, visual, vad, scored, entry.metadata
        with torch.no_grad():
            vap = self.projection_window.get_labels(track_vad.view(1, 1, frames))
        return (
            waveform,
            visual,
            vap[0, start : start + length],
            vad,
            scored,
            entry.metadata,
        )


class ASDCollator:
    """Bring a batch to one length, by cropping or by padding.

    Training crops to the shortest member. That keeps the model free of a
    padding mask, and it is affordable because chunking bounds the spread of
    lengths and the batch sampler groups similar lengths together; measured on
    AVA plus WASD the pair loses 0.01% of frames per epoch, against 64% for
    random batches.

    Evaluation cannot crop: the official export needs every frame of every
    track, which is why it would otherwise run one track at a time. `pad=True`
    right-pads to the longest member instead and marks the filler unscored.
    Every stage of the model is causal - the CPC and Mimi frontends, the visual
    TCN, and both transformer stacks - so appended frames cannot change any
    earlier output, and the visual frontend's batch norm is on running
    statistics in eval mode. A test pins both properties.
    """

    def __init__(
        self, samples_per_frame: int = SAMPLE_RATE // int(MODEL_FPS), pad: bool = False
    ):
        self.samples_per_frame = samples_per_frame
        self.pad = pad

    @staticmethod
    def _to_length(tensor: torch.Tensor, length: int, axis: int) -> torch.Tensor:
        """Crop or right-pad one stream along `axis` (counted from the end)."""
        current = tensor.shape[-axis]
        if current == length:
            return tensor
        if current > length:
            index = [slice(None)] * tensor.ndim
            index[-axis] = slice(0, length)
            return tensor[tuple(index)]
        padding = [0, 0] * (axis - 1) + [0, length - current]
        return F.pad(tensor, padding)

    def __call__(self, batch):
        lengths = [item[1].shape[0] for item in batch]
        frames = max(lengths) if self.pad else min(lengths)
        samples = frames * self.samples_per_frame

        audio = torch.stack([self._to_length(item[0], samples, 1) for item in batch])
        visual = torch.stack([self._to_length(item[1], frames, 3) for item in batch])
        vap = torch.stack([self._to_length(item[2], frames, 2) for item in batch])
        vad = torch.stack([self._to_length(item[3], frames, 1) for item in batch])
        scored = torch.stack([self._to_length(item[4], frames, 1) for item in batch])
        return audio, visual, vap, vad, scored, [item[5] for item in batch]


class FrameBudgetBatchSampler(Sampler):
    """Group similar-length chunks into batches of roughly constant frame count.

    Holding `batch * frames` near a budget keeps three things stable that a
    fixed batch size does not: activation memory, the `B*T` population the
    visual frontend's batch norm sees, and the per-frame gradient noise of the
    mean-reduced losses. The batch size is clamped because the contrastive term
    is a `B`-way classification whose difficulty would otherwise swing with the
    chunk length.
    """

    def __init__(
        self,
        lengths,
        frame_budget: int,
        min_batch: int = 4,
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
        self.frame_budget = frame_budget
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
        order = np.argsort(-self.lengths, kind="stable")
        batches = []
        cursor = 0
        while cursor < len(order):
            longest = int(self.lengths[order[cursor]])
            size = min(
                max(self.frame_budget // max(longest, 1), self.min_batch), self.max_batch
            )
            batch = order[cursor : cursor + size]
            cursor += size
            if self.drop_last and len(batch) < self.min_batch:
                continue
            batches.append([int(index) for index in batch])

        if self.shuffle:
            rng = random.Random(self.seed + self.epoch)
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
