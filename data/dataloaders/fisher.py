"""Fisher datasets for VAP training and turn-taking evaluation.

Segments are stored as stereo audio beside the raw two-speaker VAD, not as
pre-encoded class indices. Labels are projected in the dataloader, so changing
`projection_window.mode` between the role-based and original-VAP setups is a
config edit rather than a re-run of preprocessing.

Audio comes from one of two sources, selected by `vap.source`. `npy` reads the
segments written under `seg/` and `tune/`; `raw` seeks the same span in the
source recording and resamples it, which costs an order of magnitude less
storage and leaves segment length, stride, and sample rate free to change
without re-running preprocessing. The two yield the same window.
"""

from functools import lru_cache
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as AF
from torch.utils.data import Dataset

from data.media import AudioSpec, load_waveform
from preprocess.vap.utils import EVENT_SEG_LEN_SEC, event_crop

SOURCES = ("npy", "raw")
TARGET_SAMPLE_RATE = 16_000
# Resampling a slice is not the same as slicing a resampled recording: the
# filter needs signal beyond both edges. Read this much extra and trim it off.
RESAMPLE_PAD_SEC = 0.1


@lru_cache(maxsize=1024)
def _source_rate(path):
    return sf.info(path).samplerate


def read_source_window(path, start_sec, duration_sec, channels):
    """Read `duration_sec` from a source recording, resampled to 16 kHz.

    Fisher ships as 8 kHz mu-law NIST Sphere, which `soundfile` reads.
    """
    path = str(path)
    rate = _source_rate(path)
    pad = int(round(RESAMPLE_PAD_SEC * rate))
    start = int(round(start_sec * rate))
    frames = int(round(duration_sec * rate))

    lead = min(pad, max(0, start))
    with sf.SoundFile(path) as handle:
        handle.seek(start - lead)
        block = handle.read(frames + lead + pad, dtype="float32", always_2d=True).T

    waveform = torch.from_numpy(np.ascontiguousarray(block))
    if channels == 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    elif waveform.shape[0] != channels:
        raise ValueError(
            f"{path} holds {waveform.shape[0]} channel(s); {channels} requested"
        )

    waveform = AF.resample(waveform, rate, TARGET_SAMPLE_RATE)
    scale = TARGET_SAMPLE_RATE / rate
    offset = int(round(lead * scale))
    target = int(round(duration_sec * TARGET_SAMPLE_RATE))
    waveform = waveform[:, offset : offset + target]
    if waveform.shape[1] < target:  # the recording ended inside the window
        waveform = torch.nn.functional.pad(waveform, (0, target - waveform.shape[1]))
    # Clip after resampling, exactly where the `npy` path clips, so a checkpoint
    # trained on one source evaluates identically on the other. The resampling
    # filter overshoots at transients; mu-law input never does.
    return waveform.clamp(-1.0, 1.0).contiguous()


def _check_source(source):
    if source not in SOURCES:
        raise ValueError(f"source must be one of {SOURCES}, got {source!r}")
    return source


def _audio_spec(channels):
    if channels not in (1, 2):
        raise ValueError(f"channels must be 1 or 2, got {channels}")
    return AudioSpec(sample_rate=16_000, mono=channels == 1)


class Fisher(Dataset):
    """Fixed-length conversation segments with labels projected on the fly."""

    def __init__(
        self,
        fisher_path,
        split_path,
        projection,
        channels=1,
        frame_hz=25,
        sample_rate=16_000,
        swap_channels=False,
        source="npy",
    ):
        self.root = Path(fisher_path)
        self.projection = projection
        self.channels = channels
        self.source = _check_source(source)
        self.spec = _audio_spec(channels)
        self.samples_per_frame = sample_rate // frame_hz
        self.swap_channels = swap_channels and channels == 2
        if projection.frame_hz != frame_hz:
            raise ValueError(
                f"projection is defined at {projection.frame_hz} Hz but the dataset runs at {frame_hz} Hz"
            )
        self.samples = []
        self._build(split_path)

    def _paths(self, line):
        # Resolved without touching the filesystem: a split lists hundreds of
        # thousands of segments, and stat-ing each one costs minutes of startup.
        part, group, conversation, segment, start, end, *_ = line.split()
        seg = self.root / part / "seg"
        if self.source == "npy":
            audio = seg / "audio" / group / conversation / f"{segment}.npy"
        else:
            audio = self.root / part / "audio" / group / f"{conversation}.wav"
        vad = seg / "vad" / group / conversation / f"{segment}.npy"
        return audio, vad, float(start), float(end) - float(start)

    def _build(self, split_path):
        with open(split_path) as file:
            self.samples = [self._paths(line) for line in file if line.strip()]
        if not self.samples:
            raise ValueError(f"{split_path} lists no segments")
        audio_path, vad_path, _, _ = self.samples[0]
        for path in (audio_path, vad_path):
            if not path.exists():
                raise FileNotFoundError(
                    f"missing preprocessed Fisher sample {path}; "
                    "run preprocess/vap/00_prep_fisher.py first"
                )

    def __len__(self):
        return len(self.samples)

    def _labels(self, vad, num_frames):
        """Project stored VAD, then drop the context kept only for the bins."""
        context = (vad.shape[-1] - num_frames) // 2
        if context < 0:
            raise ValueError(
                f"VAD has {vad.shape[-1]} frames, fewer than the {num_frames} audio frames"
            )
        needed = max(self.projection.hist_frames, self.projection.fut_frames)
        if context < needed:
            raise ValueError(
                f"projection needs {needed} context frames on each side but the segment stores {context}; "
                "re-run preprocess/vap/00_prep_fisher.py with a wider --context-sec"
            )
        labels = self.projection.get_labels(vad.unsqueeze(0))[0]
        return labels[context : context + num_frames]

    def __getitem__(self, index):
        audio_path, vad_path, start_sec, duration_sec = self.samples[index]
        if self.source == "raw":
            audio = read_source_window(
                audio_path, start_sec, duration_sec, self.channels
            )
        else:
            audio, _ = load_waveform(audio_path, self.spec, npy_sample_rate=16_000)
        vad = torch.from_numpy(np.load(vad_path, allow_pickle=False)).float()

        if self.channels == 2 and audio.shape[0] != 2:
            raise ValueError(
                f"{audio_path} holds {audio.shape[0]} channel(s); stereo VAP needs 2"
            )
        if self.swap_channels and torch.rand(()) < 0.5:
            # Which speaker sits on which channel is arbitrary, so swapping both
            # the audio and its VAD together is a label-preserving augmentation.
            audio = audio.flip(0)
            vad = vad.flip(0)

        num_frames = audio.shape[-1] // self.samples_per_frame
        return audio, self._labels(vad, num_frames)


class FisherEvent(Dataset):
    """Ten-second clips ending at an annotated hold or shift decision point."""

    FIELDS = (
        "part", "group", "conversation", "start", "end",
        "prev_spk", "next_spk", "timing", "gap_type", "label",
    )

    def __init__(self, fisher_path, split_paths, channels=1, source="npy"):
        self.root = Path(fisher_path)
        self.channels = channels
        self.source = _check_source(source)
        self.spec = _audio_spec(channels)
        self.clip_dir = "audio" if channels == 1 else "audio_stereo"
        if isinstance(split_paths, (str, Path)):
            split_paths = [split_paths]

        self.samples = []
        for split_path in split_paths:
            with open(split_path) as file:
                for line in file:
                    if not line.strip():
                        continue
                    sample = self._sample(dict(zip(self.FIELDS, line.split())))
                    if sample is not None:
                        self.samples.append(sample)

    def _sample(self, event):
        """One sample for this event, or None if it has no complete window."""
        crop_start = 0.0
        if self.source == "raw":
            crop = event_crop(event["timing"], float(event["start"]), float(event["end"]))
            if crop is None:
                return None
            crop_start = crop[0]
            audio = self.root / event["part"] / "audio" / event["group"]
            audio /= f"{event['conversation']}.wav"
        else:
            audio = self.root / event["part"] / "tune" / self.clip_dir / event["group"]
            audio /= f"{event['conversation']}/{event['start']}.npy"

        label = 1.0 if event["label"] == "SHIFT" else 0.0
        return (
            audio, label, event["conversation"], event["timing"],
            event["gap_type"], int(event["prev_spk"]), crop_start,
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        audio_path, label, conversation, timing, gap_type, prev_spk, crop_start = (
            self.samples[index]
        )
        if self.source == "raw":
            audio = read_source_window(
                audio_path, crop_start, EVENT_SEG_LEN_SEC, self.channels
            )
        else:
            audio, _ = load_waveform(audio_path, self.spec, npy_sample_rate=16_000)
            if self.channels == 2 and audio.shape[0] != 2:
                raise ValueError(
                    f"{audio_path} holds {audio.shape[0]} channel(s); stereo evaluation "
                    "needs 2. Re-run preprocess/vap/02_prep_event.py with --channels 2, "
                    "or set vap.source: raw to read the source recording instead"
                )
        return audio, label, conversation, timing, gap_type, prev_spk
