"""Conversation pack schema: the multiparty counterpart of the ASD pack.

An ASD pack stores one face track beside the audio it is judged against. A
conversation pack stores a whole segment: *every* tracked face, the one mixed
recording they share, and one voice activity row per speaker. That grouping is
the thing the fusion model needs and the ASD pack cannot express, because
MuVAP's global stream is defined over the conversation rather than over any one
face in it.

Everything else is deliberately the same, and for the same reason
(`docs/preprocessing.md`): faces, labels, and boxes keep the native frame rate,
audio keeps its full duration and its original gain, and every model-specific
transform - the 25 Hz timeline, loudness normalization, chunking - happens in
`data.dataloaders.conversation` when a batch is built.

A pack is described by two independent axes, and one reader covers all four
combinations.

**Kind** - what a sample is:

* `segments` - one whole segment per sample, which the loader tiles into chunks
  at training time;
* `events` - one window anchored on each annotated turn event, what the reported
  turn-taking numbers are computed on. Its entries carry the event's label, its
  floor holder, and the speaker who takes over.

**Source** - what a sample holds:

* `media` - faces and the mixed recording, on the native timeline;
* `embeddings` - what the frozen VAP and ASD modules made of that media, on the
  25 Hz model timeline. Labels, boxes, and visibility travel with it, so an
  embedding pack stands on its own and is several times smaller than the media
  it came from.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

FORMAT_NAME = "muvap-conv-mmap"
FORMAT_VERSION = 1
SUPPORTED_VERSIONS = (1,)

#: Pack kinds. They differ only in how windows were chosen and in whether the
#: entries carry turn-event annotation, never in how the arrays are laid out.
SEGMENT_KIND = "segments"
EVENT_KIND = "events"
KINDS = (SEGMENT_KIND, EVENT_KIND)

#: Pack sources, and the timeline each one is necessarily written on. Media
#: keeps the corpus's frame rate because a pack is a decoded mirror of it;
#: embeddings cannot, because the encoder that produced them runs at one rate.
MEDIA_SOURCE = "media"
EMBEDDING_SOURCE = "embeddings"
SOURCES = (MEDIA_SOURCE, EMBEDDING_SOURCE)
TIMELINES = {MEDIA_SOURCE: "native", EMBEDDING_SOURCE: "model"}

# Repeated from the ASD schema rather than imported so the two formats can drift
# apart without one silently redefining the other. They agree today, which is
# what lets a conversation pack be cropped into ASD samples.
MODEL_FPS = 25.0
SAMPLE_RATE = 16_000
FACE_SIZE = 112


@dataclass(frozen=True)
class ConversationRecord:
    """One segment of one recording, with every tracked speaker in it.

    `faces`, `vad`, `bbox`, and `visible` are all indexed
    `[speaker, source_frame, ...]` and all describe the same native timeline,
    so a row of one lines up with the same row of the others. `speaker_ids` name
    those rows in the corpus's own terms, which is what the turn-event
    annotation refers to.
    """

    dataset: str
    sample_id: str
    video_id: str
    segment_id: str
    speaker_ids: Sequence[str]
    source_fps: float
    start_sec: float = 0.0
    #: Per speaker, whether the *recording* ever boxes that face. Distinct from
    #: the per-frame visibility beside it: a tracked speaker can be missing from
    #: any given window, while an untracked one has no face to find anywhere.
    #: The annotation uses bare labels like `B` for voices that are only ever
    #: heard, and they are not candidates for a question about who is on screen.
    #: Defaults to whatever the sample's own visibility shows.
    tracked: Sequence[bool] | None = None

    def __post_init__(self) -> None:
        if self.source_fps <= 0:
            raise ValueError(f"{self.sample_id}: source_fps must be positive")
        if self.start_sec < 0:
            raise ValueError(f"{self.sample_id}: start_sec cannot be negative")
        if not self.speaker_ids:
            raise ValueError(f"{self.sample_id}: a conversation needs a speaker")


@dataclass(frozen=True)
class EventAnnotation:
    """What one annotated turn event claims, carried through to evaluation.

    One row of the benchmark file, minus the two columns that locate it:
    `previous` held the floor going into the mutual silence and `following`
    takes it afterwards, so they are equal exactly when `label` is a hold.

    `pause` is how long that silence lasts. It is recorded because it is part of
    the annotation, and nothing reads it: an event window ends where the silence
    begins, so the pause is on the far side of what the model is shown.
    """

    label: str
    previous: str
    following: str
    pause: float = 0.0


def manifest(kind: str, source: str, shards: int, samples: int, **extra) -> dict:
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")
    if source not in SOURCES:
        raise ValueError(f"source must be one of {SOURCES}, got {source!r}")
    common = {
        "format": FORMAT_NAME,
        "version": FORMAT_VERSION,
        "kind": kind,
        "source": source,
        "timeline": TIMELINES[source],
        "label_dtype": "uint8",
        "bbox": "xyxy normalized to the frame, [0, 1]",
        "shards": shards,
        "samples": samples,
    }
    if source == MEDIA_SOURCE:
        common |= {
            "audio_sample_rate": SAMPLE_RATE,
            "audio_normalization": None,
            "face_shape": [FACE_SIZE, FACE_SIZE],
            "face_color": "grayscale",
            "face_dtype": "uint8",
            "audio_dtype": "int16",
        }
    else:
        common |= {"model_fps": MODEL_FPS, "feature_dtype": "float32"}
    return common | extra


def check_manifest(manifest_data: dict, root: Path, kind: str | None = None) -> None:
    """Reject a pack this reader would silently misinterpret."""
    if (
        manifest_data.get("format") != FORMAT_NAME
        or manifest_data.get("version") not in SUPPORTED_VERSIONS
    ):
        raise ValueError(f"unsupported conversation pack format: {root}")
    source = manifest_data.get("source")
    if source not in SOURCES:
        raise ValueError(f"{root} does not say what it holds: source={source!r}")
    if manifest_data.get("timeline") != TIMELINES[source]:
        raise ValueError(
            f"{root} is a {source} pack on the "
            f"{manifest_data.get('timeline')!r} timeline, which cannot be right"
        )
    if source == MEDIA_SOURCE:
        if manifest_data.get("audio_sample_rate") != SAMPLE_RATE:
            raise ValueError(f"unexpected audio sample rate: {root}")
        if manifest_data.get("audio_normalization") is not None:
            raise ValueError(f"pack has baked-in audio normalization: {root}")
    elif manifest_data.get("model_fps") != MODEL_FPS:
        raise ValueError(
            f"{root} holds features at {manifest_data.get('model_fps')} Hz, but this "
            f"model runs at {MODEL_FPS} Hz"
        )
    if kind is not None and manifest_data.get("kind") != kind:
        raise ValueError(
            f"{root} is a {manifest_data.get('kind')!r} pack, but a {kind!r} pack is "
            "needed here"
        )
