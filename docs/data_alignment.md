# Audio-visual data contract

## Canonical timeline

MuVAP+ uses a **25 Hz model timeline**, because the CPC stack emits one audio
embedding per 640 samples at 16 kHz. Original videos are not transcoded or
rewritten to 25 fps.

For model frame `k`, the timestamp is the centre of its interval:

```text
t(k) = (k + 0.5) / 25 seconds
```

The visual face and label are selected from the native source interval that
contains `t(k)`:

```text
source_index(k) = floor(t(k) * source_fps)
```

The exact same index is used for the face and its label. This prevents label
drift. At 30 fps some source frames are skipped; below 25 fps some are repeated.

**This happens in the loader, not in the pack.** Packed shards store the native
frame rate, so the sampling above is applied when a batch is built and can be
changed - or replaced with a different `model_fps` - without re-packing.

## Evaluation on the native timeline

Metrics defined on the source annotations — the official AVA mAP above all —
need one score per native frame, not per model frame. `model_indices` reverses
the mapping by asking which model interval contains each source-frame centre:

```text
model_index(j) = floor(((j + 0.5) / source_fps) * 25)
```

The result always has exactly `source_frames` entries, so every groundtruth row
receives a score at any frame rate. Below 25 fps this is an exact inverse of
`source_indices`. Above it, several source frames share a model frame, which is
information the 25 Hz timeline genuinely does not carry. Only the final source
frame can fall past the last complete model interval; it is clamped to that
interval, and no interior frame is ever mis-scored.

## Audio

Audio is loaded directly from WAV/FLAC (or NumPy waveforms), mixed to
mono, and resampled to 16 kHz for CPC. Packed shards keep the full span the face
frames cover, at the recording's own gain.

The loader takes it from there. A batch of `T` model frames always carries
exactly `T * 640` samples, right-padded with silence if the recording is short.
Loudness normalization to -25 dBFS RMS is applied at that point, using level
statistics measured over the **whole recording** and stored in the index -
computing gain from a chunk alone would make a sample's level depend on which
chunk it landed in.

## Required manifest fields

Each row must resolve to:

| Field | Meaning |
|---|---|
| `sample_id` | Stable, unique face-track/clip identifier |
| `audio_path` | Recording or clip audio path |
| `visual_path` | NumPy face array shaped `[source_frames, H, W]` |
| `source_fps` | Native FPS used by faces and labels |
| `labels` | One active-speaker label per native face frame |
| `start_sec` | Clip start on the audio recording timeline; default `0` only when audio is already clip-local |

The canonical file is a tab-separated text file with a header. `labels` is a
JSON array; paths may be absolute or relative to the manifest directory:

```text
sample_id\taudio_path\tvisual_path\tsource_fps\tlabels\tstart_sec
clip_001\taudio/recording.wav\tfaces/clip_001.npy\t29.97\t[0,0,1,1]\t12.48
```

`len(labels)` and the face-frame count must match. Preprocessing rejects a
sample where they disagree rather than trimming to the shorter one: a length
mismatch means the face track and the annotation disagree about the clip, which
would silently shift every later label.

## Dataset policy

- AVA, WASD, and MSDWild use the same alignment implementation.
- FPS is never ignored and must never be replaced with a global constant.
- Packs are decode-only. Any transform whose parameters belong to the model -
  frame rate, loudness, crop duration - happens in the loader, so a pack stays
  usable by code that does not share those conventions.
- Labels are aligned before projection-window targets are computed.
- Train-time batching may crop examples to a shared duration, but audio, faces,
  VAD labels, and projected labels must all use that same duration.
- A WASD face track using full-recording audio must provide `start_sec`. It is
  mathematically impossible to recover that offset from FPS and labels alone.
