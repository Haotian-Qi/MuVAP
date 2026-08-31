# Canonical ASD preprocessing

All datasets are converted to the same versioned `muvap-asd-mmap` format.
Dataset adapters only resolve source metadata; `preprocess.asd.writer` performs
all media conversion and alignment.

## Packed format

A pack is a **decoded mirror of the source corpus, not a model-ready tensor
set**. Faces and labels keep the native frame rate, audio keeps its full
duration and its original gain. Everything MuVAP+ specific - sampling onto the
25 Hz model timeline, loudness normalization, chunking - happens in
`data.dataloaders.packed` when a batch is built. That split is what lets a pack
serve a model whose conventions differ from this one.

Each corpus keeps its pack beside its own media, so `--output` defaults to
`<dataset root>/packed/<split>` and a dataset stays self-contained:

```text
AVA/packed/train/
  dataset.json
  shard-000000/
    faces.npy   # uint8 [native_frames, 112, 112], grayscale
    audio.npy   # int16 mono 16 kHz, unnormalized, full clip duration
    labels.npy  # uint8, one label per NATIVE frame
    index.json  # offsets, source_fps, source_frames, audio level statistics
```

Each shard is approximately 1 GB. The arrays are uncompressed and opened with
`mmap_mode="r"`, so workers and DDP ranks share the operating-system page cache
without HDF5 handles or a per-process copy. Copy complete dataset directories
to node-local storage before training if the shared filesystem is the
bottleneck.

The per-sample invariant is:

```text
source_frames faces == source_frames labels
audio_samples == round(source_frames / source_fps * 16000)
```

### What the reader adds

`index.json` carries `source_fps`, `source_frames`, and the recording's
`audio_mean`, `audio_rms`, and `audio_peak` measured after DC removal. Those
statistics are what let the loader reproduce a whole-recording normalization
from any slice: gain computed from a chunk alone would drift with that chunk's
content. The loader removes DC, scales to -25 dBFS RMS, caps gain at 20 dB and
the peak at 0.99, and samples faces and labels onto the 25 Hz timeline with one
shared index so a label stays attached to the face it describes.

The one convention still baked into the pack is the face crop: 112x112
grayscale, recorded in `dataset.json` as `face_shape` and `face_color`. It is
the ASD community's standard geometry rather than a MuVAP+ choice, but a reader
should check it rather than assume it.

## AVA

```bash
python -m preprocess.asd.prepare ava \
  --loader /data/AVA/csv/train_loader.csv \
  --visual-root /data/AVA/clips_videos \
  --audio-root /data/AVA/clips_audios \
  --split train          # writes /data/AVA/packed/train
```

The adapter reads nested JPEG face directories and clip-local WAV files.
Missing inputs fail preprocessing. `--skip-missing` exists only for deliberately
partial development copies.

## WASD

WASD ships the same layout and the same loader schema as AVA, so it uses the
same reader:

```bash
python -m preprocess.asd.prepare wasd \
  --loader /data/WASD/csv/train_loader.csv \
  --visual-root /data/WASD/clips_videos \
  --audio-root /data/WASD/clips_audios \
  --split train          # writes /data/WASD/packed/train
```

Two differences from AVA are worth knowing when training on both. WASD labels
are strictly binary - `SPEAKING_NOT_AUDIBLE` never occurs - so the rule that
inaudible speech counts as negative is something only AVA teaches. And every
face track in a WASD clip shares one clip-level WAV, so audio alone cannot tell
its speakers apart; any augmentation that mixes in a second voice must draw it
from a different clip.

## MSDWild

Create 30-second native-FPS clips with 10 seconds of overlap directly from the
paired raw MP4 and bbox CSV files. The first window starts at the first frame
containing any face. Tracks must be present in at least 90% of a complete
window, which removes transient detections before face cropping.
Incomplete final windows are discarded rather than padded, because the model
does not use a padding mask and would otherwise learn from artificial silence.
Speaking labels come from `all.rttm`, which ships separately from the
bounding-box release and must be downloaded before this script can run; the
final bbox CSV column is not an activity label. `prepare.py` has no MSDWild
path - there is no loader TSV to read.

```bash
python preprocess/asd/prep_msdwild.py \
  --input /data/MSDWild/msdwild_boundingbox_labels \
  --output /data/MSDWild/preprocessed
```

The script uses the shared audio loader and face cropper and writes the same
canonical memory-mapped shards as AVA and WASD. No intermediate per-clip files
or loader CSV are created.

## AVCC

AVCC uses the canonical headered TSV because conversation corpora vary in
their crop-generation layout:

```text
sample_id\tvisual_path\taudio_path\tsource_fps\tlabels\tstart_sec\tvisual_kind\taudio_sample_rate
conv1_spk1\tfaces/conv1_spk1.npy\taudio/conv1.wav\t25.0\t[0,1,1,0]\t12.0\tnpy\t16000
```

Paths may be absolute or relative to `--root`:

```bash
python -m preprocess.asd.prepare avcc \
  --loader /data/AVCC/asd_manifest.tsv \
  --root /data/AVCC \
  --output /data/packed/avcc_train
```

Each row represents one speaker face track. AVCC conversation grouping,
bounding boxes, and speaker identities remain separate MuVAP metadata; its ASD
embeddings must be extracted from these canonical samples.

## Lightning

```yaml
packed_asd:
  train:
    - /scratch/local/muvap/AVA/packed/train
    - /scratch/local/muvap/WASD/packed/train
    - /scratch/local/muvap/WASD/packed/val
  val:
    - /scratch/local/muvap/AVA/packed/val
```

Use `PackedASDDataModule`. Start with four workers per GPU,
`persistent_workers=True`, `prefetch_factor=2`, and pinned memory, then measure
0/2/4/8 workers on the actual allocation.

### Training batches

Track lengths are wildly uneven - AVA's median is 64 frames against WASD's 746 -
and the collator crops a batch to its shortest member. Batching at random
therefore threw away most of the data: measured on the AVA pack, a shuffled
`batch_size: 8` used **36%** of its frames per epoch.

Two settings fix that, and they only work together:

- `train_window` splits any track longer than it into near-equal chunks, so one
  epoch still covers every frame and no chunk is much longer than another.
- `frame_budget` builds each batch from similar-length chunks up to a target
  `batch * frames`, clamped by `min_batch` and `max_batch`.

Together they reach **99.95%** frame utilisation on AVA and 99.98% on AVA plus
WASD, which is why the model still needs no padding mask. Holding
`batch * frames` roughly constant also stabilises activation memory, the `B*T`
population the visual frontend's batch norm sees, and the per-frame gradient
noise of the mean-reduced losses. The clamp exists because the contrastive term
is a `B`-way classification whose difficulty would otherwise swing with the
chunk length.

### Context prefix

A chunk boundary creates a second, subtler problem. The model is causal, so a
frame near the start of a chunk can only attend to the few frames before it -
yet its projection target depends on 2 s of history. Without help, 31.6% of
scored frames on AVA plus WASD are asked to predict from history the encoder was
never shown.

The loader therefore reads `hist_frames` extra frames *before* each chunk and
marks them unscored. They provide history and never contribute to a loss, so:

- every frame is still scored exactly once per epoch,
- the artificial starvation disappears entirely - the remaining 21.5% are all
  real track starts, where the missing history is genuine and the model has to
  cope with it at inference too,
- the cost is 1.15x compute and no extra storage.

Overlapping the chunks instead would duplicate the data on disk *and* keep
training on the truncated copies, since it adds well-formed frames beside the
starved ones rather than removing them.

Validation ignores all of this: it runs whole tracks at batch size one, so every
native frame receives a score.

Because the train loader shards itself across ranks, `Trainer` is constructed
with `use_distributed_sampler=False` and validation gets an explicit
`DistributedSampler`.

Validate every completed dataset and compare amplitude distributions before
training:

```bash
python -m preprocess.asd.validate /data/AVA/packed/train /data/WASD/packed/train
```

The report shows the stored source levels beside the levels the loader produces
after normalization. Large source differences between corpora are expected;
large differences in the *normalized* RMS usually mean peak limiting,
near-silence, or malformed input that should be inspected.
