# MuVAP: multiparty fusion

The fusion module reads two frozen streams - the VAP module's account of the
conversation as a whole, and the ASD module's account of each visible face -
and predicts two things per frame:

* a **global** class over the configured codebook, the same readout the VAP
  module is scored on, so hold and shift decode from it zero shot;
* a **per-speaker** future, four independent bins for each face on screen.

Neither frozen module is updated. What trains is the fusion: two projections
into a shared width, one attention pass across the speakers inside each frame,
one causal pass along time, and a gate that decides per channel how much of the
conversation-level embedding each speaker's own embedding should absorb.

```bash
python train_muvap.py --config config/yaml/muvap.yaml --name muvap
```

## Why the speaker axis is consumed inside a frame

A conversation has however many speakers it has, and the count changes between
segments. The frame encoder attends the single global query to the speakers of
one frame, which makes the speaker axis a *set*: it has no order, carries no
position, and is gone by the time anything sees the time axis. That is what lets
one set of weights serve two-person and five-person conversations, and it is why
the per-frame stack builds no positional encoding at all while the temporal
stack stays causal and ALiBi-biased like every other temporal stack here.

The cost is that a batch must agree on its speaker count, because the axis is a
real tensor dimension. Padding it would mean inventing a face that is invisible
in every frame, and the frame encoder already has to open its mask where *no*
face is visible - otherwise attention over an all-masked row returns NaN - so a
padded speaker would be attended to exactly there. The sampler groups by speaker
count instead.

## Two projection windows

The two heads are deliberately not the same kind of object.

`gvap_projection_window` is a codebook. A class names an ordered *pair* of
speakers - who holds the floor and who takes it - which is a statement about the
conversation, not about any one participant. `role_relative` ranks the speakers
down to that pair on a 1.4, 0.6, 0.6, 1.4 s window - two history bins and two
future bins each - and folds the resulting 8 bits onto 136 classes, because
`(Scurr, Snext)` and `(Snext, Scurr)` name the same state. Each role keeps its
own history in the row, so the ranking can be read back off the class and the
fold is a bijection.

This is the VAP module's own projection window, unchanged, which is what makes
`test/<cell>/f1_macro` here and on Fisher the same measurement on two corpora.

`role_future` is also available: it ranks on the same coarse window, then
re-encodes the ranked pair's future four times finer and drops the history. It
buys temporal resolution at the cost of the bijection - 48 of its 136 classes
then hold both a hold and its mirror shift - and it is not what the paper
reports.

`svap_projection_window` is not a codebook at all. One face can only answer for
itself, so its window stays four independent bins scored with a per-bin binary
loss, and only where that face is actually on screen.

## Packs

Conversations are packed in the same spirit as the ASD corpora
([preprocessing](preprocessing.md)): a pack is a decoded mirror of the corpus,
faces and labels at the native frame rate, audio at its original gain, with
every model-specific transform left to the loader. What the multiparty format
adds is the grouping the fusion needs and the ASD format cannot express - all of
a segment's faces, the one mixed recording they share, and one activity row per
speaker.

```text
AVCC/packed/segments.train/
  dataset.json
  shard-000000/
    faces.npy   # uint8 [speakers * native_frames, 112, 112], speaker-major
    audio.npy   # int16 mono 16 kHz, unnormalized, the whole segment
    labels.npy  # uint8 [speakers * native_frames]: bit 0 speaking, bit 1 visible
    bbox.npy    # float32 [speakers * native_frames, 4], xyxy normalized
    index.json  # offsets, speaker ids, tracked flags, source_fps, audio levels
```

### Speaker activity, and who counts as a speaker

Speaker activity is read from `rttm/<video>/<segment>.rttm`, one file per
segment, and a segment is a segment if `rttm/` names it.

A segment's roster is the union of its boxes and its activity file: everyone
boxed and everyone heard. Those are not the same set. The annotation also labels voices
that are never on screen - `B`, an off-screen speaker - and they matter, because
the global label is defined over everyone audible and treating their speech as
silence would corrupt the very thing the global head predicts. They get a row,
blank faces, and `visible = false`, so the per-speaker loss skips them.

`tracked` in the index is the other half of that: per speaker, whether the
recording ever boxes that face at all. It is deliberately segment-level rather
than per-window, and the corpus says why - across 3460 annotated turn
participants, none is untracked in its segment, while three are missing from
their own ten-second window. A per-window rule would have scored those three
wrong by construction. The previous- and next-speaker questions run over tracked
speakers only, and the `<n>spk` cell counts them, so an off-screen voice does not
make a two-way question look like a three-way one.

Two axes describe a pack. **Kind** is what a sample is: `segments` holds whole
segments, which the loader tiles into chunks per epoch; `events` holds one
window per annotated turn event, ending where the mutual silence begins.
**Source** is what a sample holds: `media` as above, or `embeddings`, which is
what the frozen modules made of that media.

```bash
python -m preprocess.muvap.prepare segments --root /data/AVCC --split train
python -m preprocess.muvap.prepare segments --root /data/AVCC --split val
python -m preprocess.muvap.prepare events   --root /data/AVCC --split test \
    --events /data/AVCC/test_events.txt
```

AVCC keeps its recordings outside the annotation tree. The canonical layout is
`orig_video/<video>/<segment>.mp4` beside `orig_audio/<video>/<segment>.wav`,
resolved next to `--root` and overridable with `--media-root`; `videos/` is
accepted as the older spelling. Where a segment has no WAV, the audio is read
from its MP4 instead - the shared loader brings either to 16 kHz mono, so a
separate track is a convenience rather than a requirement. Add `--only <video>`
or `--only <video>/<segment>` to pack one of them.

Face crops are decoded to a scratch array beside the pack and streamed into the
shard in blocks, so they cost a fixed buffer rather than growing with the
segment - this corpus runs to half-hour segments, whose crops are 1.1 GB at two
speakers and would otherwise decide how long a segment may be.

The audio is the larger cost and is deliberately left resident: decoding half an
hour of 44.1 kHz stereo down to 16 kHz mono peaks around 2.6 GB, and resampling
it in pieces would make MuVAP's audio differ from the ASD and VAP packs, which
all go through that one loader. Measured end to end, the longest segment in AVCC
(`GqShvscnBBs/seg01`, 30 min, 45503 frames) packs in 83 s at a 3.6 GB peak.

Validate a completed pack before training on it — the speaker-axis invariants it
checks are the ones a silent bug would otherwise carry into every batch:

```bash
python -m preprocess.muvap.validate /data/AVCC/packed/segments.train
```

The segment pack stores whole segments on purpose. Choosing a window is a
training decision and the loader makes it; windows baked into the pack would
duplicate the overlap on disk and read no faster.

### Turn events

`test_events.txt` is the benchmark. Eight whitespace-separated columns:

```text
video_id  segment_id  previous  start  duration  following  label  n_speakers
```

`start` and `duration` describe the mutual silence between two turns, and
`previous` and `following` are the speakers on either side of it - so a hold is
exactly an event whose two speakers are the same one. `validate` checks that
against the label rather than trusting either. `n_speakers` is the event's
speaker count, which the benchmark scorer keys its cells on.

An event's window **ends where the silence begins**, so the last frame the model
is given is the last frame of the previous speaker's turn and none of the pause
is shown. That matters more than it looks: in this file a pause under 0.5 s is a
shift 452 times against 6 holds, so the pause length alone very nearly solves the
task. It is part of the answer, and the model never sees it.

This is the VAP module's convention, arrived at from the other side. There,
`event_crop` *adds* `EVENT_PADDING_SEC` to reach a hair into the silence;
here, `extract_turn_events.py` has already trimmed 0.1 s off each side of the
gap before writing the file, so the `start` column is that same padded edge.
Applying the padding again would double-count it.

`--window-sec` is the most history a window carries, not a fixed size. An event
closer to the start of its segment keeps the history it has rather than being
dropped - the recording really does begin there, which is the same genuine
truncation the loader accepts at a track's first frames. **Every annotated event
is scored.**

What the file holds, and what survives packing:

| | |
| --- | --- |
| events | 1730, over 88 segments of 7 videos |
| labels | 865 hold, 865 shift - balanced exactly |
| splits | all 7 videos are `test`, and no `test` video is missing |
| **scored** | **1730 - all of them**, 865 hold, 865 shift |
| by candidate count | 933 events with 2 speakers on screen, 797 with 3 |
| truncated windows | 55 shorter than 10 s, the shortest 7 frames |

Those 55 are worth knowing about when a cell looks weak: a 7-frame window is
0.28 s of history, and nothing can do much with it. They are 3% of the
benchmark and they are in it, rather than quietly removed to flatter the score.

Batches group by speaker count and then by length, so the padding that variable
windows cost is 1.6% of frames.

Preprocessing fails by name on a segment missing any of its inputs rather than
quietly skipping it; `--skip-missing` turns that into a warning, at the cost of
whatever events the segment held.

## Embeddings, or raw media

Nothing in training changes the frozen modules, so running them every epoch
recomputes the same numbers. Extraction does it once:

```bash
python -m preprocess.muvap.extract \
    --pack /data/AVCC/packed/segments.train \
    --output /data/AVCC/packed/embeddings.segments.train \
    --vap-weights weights/vap-role-mimi --asd-weights weights/asd-mimi
```

An embedding pack carries its labels, boxes, and visibility, so it stands on its
own and is several times smaller than the media it came from. `muvap.source`
picks which one the model expects: `embeddings` reads the cache, `media` reads
faces and audio and runs `FrozenEncoders` on every batch, which needs `vap:` and
`asd:` sections in the config holding the architectures those weights were
trained with. Both paths run the same code on the same inputs; reach for `media`
to change or fine-tune a frozen module, and for `embeddings` otherwise.

## What it logs

Training logs `train/loss` with `train/gvap_loss` and `train/svap_loss` beside
it - the global codebook term and the per-speaker term that sum to it - and the
same three under `val/`. `val/loss` is what selects `best.ckpt`.

`test/` is the headline, `ablation/` sweeps the shift prior the VAP papers
report. Each is written for every cell:

| Metric | Question |
| --- | --- |
| `f1_macro`, `bacc` | hold or shift, from the global head |
| `previous_accuracy` | which tracked speaker just held the floor |
| `next_accuracy` | which tracked speaker takes it, from the speaker head alone |
| `next_accuracy_gvap` | the same, conditioned on the global head |
| `next_accuracy_gvap_gt` | conditioned on the global head and the true floor holder |

The last two are the paper's `+GVAP` and `+GVAP+GT`. A predicted hold names the
floor holder outright, since the turn does not change hands; a predicted shift
rules that speaker out and leaves the rest to the speaker head. The conditioning
is only as good as the floor holder it is handed, so the variant using the true
one is reported beside it as the bound on what better previous-speaker detection
could buy.

The first two are named exactly as the VAP module names them, and decoded by the
same `get_shift_hold` off the same codebook, so `test/2spk/f1_macro` here and
`test/SILENT/f1_macro` there are the same measurement on two corpora. Every AVCC
event is a mutual silence, so AVCC has no ACTIVE pool to compare against.

Cells are `all`, then one per speaker count - `2spk` and `3spk` on this
benchmark. The pooled `all` number is what the benchmark as a whole says; the
cells are unbalanced, so neither replaces the other. Chance is 0.5 on the turn
metrics in every cell, and 0.5 / 0.333 on the speaker metrics for `2spk` /
`3spk`. `previous_accuracy` is a check rather than a headline: the
model is never told who was speaking, so a chance-level value means the fusion
is tracking when speech stops without tracking whose it was.
