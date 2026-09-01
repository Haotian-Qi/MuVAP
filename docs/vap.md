# The VAP module

Voice Activity Projection setups share one training program. They differ in
config only - how many audio channels the model reads, which codebook
`ProjectionWindow` projects the VAD onto, and which frozen audio frontend runs
in front - so any result difference between them comes from the setup rather
than from a different pipeline.

| Experiment | Config | Frontend | Channels | `mode` | Classes |
| --- | --- | --- | --- | --- | --- |
| Original VAP | `vap_original.yaml` | CPC | 2 (one per speaker) | `speaker_based` | 256 |
| Role-based | `vap.yaml` | CPC | 1 (downmix) | `role_relative` | 136 |

Hold and shift come from the logits in both setups: `role_relative` names the
roles in the class, and `speaker_based` names the channels, which needs
`prev_spk` to say which one held the floor.

## What each setup is

**Role-Based** is the MuVAP+ default. Both speakers are mixed into one channel,
so the model is never told who is who. The class names the *roles* - current
speaker and next speaker - through an order-invariant pair codebook over two
history and two future bins. Hold and shift fall straight out of the logits
because the class already says which role continues.

**Original VAP** reproduces Ekstedt & Skantze (2022): one audio channel per
speaker, a shared encoder and causal self-attention stack applied to each
channel, cross-attention between the two channels, and one head over their
concatenation. The projection window works per channel - one channel per row of
the code - giving `2 speakers x 4 future bins = 8 bits = 256` classes. Because
the classes name *channels* rather than roles, reading hold and shift needs to
know which channel held the floor, which is why the event manifest's
`prev_spk` column is carried into the test step.

## Audio frontend

`vap.audio_encoder` selects the frozen frontend. Both land on the same 25 Hz
grid the projection window labels, so they are drop-in alternatives:

| | `cpc` (default) | `mimi` |
| --- | --- | --- |
| Weights | CPC, LibriLight 60k | `kyutai/mimi`, from Moshi |
| Frozen params | 1.8M | 37.8M |
| Feature dim | 256 | 512 |
| Native rate | 16 kHz | 24 kHz (resampled causally inside the encoder) |

```yaml
audio_encoder:
  name: mimi
  tap: encoder     # 25 Hz. `latent` is Mimi's own 12.5 Hz rate
```

Mimi is causal by construction - its SEANet convolutions left-pad and its
encoder transformer uses a sliding causal mask. Getting there causally
needs care in one place: `torchaudio`'s resampler convolves with a symmetric
sinc, so a plain 16 -> 24 kHz resample leaks about 9 output samples of future.
`causal_resample_16k_to_24k` delays the signal by 1 ms instead, which removes
the leak at the cost of a 1 ms lag against a 40 ms frame.

The `latent` tap is refused at build time under a 25 Hz codebook rather than
failing later on a shape mismatch.

Mimi's 512-dim features are bridged onto `d_model` by the model's encoder
projection, so `d_model` need not change.

## Positional encoding

`vap.temporal.pos_encoding` selects how position reaches attention. It never
changes what a frame can see: every temporal stack is causal, frame `t` attends
to frames `<= t` only, so a trained model stays usable as a streaming predictor.

- `alibi` - per-head linear distance penalty on the logits (the previous default).
- `rope` - **causal rotary position embedding**. Query and key are rotated by an
  angle proportional to the absolute frame index, so the attention logit depends
  only on the distance between two frames. Under the causal mask that distance is
  never negative, and no future frame can leak in through the encoding. A query
  shorter than its key stream is offset to the end of that stream, so streaming
  inference with a cached history lines up with training.
- `sinusoidal` - an additive input-side embedding, for ablation.
- `none` - no positional information beyond the causal mask itself.

`ffn` (`mlp` / `swiglu`) and `norm` (`layer` / `rms`) are independent knobs on
the same stack.

Switch encoding without editing the file:

```bash
python train_vap.py --config config/yaml/vap.yaml --set vap.temporal.pos_encoding=rope
```

## Data

Two paths in each config are the only machine-specific values:

```yaml
fisher_path: data/fisher      # the corpus
checkpoint_dir: runs/vap      # checkpoints and logs
```

`checkpoint_dir` is also the trainer's root, so Lightning's logs land beside
the checkpoints.

The corpus tree looks like this:

```text
fisher/
  p1/ p2/
    audio/<group>/<conv>.wav          # source, stereo
    regions_vap/<group>/<conv>_*_words.txt
    seg/audio/<group>/<conv>/<i>.npy  # 30 s stereo segment
    seg/vad/<group>/<conv>/<i>.npy    # 2 x (750 + 2*context) VAD frames
    tune/audio/...                    # 10 s mono event clips
    tune/audio_stereo/...             # 10 s stereo event clips (original VAP only)
  train.txt val.txt test.txt
  turn_events.txt
  train_events.txt val_events.txt test_events.txt
```

**Labels are not stored.** Segments hold the raw two-speaker VAD with
`--context-sec` of extra frames on each side - everything the widest projection
window needs - and the dataloader projects them at load time. Changing
`projection_window.mode` between the three setups is therefore a config edit,
not a re-run of preprocessing.

Build the tree:

```bash
python preprocess/vap/00_prep_fisher.py --config config/yaml/vap.yaml   # segments + VAD
python preprocess/vap/01_create_event.py --config config/yaml/vap.yaml  # turn events
python preprocess/vap/02_prep_event.py --config config/yaml/vap.yaml    # mono event clips
```

The original-VAP setup evaluates on two-channel clips. `config/yaml/vap_original.yaml`
ships with `source: raw`, which reads them straight from the source recording,
so nothing more is needed. To evaluate it from pre-cut clips instead, write a
two-channel copy - it lands in `tune/audio_stereo`, leaving the mono clips
untouched - and set `source: npy`:

```bash
python preprocess/vap/02_prep_event.py --config config/yaml/vap.yaml --channels 2
```

`--splits` defaults to `val test`, which is what evaluation reads. Adding
`train` cuts clips for roughly 2.5 M more events and is only worth it to
enlarge the probe's tuning pool.

If a projection window needs more context than a segment stores, the dataloader
says so by name and asks for a wider `--context-sec`; the 2 s default covers all
three setups shipped here.

## Running

```bash
# Role-Based, the default
python train_vap.py --config config/yaml/vap.yaml --name rolebased_alibi

# Role-Based with causal RoPE
python train_vap.py --config config/yaml/vap.yaml --name rolebased_rope \
  --set vap.temporal.pos_encoding=rope

# Original VAP, two channels
python train_vap.py --config config/yaml/vap_original.yaml --name originalvap
```

Evaluate a checkpoint without training:

```bash
python train_vap.py --config config/yaml/vap.yaml --test --checkpoint runs/vap/rolebased_alibi/best.ckpt
```

Add `--wandb` for tracking, `--seed` to change the seed, and `--name` to choose
the checkpoint directory. `--set KEY=VALUE` overrides any existing config key
(dotted path, value parsed as YAML) and can be repeated; it fails loudly on a
key the config does not already define, so a typo cannot silently do nothing.

## Metrics

`trainer.test` runs two dataloaders: the tune events fit the linear probe, the
test events are scored. Events are pooled three ways - `all`, `SILENT`,
`ACTIVE` - and three prefixes keep the panels apart:

- `test/<pool>/f1_macro`, `test/<pool>/bacc` - the headline. Hold and shift are
  decoded from the logits, zero shot, over `p_future` at the shift prior the
  codebook defaults to: 2.0 for `role_relative`, 1.0 for `speaker_based`, whose
  rows name channels and so also need the event's `prev_spk`.
- `probe/<pool>/f1_macro`, `probe/<pool>/bacc` - a logistic probe on the
  last-frame embedding. It bypasses the codebook, so the gap against `test/`
  is what the readout costs rather than anything about the model. Being fitted,
  it is a diagnostic and not a score.
- `ablation/<pool>/f1_macro_scale_<s>` - the shift prior swept 1.0 ... 3.0, the
  way the VAP papers report it. `speaker_based` gains about 0.045 from a prior
  of 2.0; the role setups are already at their optimum and gain 0.001.
