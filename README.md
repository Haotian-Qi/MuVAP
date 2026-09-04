# MuVAP

**MuVAP: Multimodal Multiparty Voice Activity Projection for Turn-taking
Prediction in the Wild** — Interspeech 2026.

Demo: https://interspeech2026anonymous-eng.github.io/demo/

MuVAP is built from modules that share one codebase and are released one at a
time. Each is self-contained: it trains, evaluates, and publishes weights on its
own, without the modules that have not landed yet.

| Module | Status | Entry point |
| --- | --- | --- |
| VAP — voice activity projection from audio | released | `train_vap.py` |
| ASD — audio-visual active speaker detection | released | `train_asd.py` |
| MuVAP — multiparty fusion of the two | to follow | — |

## Install

```bash
python -m pip install -e .              # add ".[mimi]" for the Mimi frontend
```

Install the PyTorch build matching your CUDA version first.

The frozen audio frontends download their pretrained weights the first time a
model is built. Fetch them ahead of time — worth doing before a batch job, on a
machine whose compute nodes have no outbound network:

```bash
python tools/fetch_frontends.py          # or: fetch_frontends.py cpc
```


## VAP

Voice Activity Projection: a causal model that predicts who will hold the floor
next, from audio alone. Four configurations share one training program and
differ only in the config file, so a result difference between them comes from
the setup rather than from a different pipeline.

| Configuration | Config | Input | Frontend | Codebook |
| --- | --- | --- | --- | --- |
| Role-based VAP | `config/yaml/vap_role.yaml` | 1 mixed channel | CPC | 136 role-relative classes |
| Speaker-based VAP | `config/yaml/vap_speaker.yaml` | 2 channels, one per speaker | CPC | 256 classes, one channel per row |

```bash
python train_vap.py --config config/yaml/vap_role.yaml --name role
```

### Positional encoding

`vap.temporal.pos_encoding` selects `alibi`, causal `rope`, `sinusoidal`, or
`none`. The stack stays causal regardless — only how position reaches attention
changes.

See [`docs/vap.md`](docs/vap.md) for what each setup means, what it logs, how to
prepare Fisher, and the exact causality and alignment contracts.


## ASD

Audio-visual active speaker detection: given a face track and the audio around
it, decide frame by frame whether that face is speaking. The reported number is
the official AVA-ActiveSpeaker mAP.

| Configuration | Config | Frontend |
| --- | --- | --- |
| ASD | `config/yaml/asd.yaml` | CPC |
| ASD + Mimi | `config/yaml/asd_mimi.yaml` | Mimi |

The frontends are the same two the VAP module uses, and both emit features at
the 25 Hz rate the visual frontend runs at, so they substitute for each other
here too.

```bash
python train_asd.py --config config/yaml/asd.yaml --name asd
```

Every source corpus is packed into one memory-mapped format first. A pack is a
decoded mirror of the corpus, not a model-ready tensor set: grayscale face
crops and one label per frame at the *native* video rate, plus 16 kHz audio at
its original gain. Sampling onto the 25 Hz model timeline and loudness
normalization happen in the dataloader, so one pack can serve models with
different conventions.

```bash
python -m preprocess.asd.prepare ava --split train \
  --loader /data/AVA/csv/train_loader.csv \
  --visual-root /data/AVA/clips_videos --audio-root /data/AVA/clips_audios
python -m preprocess.asd.validate /data/AVA/packed/train
```

`python -m preprocess.asd.prepare --help` lists the accepted source schemas.
See [`docs/preprocessing.md`](docs/preprocessing.md) and
[`docs/data_alignment.md`](docs/data_alignment.md) for the storage and
alignment contracts.

**Audio convention.** The model expects audio normalized to -25 dBFS RMS,
capped at 20 dB of gain and a 0.99 peak, with the level measured over the whole
recording rather than the evaluated slice. This is not the raw-waveform
convention other ASD codebases use, and a checkpoint evaluated on unnormalized
audio will not reproduce the mAP it was trained to. Packs store audio
unnormalized and the loader applies this; the constants live in
`data/media.py`.


## Running either module

`--set KEY=VALUE` overrides any config value (dotted path, YAML-parsed) and can
be repeated; it fails on a key the config does not already define, so a typo
cannot silently do nothing. `--wandb` enables tracking, `--name` chooses the
run directory.

`checkpoint_monitor` picks what `best.ckpt` selects on — `val/loss` for VAP,
`val/mAP_official` for ASD. Clear it where there is no held-out split to select
on, and the final epoch is the only artifact the run keeps:

```bash
python train_vap.py --config config/yaml/vap_role.yaml --set vap.checkpoint_monitor=
```

`--test` then falls back to `last` instead of `best`. See
[Weight releases](#weight-releases) for what each checkpoint is.

## Audio frontends

`vap.audio_encoder` selects the frozen frontend. Both emit 25 Hz features, which
is the rate the projection window labels, so they are drop-in alternatives.

| | `cpc` (default) | `mimi` |
| --- | --- | --- |
| Weights | CPC, LibriLight 60k | `kyutai/mimi`, from Moshi |
| Frozen params | 1.8M | 37.8M |
| Feature dim | 256 | 512 |

Both are causal, which is what makes the model usable as a streaming predictor.


## Weight releases

Trained weights live on the Hub at
[Haotian-Qi/MuVAP](https://huggingface.co/Haotian-Qi/MuVAP).

| Release | Module | Frontend | Score |
| --- | --- | --- | --- |
| `vap-speaker-cpc` | VAP | CPC | f1_macro 0.7310 |
| `vap-speaker-mimi` | VAP | Mimi | f1_macro 0.7755 |
| `vap-role-cpc` | VAP | CPC | f1_macro 0.7289 |
| `vap-role-mimi` | VAP | Mimi | f1_macro 0.7589 |
| `asd-cpc` | ASD | CPC | mAP_official 90.4983 |
| `asd-mimi` | ASD | Mimi | mAP_official 92.0429 |

`f1_macro` is zero-shot hold/shift on the Fisher events, at the shift prior each
codebook defaults to - 2.0 for the role releases, 1.0 for `vap-speaker-cpc`,
whose rows name channels. `mAP_official` is the AVA-ActiveSpeaker mAP.

Fetch one, then evaluate it:

```bash
huggingface-cli download Haotian-Qi/MuVAP --include 'vap-role-mimi/*' --local-dir weights

python train_vap.py --config config/yaml/vap_role.yaml \
    --test --weights weights/vap-role-mimi \
    --set fisher_path=/path/to/fisher
```

That reproduces the score in the table exactly. All five at once:

```bash
huggingface-cli download Haotian-Qi/MuVAP --local-dir weights
```

A run produces two checkpoints. `last.ckpt` is the final epoch — the artifact
the run committed to. `best.ckpt` is the epoch a validation metric picked, and
exists only when `checkpoint_monitor` is set; where there is no held-out split
to select on, clear it and the final epoch is the only artifact.

Publish either one with:

```bash
python tools/export_checkpoint.py <run>/last.ckpt <release dir> --module vap
```

A release carries the trained weights alone: no optimizer state, and no copy of
the frozen audio frontend, which is fetched from its own pretrained source when
the model is built. It ships `config.yaml` (the architecture) and
`provenance.json`, which records the checkpoint it came from and what that
checkpoint scored. Point `--weights` at a release directory to evaluate it:

```bash
python train_vap.py --config config/yaml/vap_role.yaml --test --weights <release>
```

The architecture comes from the release's own `config.yaml`; dataset roots and
output directories come from the `--config` you pass, so a release runs anywhere
without reproducing the paths it was trained with. A release that does not fit
the config it is given fails immediately, naming the offending weights.

Or load one directly:

```python
import yaml
from models.release import load_weights, resolve
from models.vap import build_vap

weights, config = resolve("path/to/release")
cfg = yaml.safe_load(open(config))["vap"]
model = load_weights(build_vap(cfg), weights).eval()
logits = model(waveform)          # [batch, frames, classes] at 25 Hz
```
