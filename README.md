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
| ASD — audio-visual active speaker detection | to follow | — |
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
| Role-based | `config/yaml/vap.yaml` | 1 mixed channel | CPC | 136 role-relative classes |
| Role-future | `config/yaml/vap_role_future.yaml` | 1 mixed channel | CPC | 136 role classes, four future bins |
| Role-future + Mimi | `config/yaml/vap_role_future_mimi.yaml` | 1 mixed channel | Mimi | as above |
| Original VAP | `config/yaml/vap_original.yaml` | 2 channels, one per speaker | CPC | 256 classes, one channel per row |

## Pretrained weights

Releases carry the trained weights alone: no optimizer state, and no copy of the
frozen audio frontend, which is fetched from its own pretrained source when the
model is built. Point `--weights` at a release directory to evaluate it:

```bash
python train_vap.py --config config/yaml/vap.yaml --test --weights <release>
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

## Training

```bash
python train_vap.py --config config/yaml/vap.yaml --name rolebased
```

`--set KEY=VALUE` overrides any config value (dotted path, YAML-parsed) and can
be repeated; it fails on a key the config does not already define, so a typo
cannot silently do nothing. `--wandb` enables tracking.

Every run writes `last.ckpt`, the final epoch. It also writes `best.ckpt`,
selected by `checkpoint_monitor` (`val/loss` by default). Selecting on a
validation metric only means something with a genuinely held-out split, so
where there is none, clear the setting and use the final epoch:

```bash
python train_vap.py --config config/yaml/vap.yaml --set vap.checkpoint_monitor=
```

`--test` then falls back to `last` instead of `best`.

Publish a trained checkpoint with:

```bash
python tools/export_checkpoint.py <run>/best.ckpt <release dir>
```

## Audio frontends

`vap.audio_encoder` selects the frozen frontend. Both emit 25 Hz features, which
is the rate the projection window labels, so they are drop-in alternatives.

| | `cpc` (default) | `mimi` |
| --- | --- | --- |
| Weights | CPC, LibriLight 60k | `kyutai/mimi`, from Moshi |
| Frozen params | 1.8M | 37.8M |
| Feature dim | 256 | 512 |

Both are causal, which is what makes the model usable as a streaming predictor.

## Positional encoding

`vap.temporal.pos_encoding` selects `alibi`, causal `rope`, `sinusoidal`, or
`none`. The stack stays causal regardless — only how position reaches attention
changes.

See [`docs/vap.md`](docs/vap.md) for what each setup means, what it logs, how to
prepare Fisher, and the exact causality and alignment contracts.
