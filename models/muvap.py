"""Multiparty fusion of the VAP and ASD modules.

`MultiModalVAP` reads two frozen streams and predicts two things at once:

* a **global** class per frame, over whatever codebook the configured
  `ProjectionWindow` defines - the same readout the VAP module is scored on, so
  hold and shift come out of it zero-shot;
* a **per-speaker** class per frame, the raw future bins of each visible face.

The two are not independent, and the gate is where that is spent. The global
embedding is what the conversation as a whole is doing; each speaker's own
embedding is what one face is doing. A speaker who is about to take the floor
looks different in the light of the conversation than alone, so the global
stream is offered to every speaker and a sigmoid decides per channel how much
of it to take.

The streams arrive either precomputed or produced on the fly by
`FrozenEncoders`, which is the same code in both cases: the extraction tool and
the raw path run one implementation, so an embedding pack is a cache rather than
a second version of the model. How much history each frame was encoded with is
the one thing that separates them.
"""

from pathlib import Path

import torch
import torch.nn as nn
import yaml

from projection_window import ProjectionWindow

from .asd import AudioVisualASD
from .modules.attention import FrameTemporalEncoder
from .modules.projection import LinearHead
from .release import load_weights, module_config
from .vap import build_vap

#: What the two frozen modules emit, and therefore what the fusion reads. VAP
#: publishes its temporal width; ASD publishes twice its cross width, because
#: its fused stream is the two cross-attended streams concatenated.
VAP_EMBED_DIM = 256
ASD_EMBED_DIM = 512


def speaker_bins(cfg):
    """Outputs of the per-speaker head: one independent bin per future span.

    The speaker head is deliberately not a codebook. A codebook class names a
    *pair* of speakers, which is only meaningful for the conversation as a
    whole; one face can only answer for itself, so its window stays a set of
    independent bins scored with a per-bin binary loss.
    """
    window = ProjectionWindow(**cfg["svap_projection_window"])
    if window.mode != "independent" or window.num_hist_bins:
        raise ValueError(
            "svap_projection_window must be mode: independent with no history bins, "
            f"got mode: {window.mode} with {window.num_hist_bins}"
        )
    return window.n_bins


class MultiModalVAP(nn.Module):
    """Global and per-speaker voice activity projection from two frozen streams."""

    def __init__(self, cfg):
        super().__init__()
        model_dim = cfg["temporal"]["d_model"]
        if cfg["frame"]["d_model"] != model_dim:
            raise ValueError("MuVAP frame and temporal d_model must match")

        vap_dim = cfg.get("vap_dim", VAP_EMBED_DIM)
        asd_dim = cfg.get("asd_dim", ASD_EMBED_DIM)

        self.vap_enc = nn.Sequential(
            nn.LayerNorm(vap_dim),
            nn.Linear(vap_dim, model_dim),
        )
        self.asd_enc = nn.Sequential(
            nn.LayerNorm(asd_dim),
            nn.Linear(asd_dim, model_dim),
        )
        # A second, independent projection of the same ASD stream. The one above
        # is read as context by the frame encoder and comes back mixed with
        # every other speaker; this one has to stay one speaker's own account of
        # itself, which is what the per-speaker head is scored on.
        self.asd_enc2 = nn.Sequential(
            nn.LayerNorm(asd_dim),
            nn.Linear(asd_dim, model_dim),
        )
        self.conductor_gate = nn.Sequential(
            nn.Linear(model_dim * 2, model_dim),
            nn.Sigmoid(),
        )
        self.STE = FrameTemporalEncoder(cfg)
        self.gvap_head = LinearHead(
            model_dim, ProjectionWindow(**cfg["gvap_projection_window"]).n_classes
        )
        self.svap_head = LinearHead(model_dim, speaker_bins(cfg))

    def forward(self, vap, asd, speaker_mask=None, return_embeddings=False):
        """`[B, 1, T, vap_dim]` and `[B, S, T, asd_dim]` to two logit streams."""
        vap_embedding = self.vap_enc(vap)
        asd_context = self.asd_enc(asd)
        global_embedding, _ = self.STE(
            vap_embedding, asd_context, speaker_mask=speaker_mask
        )

        speaker_embedding = self.asd_enc2(asd)
        global_per_speaker = global_embedding.expand(
            -1, speaker_embedding.shape[1], -1, -1
        )
        gate = self.conductor_gate(
            torch.cat([speaker_embedding, global_per_speaker], dim=-1)
        )
        speaker_embedding = speaker_embedding + gate * global_per_speaker

        global_logits = self.gvap_head(global_embedding)
        speaker_logits = self.svap_head(speaker_embedding)
        if return_embeddings:
            return global_logits, speaker_logits, global_embedding, speaker_embedding
        return global_logits, speaker_logits


class FrozenEncoders(nn.Module):
    """The trained VAP and ASD modules, held in eval mode and never updated.

    One object with one contract, used in two places: the extraction tool that
    writes an embedding pack, and the raw dataloader path that skips the pack
    and encodes a batch as it arrives. Anything that would make those two
    disagree - a dropout left on, a gradient, a different slicing of the audio -
    would silently make a cached run and a raw run two different experiments.

    ASD sees every speaker in one pass, with the speaker axis folded into the
    batch: a conversation with five visible faces costs one ASD pass over five
    times as many face tracks, against one VAP pass over the shared recording.
    That is the memory cost to size a batch against on the raw path.
    """

    def __init__(self, vap, asd):
        super().__init__()
        self.vap = vap
        self.asd = asd
        self.eval()
        self.requires_grad_(False)

    def train(self, mode=True):
        """Stay in eval mode whatever the surrounding module is doing.

        `LightningModule.train()` walks every child, and a frozen encoder left
        in training mode would apply dropout and update the visual frontend's
        batch-norm statistics from the data it is only supposed to be reading.
        """
        return super().train(False)

    @torch.no_grad()
    def forward(self, audio, visual):
        """`[B, 1, samples]` and `[B, S, T, 112, 112]` to the two streams.

        Returns `[B, 1, T, vap_dim]` and `[B, S, T, asd_dim]`, laid out exactly
        as `MultiModalVAP` reads them and as the embedding pack stores them.
        """
        _, vap_embedding = self.vap(audio, return_embeddings=True)

        batch, speakers = visual.shape[0], visual.shape[1]
        # One ASD pass over B*S face tracks, each paired with its conversation's
        # audio: the mixed recording is what the trained ASD module expects, and
        # it is what tells it that a moving face is not the one being heard.
        faces = visual.flatten(0, 1)
        shared_audio = audio.repeat_interleave(speakers, dim=0)
        _, _, _, fused, _, _ = self.asd(shared_audio, faces, return_embeddings=True)
        asd_embedding = fused.unflatten(0, (batch, speakers))

        return vap_embedding.unsqueeze(1), asd_embedding


def frozen_pair(vap_weights, asd_weights, vap_cfg=None, asd_cfg=None):
    """Load two published releases and freeze them.

    The weights go in *before* anything is frozen, and that ordering is the
    point. `load_weights` excuses a missing weight only when the model says that
    parameter is frozen - which is how a release gets away with omitting the
    pretrained frontend. Freeze the whole pair first and every parameter is
    excused, so a release that silently failed to load would train as though it
    had worked, and the run would look fine and mean nothing.
    """
    vap_path, vap_config = module_config(vap_weights, "vap", vap_cfg)
    asd_path, asd_config = module_config(asd_weights, "asd", asd_cfg)
    return FrozenEncoders(
        load_weights(build_vap(vap_config), vap_path),
        load_weights(AudioVisualASD(asd_config), asd_path),
    )


def build_frozen_encoders(cfg):
    """The frozen pair a `source: media` run reads, from the configured releases."""
    muvap = cfg["muvap"]
    weights = {key: muvap.get(f"{key}_weights") for key in ("vap", "asd")}
    missing = sorted(key for key, value in weights.items() if not value)
    if missing:
        raise ValueError(
            "source: media runs the frozen modules on every batch, so it needs "
            f"{['muvap.' + key + '_weights' for key in missing]} pointing at published "
            "releases. Without them the encoders would be freshly initialised and "
            "the run would be training on noise."
        )
    return frozen_pair(weights["vap"], weights["asd"], cfg.get("vap"), cfg.get("asd"))


def load_fusion(weights, config=None):
    """A trained fusion, from a published release or a training checkpoint.

    A release carries its own `config.yaml`, which is the architecture to trust.
    A `.ckpt` carries the config it was trained with among its hyperparameters;
    `config` is only read for a checkpoint that predates that. Returns the model
    and the `muvap` config it was built from.
    """
    path = Path(weights)
    fallback = yaml.safe_load(open(config))["muvap"] if config else None
    if path.suffix != ".ckpt":
        weights_path, cfg = module_config(path, "muvap", fallback)
        return load_weights(MultiModalVAP(cfg), weights_path), cfg
    state = torch.load(path, map_location="cpu", weights_only=False)
    cfg = state.get("hyper_parameters", {}).get("muvap") or fallback
    if cfg is None:
        raise SystemExit(f"{path} carries no muvap config; pass the one it was trained with")
    model = MultiModalVAP(cfg)
    model.load_state_dict(
        {k[6:]: v for k, v in state["state_dict"].items() if k.startswith("model.")}
    )
    return model, cfg
