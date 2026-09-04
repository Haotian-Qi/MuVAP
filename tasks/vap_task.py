"""Lightning task wrapping any VAP architecture and projection setup.

Training is a single next-state classification over the configured codebook.
Testing is the papers' turn-taking protocol, zero shot: hold and shift are
decoded straight from the logits at each annotated event and scored on the
SILENT and ACTIVE pools. Two supporting views log beside it under their own
prefixes, so `test/` stays the headline: `probe/` fits a logistic probe on the
last-frame embedding, an upper bound on what the representation carries, and
`ablation/` sweeps the shift prior the VAP papers report.
"""

from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from lightning.pytorch import LightningModule
from sklearn.metrics import balanced_accuracy_score, f1_score

from models.vap import build_vap, needs_vad
from tasks.setup import (
    LogisticProber,
    get_adamw_optimizer,
    get_cosine_warmup_scheduler,
    get_optimizer_cfg,
    get_param_groups,
)


# The shift prior the VAP papers sweep over, reported as an ablation.
SHIFT_SCALES = tuple(round(0.1 * step, 1) for step in range(10, 31, 2))


class VAPTask(LightningModule):
    def __init__(self, cfg, proj_win):
        super().__init__()
        self.save_hyperparameters(cfg, ignore=["proj_win"])
        self.model = build_vap(cfg["vap"])
        # The anchored architecture is conditioned on voice activity as well as
        # audio, so its batches carry one more tensor than the others'.
        self.anchored = needs_vad(cfg["vap"])
        self.proj_win = proj_win
        self.tune_outputs = []
        self.test_outputs = []

    def forward(self, audio, vad=None):
        return self.model(audio, vad) if self.anchored else self.model(audio)

    def _shared_step(self, batch):
        audio, labels, *rest = batch
        logits = self(audio, rest[0] if self.anchored else None)
        if self.proj_win.mode == "independent":
            return F.binary_cross_entropy_with_logits(logits, labels.float())
        return F.cross_entropy(logits.transpose(1, 2), labels.long())

    def training_step(self, batch, batch_idx):
        loss = self._shared_step(batch)
        self.log("train/loss", loss, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._shared_step(batch)
        self.log("val/loss", loss, prog_bar=True, sync_dist=True)

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        audio, labels, _, timing, _, prev_spk, *rest = batch
        logits, embedding = (
            self.model(audio, rest[0], return_embeddings=True)
            if self.anchored
            else self.model(audio, return_embeddings=True)
        )
        output = {
            "embedding": embedding[:, -1].float().detach().cpu(),
            "logits": logits[:, -1].float().detach().cpu(),
            "labels": labels.detach().cpu(),
            "timing": timing,
            "prev_spk": prev_spk.detach().cpu(),
        }
        # The probe is fitted on held-out events and scored on the test events,
        # so the two never share a conversation group.
        target = self.tune_outputs if dataloader_idx == 0 else self.test_outputs
        target.append(output)

    def on_test_epoch_end(self):
        tune = self._pools(self.tune_outputs)
        for pool, values in self._pools(self.test_outputs).items():
            prev = values["prev_spk"] if self.proj_win.mode == "speaker_based" else None
            labels = values["labels"]

            prediction = self.proj_win.get_shift_hold(
                values["logits"], prev_speaker=prev
            )["pred"].numpy()
            self.log(f"test/{pool}/f1_macro", f1_score(labels, prediction, average="macro"))
            self.log(f"test/{pool}/bacc", balanced_accuracy_score(labels, prediction))

            for scale in SHIFT_SCALES:
                swept = self.proj_win.get_shift_hold(
                    values["logits"], shift_scale=scale, prev_speaker=prev
                )["pred"].numpy()
                self.log(
                    f"ablation/{pool}/f1_macro_scale_{scale}",
                    f1_score(labels, swept, average="macro"),
                )

            if pool in tune:
                probe = LogisticProber().fit_predict(
                    tune[pool]["embedding"], tune[pool]["labels"], values["embedding"]
                )
                if probe is not None:
                    self.log(f"probe/{pool}/f1_macro", f1_score(labels, probe, average="macro"))
                    self.log(f"probe/{pool}/bacc", balanced_accuracy_score(labels, probe))
        self.tune_outputs.clear()
        self.test_outputs.clear()

    @staticmethod
    def _pools(outputs):
        """Every event, then the SILENT and ACTIVE pools on their own."""
        grouped = defaultdict(
            lambda: {"embedding": [], "labels": [], "logits": [], "prev_spk": []}
        )
        for output in outputs:
            for index, label in enumerate(output["labels"]):
                for pool in ("all", output["timing"][index]):
                    values = grouped[pool]
                    values["embedding"].append(output["embedding"][index].numpy())
                    values["labels"].append(float(label))
                    values["logits"].append(output["logits"][index])
                    values["prev_spk"].append(output["prev_spk"][index])

        for values in grouped.values():
            values["embedding"] = np.asarray(values["embedding"])
            values["labels"] = np.asarray(values["labels"], dtype=np.int64)
            values["logits"] = torch.stack(values["logits"])
            values["prev_spk"] = torch.stack(values["prev_spk"]).long()
        return grouped

    def configure_optimizers(self):
        cfg = self.hparams["vap"]
        weight_decay = float(cfg["weight_decay"])
        optimizer = get_adamw_optimizer(
            get_param_groups(self.named_parameters(), weight_decay),
            float(cfg["lr"]),
            weight_decay,
        )
        scheduler = get_cosine_warmup_scheduler(
            optimizer,
            self.trainer.estimated_stepping_batches,
            float(cfg["warmup_ratio"]),
            float(cfg["min_lr"]),
        )
        return get_optimizer_cfg(optimizer, scheduler)
