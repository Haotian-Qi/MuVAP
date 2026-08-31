"""Lightning task wrapping any VAP architecture and projection setup.

Training is a single next-state classification over the configured codebook.
Testing is the paper's turn-taking protocol: hold/shift F1 on annotated events
where the codebook preserves speaker roles, and a linear probe on the model's
last-frame embedding, which works for every codebook including `role_future`.
"""

from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from lightning.pytorch import LightningModule
from sklearn.metrics import balanced_accuracy_score, f1_score

from models.vap import build_vap
from tasks.setup import (
    LogisticProber,
    get_adamw_optimizer,
    get_cosine_warmup_scheduler,
    get_optimizer_cfg,
    get_param_groups,
)

SHIFT_SCALES = tuple(round(0.1 * step, 1) for step in range(10, 31, 2))


class VAPTask(LightningModule):
    def __init__(self, cfg, proj_win):
        super().__init__()
        self.save_hyperparameters(cfg, ignore=["proj_win"])
        self.model = build_vap(cfg["vap"])
        self.proj_win = proj_win
        self.tune_outputs = []
        self.test_outputs = []

    @property
    def scores_shift_hold(self):
        """Whether the codebook lets hold and shift be read off the logits."""
        return self.proj_win.mode in {"role_relative", "speaker_based"}

    def forward(self, audio):
        return self.model(audio)

    def _shared_step(self, batch):
        audio, labels = batch
        logits = self(audio)
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
        audio, labels, _, timing, gap_type, prev_spk = batch
        logits, embedding = self.model(audio, return_embeddings=True)
        output = {
            "embedding": embedding[:, -1].float().detach().cpu(),
            "logits": logits[:, -1].float().detach().cpu(),
            "labels": labels.detach().cpu(),
            "timing": timing,
            "gap_type": gap_type,
            "prev_spk": prev_spk.detach().cpu(),
        }
        target = self.tune_outputs if dataloader_idx == 0 else self.test_outputs
        target.append(output)

    def on_test_epoch_end(self):
        tune = self._consolidate(self.tune_outputs)
        test = self._consolidate(self.test_outputs)

        for category, values in test.items():
            if self.scores_shift_hold:
                self._log_shift_hold(category, values)
            if category in tune:
                score = LogisticProber().fit_and_score(
                    tune[category]["embedding"],
                    tune[category]["labels"],
                    values["embedding"],
                    values["labels"],
                )
                self.log(f"test/{category}/probe_f1", score)

        self.tune_outputs.clear()
        self.test_outputs.clear()

    def _log_shift_hold(self, category, values):
        """Sweep the shift prior the way the VAP papers report it."""
        prev_spk = values["prev_spk"] if self.proj_win.mode == "speaker_based" else None
        best = (-1.0, None)
        for scale in SHIFT_SCALES:
            prediction = self.proj_win.get_shift_hold(
                values["logits"], shift_scale=scale, prev_speaker=prev_spk
            )["pred"].numpy()
            score = f1_score(values["labels"], prediction, average="macro")
            self.log(f"test/{category}/f1_scale_{scale}", score)
            if score > best[0]:
                best = (score, prediction)

        self.log(f"test/{category}/f1_best", best[0])
        self.log(
            f"test/{category}/bacc_best",
            balanced_accuracy_score(values["labels"], best[1]),
        )

    @staticmethod
    def _categories(timing, gap_type):
        """Report each event pool and the timing pool it belongs to."""
        return (timing, f"{timing}_{gap_type}")

    @classmethod
    def _consolidate(cls, outputs):
        grouped = defaultdict(
            lambda: {"embedding": [], "labels": [], "logits": [], "prev_spk": []}
        )
        for output in outputs:
            for index, label in enumerate(output["labels"]):
                for category in cls._categories(
                    output["timing"][index], output["gap_type"][index]
                ):
                    values = grouped[category]
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
