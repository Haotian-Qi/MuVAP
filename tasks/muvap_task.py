"""Lightning task for the multiparty fusion module.

Training is two losses at once: a classification over the global codebook, the
same one the VAP module is trained on, and a per-bin binary loss over each
visible speaker's own future. Testing is the AVCC turn-taking protocol, zero
shot - nothing is fitted on the test events. Three numbers come off each event,
and they ask different things of the model:

* **turn** - hold or shift, decoded from the global head exactly the way the VAP
  module decodes it, and logged under the same `f1_macro` and `bacc` names, so
  the multiparty and dyadic numbers are directly comparable;
* **previous** - which of the tracked speakers just held the floor, read off the
  per-speaker head. The model is never told, so this is a check that the fusion
  actually tracks who is speaking rather than only when speech stops;
* **next** - which of them takes the floor. This is the multiparty question the
  dyadic codebook cannot even pose, and it is reported three ways: from the
  per-speaker head alone, conditioned on what the global head says about the
  turn (`_gvap`), and conditioned on that plus the true previous speaker
  (`_gvap_gt`), which bounds what the conditioning could buy if the previous
  speaker were always identified correctly.

Both speaker questions run over the speakers the recording tracks. A voice the
annotation rosters but never puts on screen still counts toward the global
label - it is speech, and the conversation is what it is - but it is not an
answer to a question about which face is talking.

Results are broken down by how many speakers were in the conversation, because
a shift among five faces is a harder question than a shift among two and an
average over both hides it.
"""

from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from lightning.pytorch import LightningModule
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

from models.muvap import MultiModalVAP, build_frozen_encoders
from tasks.setup import (
    LogisticProber,
    get_adamw_optimizer,
    get_cosine_warmup_scheduler,
    get_optimizer_cfg,
    get_param_groups,
)
from tasks.vap_task import SHIFT_SCALES

#: Frames of per-speaker output averaged to read off who just held the floor.
#: One second: long enough to outvote a pause, short enough to end at the event.
PREVIOUS_FRAMES = 25
#: The per-speaker bin whose activity names the current speaker, and the bins
#: whose activity names the next one. The first bin covers the 0.2s straddling
#: the event; the later ones are far enough ahead to be about the next turn.
PREVIOUS_BIN = 0
NEXT_BINS = slice(2, None)


class MuVAPTask(LightningModule):
    def __init__(self, cfg, proj_win):
        super().__init__()
        self.save_hyperparameters(cfg, ignore=["proj_win"])
        self.model = MultiModalVAP(cfg["muvap"])
        # Present only for the raw media path, where the two frozen modules run
        # on the batch instead of having been run over the pack in advance.
        # `FrozenEncoders` keeps itself in eval mode and out of the optimizer.
        self.encoders = (
            build_frozen_encoders(cfg) if cfg["muvap"].get("source") == "media" else None
        )
        self.proj_win = proj_win
        self.test_outputs = []
        # Set when `muvap.probe` is on: events the logistic probe fits on,
        # filled from test dataloader 0 and never scored itself.
        self.probe = bool(cfg["muvap"].get("probe"))
        self.fit_outputs = []

    def forward(self, vap, asd, speaker_mask=None, return_embeddings=False):
        return self.model(
            vap, asd, speaker_mask=speaker_mask, return_embeddings=return_embeddings
        )

    def streams(self, batch):
        """The two input streams, however this batch happens to carry them."""
        if "vap_emb" in batch:
            return batch["vap_emb"], batch["asd_emb"]
        if self.encoders is None:
            raise RuntimeError(
                "this batch holds raw media, but no frozen encoders were built; "
                "set muvap.source: media in the config used to build the task"
            )
        return self.encoders(batch["audio"], batch["visual"])

    def _shared_step(self, batch):
        vap, asd = self.streams(batch)
        global_logits, speaker_logits = self(vap, asd, batch.get("visual_mask"))

        time_mask = batch["mask"].bool()
        global_target = batch["gvap_gt"].long().masked_fill(~time_mask, -100)
        gvap_loss = F.cross_entropy(
            global_logits.squeeze(1).transpose(1, 2),
            global_target,
            ignore_index=-100,
        )

        svap_loss = F.binary_cross_entropy_with_logits(
            speaker_logits, batch["svap_gt"].float(), reduction="none"
        )
        # A frame is scored for a speaker only where both hold: the chunk's own
        # scored span, and a face actually being on screen to answer for.
        valid = time_mask[:, None, :, None].expand_as(svap_loss)
        if "visual_mask" in batch:
            valid = valid & batch["visual_mask"].bool().unsqueeze(-1)
        svap_loss = svap_loss[valid].mean()
        return gvap_loss + svap_loss, gvap_loss, svap_loss

    def training_step(self, batch, batch_idx):
        loss, gvap_loss, svap_loss = self._shared_step(batch)
        self.log("train/loss", loss, prog_bar=True, sync_dist=True)
        self.log("train/gvap_loss", gvap_loss, sync_dist=True)
        self.log("train/svap_loss", svap_loss, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, gvap_loss, svap_loss = self._shared_step(batch)
        self.log("val/loss", loss, prog_bar=True, sync_dist=True)
        self.log("val/gvap_loss", gvap_loss, sync_dist=True)
        self.log("val/svap_loss", svap_loss, sync_dist=True)

    # -------------------------------------------------------------- evaluation

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        vap, asd = self.streams(batch)
        global_logits, speaker_logits, global_embedding, _ = self(
            vap, asd, batch.get("visual_mask"), return_embeddings=True
        )
        probabilities = speaker_logits.sigmoid()
        # Dataloader 0 is the probe's fitting pool whenever one is configured;
        # `MuVAPDataModule.test_dataloader` owns that ordering.
        fitting = self.probe and dataloader_idx == 0
        target = self.fit_outputs if fitting else self.test_outputs

        # An event is judged at the last frame it was given, which is the last
        # frame before the silence: the answer is still entirely in the future.
        # A truncated window is shorter than the batch it rides in, so this is
        # read off the mask rather than assumed to be the last column.
        last = batch["mask"].bool().sum(dim=1) - 1
        for index, (event, metadata) in enumerate(
            zip(batch["event"], batch["metadata"])
        ):
            end = int(last[index])
            speakers = probabilities[index, :, : end + 1]
            previous = speakers[:, -PREVIOUS_FRAMES:, PREVIOUS_BIN].mean(dim=1)
            following = speakers[:, -1, NEXT_BINS].sum(dim=-1)
            # Both questions ask which speaker, and only a speaker the recording
            # tracks can be the answer. The annotation rosters voices that are
            # never on screen so their speech reaches the global label; leaving
            # them in the running here would let a face-less row absorb a
            # prediction it can never be right about.
            candidates = torch.tensor(
                metadata["tracked"], dtype=torch.bool, device=previous.device
            )
            if candidates.any():
                previous = previous.masked_fill(~candidates, -torch.inf)
                following = following.masked_fill(~candidates, -torch.inf)
            rows = {
                speaker: row
                for row, speaker in enumerate(metadata["speaker_ids"])
            }
            target.append(
                {
                    # Carried so a prediction can be joined back to the row of
                    # the benchmark file it came from.
                    "sample_id": metadata["sample_id"],
                    # The row order the two speaker answers index into, so a
                    # predicted row can be named as the speaker it stands for.
                    "speaker_ids": list(metadata["speaker_ids"]),
                    "global": global_logits[index, 0, end].float().detach().cpu(),
                    # The judging frame of the global stream - the same frame
                    # the codebook readout is taken from, so the probe is an
                    # upper bound on what that representation carries.
                    "embedding": global_embedding[index, 0, end]
                    .float()
                    .detach()
                    .cpu()
                    .numpy(),
                    "label": int("SHIFT" in str(event["label"]).upper()),
                    "previous_pred": int(previous.argmax().item()),
                    "previous_true": rows[event["previous"]],
                    # Kept per event rather than stacked: the pooled condition
                    # mixes conversations of different size, so there is no
                    # common candidate axis to work along.
                    "next_scores": following.float().detach().cpu(),
                    "next_pred": int(following.argmax().item()),
                    "next_true": rows[event["following"]],
                    # The cell counts candidates, not roster rows: an off-screen
                    # voice makes the conversation no harder to answer about.
                    "speakers": int(candidates.sum()) or int(metadata["speakers"]),
                }
            )

    def on_test_epoch_end(self):
        # Grouped the same way the test events are, so each cell's probe is
        # fitted on the speaker count it is scored on.
        fitting = self._conditions(self.fit_outputs) if self.fit_outputs else {}
        for category, samples in self._conditions(self.test_outputs).items():
            global_logits = torch.stack([sample["global"] for sample in samples])
            labels = np.asarray([sample["label"] for sample in samples])
            self.log(
                f"test/{category}/previous_accuracy",
                accuracy_score(
                    [sample["previous_true"] for sample in samples],
                    [sample["previous_pred"] for sample in samples],
                ),
            )
            next_true = [sample["next_true"] for sample in samples]
            self.log(
                f"test/{category}/next_accuracy",
                accuracy_score(next_true, [sample["next_pred"] for sample in samples]),
            )
            # Every projection mode decodes hold and shift from the class, so
            # the global head is scored the same zero-shot way VAP is.
            prediction = self.proj_win.get_shift_hold(global_logits)["pred"].numpy()
            for suffix, floor in (("_gvap", "previous_pred"), ("_gvap_gt", "previous_true")):
                self.log(
                    f"test/{category}/next_accuracy{suffix}",
                    accuracy_score(
                        next_true,
                        [
                            self._conditioned_next(sample, int(shift), floor)
                            for sample, shift in zip(samples, prediction)
                        ],
                    ),
                )
            # Named exactly as the VAP module names them, so the dyadic and
            # multiparty runs of the same decision group together.
            self.log(
                f"test/{category}/f1_macro",
                f1_score(labels, prediction, average="macro"),
            )
            self.log(
                f"test/{category}/bacc", balanced_accuracy_score(labels, prediction)
            )
            for scale in SHIFT_SCALES:
                swept = self.proj_win.get_shift_hold(global_logits, shift_scale=scale)[
                    "pred"
                ].numpy()
                self.log(
                    f"ablation/{category}/f1_macro_scale_{scale}",
                    f1_score(labels, swept, average="macro"),
                )

            # A linear read of the same frame, fitted on held-out events.
            # `class_weight="balanced"` puts the boundary where a 50/50 test
            # set wants it, so a naturally skewed fitting pool needs no
            # downsampling of its own.
            pool = fitting.get(category)
            if pool:
                probe = LogisticProber().fit_predict(
                    np.stack([sample["embedding"] for sample in pool]),
                    np.asarray([sample["label"] for sample in pool]),
                    np.stack([sample["embedding"] for sample in samples]),
                )
                if probe is not None:
                    self.log(
                        f"probe/{category}/f1_macro",
                        f1_score(labels, probe, average="macro"),
                    )
                    self.log(
                        f"probe/{category}/bacc",
                        balanced_accuracy_score(labels, probe),
                    )
        # Per-event records, for scoring that does not belong in the task:
        # re-keying cells to an external benchmark file, or deriving identity
        # and hold/shift from one prediction.
        dump = self.hparams["muvap"].get("dump_predictions")
        if dump:
            torch.save({"test": self.test_outputs, "fit": self.fit_outputs}, dump)
            print(
                f"wrote {len(self.test_outputs)} test and {len(self.fit_outputs)} "
                f"fit records to {dump}"
            )
        self.test_outputs.clear()
        self.fit_outputs.clear()

    @staticmethod
    def _conditioned_next(sample, shift, floor):
        """Next speaker, constrained by what the global head says about the turn.

        A hold names the floor holder outright - the turn does not change hands,
        so whoever held it keeps it. A shift says it does change hands, which
        rules that speaker out and leaves the per-speaker head to pick among the
        rest. The conditioning is only as good as the floor holder it is given,
        which is why the true one is reported beside the predicted one.
        """
        previous = sample[floor]
        if not shift:
            return previous
        scores = sample["next_scores"].clone()
        if scores.numel() > 1:
            scores[previous] = -torch.inf
        return int(scores.argmax().item())

    @staticmethod
    def _conditions(outputs):
        """Every event, then each speaker count on its own.

        A cell is only meaningful beside the pooled number: the counts are
        unbalanced, so `all` is what the benchmark as a whole says and the rest
        is where the difficulty sits.
        """
        grouped = defaultdict(list)
        for output in outputs:
            grouped["all"].append(output)
            grouped[f"{output['speakers']}spk"].append(output)
        return grouped

    def configure_optimizers(self):
        cfg = self.hparams["muvap"]
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
            float(cfg["min_lr"]) if cfg.get("min_lr") else None,
        )
        return get_optimizer_cfg(optimizer, scheduler)
