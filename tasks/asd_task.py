import torch
import torch.distributed as dist
import torch.nn.functional as F
from lightning.pytorch import LightningModule
from torchmetrics.classification import (
    BinaryConfusionMatrix,
    BinaryF1Score,
    BinaryPrecision,
    BinaryRecall,
)

from models.asd import AudioVisualASD
from tasks.asd_metrics import ava_active_speaker_map
from tasks.ava_evaluation import export_ava_predictions
from tasks.setup import (
    get_adamw_optimizer,
    get_cosine_warmup_scheduler,
    get_optimizer_cfg,
    get_param_groups,
)


class ASDTask(LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters(cfg)
        self.model = AudioVisualASD(cfg["asd"])
        self.precision = BinaryPrecision()
        self.recall = BinaryRecall()
        self.f1 = BinaryF1Score()
        self.confusion = BinaryConfusionMatrix()
        self.validation_entities = []
        evaluation = cfg.get("evaluation", {})
        self.ava_groundtruth = evaluation.get("ava_groundtruth")
        self.ava_predictions = evaluation.get("ava_predictions")

    def forward(self, audio, visual):
        return self.model(audio, visual, return_embeddings=True)

    @staticmethod
    def _masked_bce(logits, targets, scored):
        """Average binary cross-entropy over scored frames only."""
        loss = F.binary_cross_entropy_with_logits(
            logits, targets.float(), reduction="none"
        )
        weight = scored.float()
        if loss.dim() == 3:
            weight = weight.unsqueeze(-1)
        weight = weight.expand_as(loss)
        return (loss * weight).sum() / weight.sum().clamp_min(1.0)

    def _shared_step(self, batch):
        audio, visual, projection_labels, activity_labels, scored = batch[:5]
        projection, audio_logits, visual_logits, _, audio_emb, visual_emb = self(
            audio, visual
        )
        audio_logits = audio_logits.squeeze(-1)
        visual_logits = visual_logits.squeeze(-1)
        activity_logits = (audio_logits + visual_logits) / 2

        projection_loss = self._masked_bce(projection, projection_labels, scored)
        audio_loss = self._masked_bce(audio_logits, activity_labels, scored)
        visual_loss = self._masked_bce(visual_logits, activity_labels, scored)
        contrastive_loss = self._talknce(
            audio_emb, visual_emb, activity_labels, scored
        )
        loss = projection_loss + 0.4 * audio_loss + 0.4 * visual_loss + 0.3 * contrastive_loss
        losses = {
            "loss": loss,
            "projection_loss": projection_loss,
            "audio_loss": audio_loss,
            "visual_loss": visual_loss,
            "contrastive_loss": contrastive_loss,
        }
        return activity_logits, losses

    def training_step(self, batch, batch_idx):
        _, losses = self._shared_step(batch)
        for name, value in losses.items():
            self.log(f"train/{name}", value, prog_bar=name == "loss", sync_dist=True)
        return losses["loss"]

    def validation_step(self, batch, batch_idx):
        logits, losses = self._shared_step(batch)
        labels = batch[3].int()
        probabilities = logits.sigmoid()
        if len(batch) == 6:
            scores = probabilities.detach().cpu()
            targets = labels.detach().cpu()
            for index, metadata in enumerate(batch[5]):
                frames = min(int(metadata["model_frames"]), scores.shape[-1])
                self.validation_entities.append(
                    (
                        metadata["sample_id"],
                        scores[index, :frames],
                        targets[index, :frames],
                        float(metadata["source_fps"]),
                    )
                )
        scored = batch[4].bool()
        flat_scores, flat_labels = probabilities[scored], labels[scored]
        self.precision.update(flat_scores, flat_labels)
        self.recall.update(flat_scores, flat_labels)
        self.f1.update(flat_scores, flat_labels)
        self.confusion.update(flat_scores, flat_labels)
        for name, value in losses.items():
            self.log(f"val/{name}", value, prog_bar=name == "loss", sync_dist=True)

    def _validation_is_partial(self):
        """True when this pass deliberately saw only part of the val set.

        The official export needs one score for every groundtruth entity, so it
        cannot run during the sanity check or a truncated debug pass. An
        unexpectedly incomplete set still raises inside the exporter.
        """
        return bool(
            self.trainer.sanity_checking
            or self.trainer.fast_dev_run
            or self.trainer.limit_val_batches != 1.0
        )

    def on_validation_epoch_end(self):
        if dist.is_available() and dist.is_initialized():
            gathered = [None] * dist.get_world_size()
            dist.all_gather_object(gathered, self.validation_entities)
            entities = [entity for item in gathered for entity in item]
        else:
            entities = self.validation_entities

        # The distributed val sampler pads the epoch by repeating samples, so
        # the same entity can arrive from several ranks. Ranks are not required
        # to agree bit for bit, so the first copy wins.
        scores_by_entity = {}
        labels_by_entity = {}
        fps_by_entity = {}
        for entity_id, entity_scores, entity_labels, source_fps in entities:
            if entity_id in scores_by_entity:
                continue
            scores_by_entity[entity_id] = entity_scores
            labels_by_entity[entity_id] = entity_labels
            fps_by_entity[entity_id] = source_fps
        if scores_by_entity:
            scores = torch.cat(list(scores_by_entity.values()))
            labels = torch.cat(list(labels_by_entity.values()))
        else:
            scores = torch.empty(0)
            labels = torch.empty(0, dtype=torch.int)
        map_25hz = ava_active_speaker_map(scores, labels)

        matrix = self.confusion.compute()
        false_positive_rate = matrix[0, 1] / matrix[0].sum().clamp_min(1)
        self.log("val/mAP_25hz", map_25hz, prog_bar=True)
        if (
            self.ava_groundtruth
            and self.ava_predictions
            and scores_by_entity
            and not self._validation_is_partial()
        ):
            # Only rank zero writes the prediction file, but every rank has to
            # log the metric or a ModelCheckpoint monitoring it will hang.
            official_map = None
            if self.trainer.is_global_zero:
                official_map = export_ava_predictions(
                    self.ava_groundtruth,
                    self.ava_predictions,
                    scores_by_entity,
                    fps_by_entity=fps_by_entity,
                )
            if dist.is_available() and dist.is_initialized():
                payload = [official_map]
                dist.broadcast_object_list(payload, src=0)
                official_map = payload[0]
            self.log("val/mAP_official", official_map, prog_bar=True)
        self.log("val/precision", self.precision.compute(), sync_dist=True)
        self.log("val/recall", self.recall.compute(), sync_dist=True)
        self.log("val/f1", self.f1.compute(), prog_bar=True, sync_dist=True)
        self.log("val/fpr", false_positive_rate, sync_dist=True)
        for metric in (
            self.precision,
            self.recall,
            self.f1,
            self.confusion,
        ):
            metric.reset()
        self.validation_entities.clear()

    @staticmethod
    def _talknce(audio, visual, labels, scored=None, temperature=0.07):
        positive = labels.bool()
        if scored is not None:
            positive = positive & scored.bool()
        if audio.shape[0] < 2 or not positive.any():
            return audio.sum() * 0

        audio = F.normalize(audio, dim=-1)
        visual = F.normalize(visual, dim=-1)
        logits = torch.einsum("btd,ctd->tbc", audio, visual) / temperature
        batch_size = audio.shape[0]
        targets = torch.arange(batch_size, device=audio.device).repeat(audio.shape[1])
        active = positive.transpose(0, 1).reshape(-1)
        return F.cross_entropy(logits.reshape(-1, batch_size)[active], targets[active])

    def configure_optimizers(self):
        cfg = self.hparams["asd"]
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
