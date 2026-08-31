"""Lightning data module for canonical packed ASD datasets."""

import torch.distributed as dist
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader

from data.augment import AugmentConfig
from data.dataloaders.packed import (
    ASDCollator,
    FrameBudgetBatchSampler,
    PackedASDDataset,
)


class PackedASDDataModule(LightningDataModule):
    def __init__(self, cfg: dict, projection_window):
        super().__init__()
        self.cfg = cfg
        self.projection_window = projection_window
        asd = cfg["asd"]
        self.num_workers = asd["num_workers"]
        self.train_window = asd.get("train_window", 250)
        self.frame_budget = asd.get("frame_budget", 2048)
        self.min_batch = asd.get("min_batch", 4)
        self.max_batch = asd.get("max_batch", 16)
        self.seed = asd.get("seed", 0)
        # Validation is forward-only, so it affords a larger budget than training.
        self.val_frame_budget = asd.get("val_frame_budget", 4 * self.frame_budget)
        self.val_max_batch = asd.get("val_max_batch", 64)
        self.augment = AugmentConfig.from_cfg(asd.get("augment"))

    def _replicas(self):
        if dist.is_available() and dist.is_initialized():
            return dist.get_world_size(), dist.get_rank()
        return 1, 0

    def setup(self, stage: str | None = None):
        packed = self.cfg["packed_asd"]
        if stage in ("fit", None):
            # The context prefix is exactly the projection window's history, so
            # every scored frame can see the past its target depends on.
            self.train_dataset = PackedASDDataset(
                packed["train"],
                projection_window=self.projection_window,
                window=self.train_window,
                context=self.projection_window.hist_frames,
                jitter=True,
                augment=self.augment,
            )
        if stage in ("fit", "validate", "test", None):
            # Validation runs whole tracks so every native frame gets a score.
            self.val_dataset = PackedASDDataset(
                packed["val"], projection_window=self.projection_window
            )

    def train_dataloader(self):
        replicas, rank = self._replicas()
        sampler = FrameBudgetBatchSampler(
            self.train_dataset.chunk_lengths(),
            frame_budget=self.frame_budget,
            min_batch=self.min_batch,
            max_batch=self.max_batch,
            shuffle=True,
            drop_last=True,
            seed=self.seed,
            num_replicas=replicas,
            rank=rank,
        )
        return DataLoader(
            self.train_dataset,
            batch_sampler=sampler,
            num_workers=self.num_workers,
            collate_fn=ASDCollator(),
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            prefetch_factor=2 if self.num_workers > 0 else None,
        )

    def val_dataloader(self):
        """Batch whole tracks by grouping similar lengths and padding the rest.

        Evaluation needs every frame of every track, so the batch cannot be
        cropped to its shortest member the way training's is. Grouping by
        length first keeps the padding that replaces it small, which is what
        makes batching worth doing at all.

        Lightning's automatic sampler injection is off because the train loader
        shards itself, so this shards too. The last rank is padded by repeating
        batches; the task de-duplicates by entity before scoring.
        """
        replicas, rank = self._replicas()
        sampler = FrameBudgetBatchSampler(
            self.val_dataset.chunk_lengths(),
            frame_budget=self.val_frame_budget,
            min_batch=1,
            max_batch=self.val_max_batch,
            shuffle=False,
            drop_last=False,
            num_replicas=replicas,
            rank=rank,
        )
        return DataLoader(
            self.val_dataset,
            batch_sampler=sampler,
            num_workers=self.num_workers,
            collate_fn=ASDCollator(pad=True),
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            prefetch_factor=2 if self.num_workers > 0 else None,
        )
