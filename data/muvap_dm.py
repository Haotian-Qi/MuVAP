"""Lightning data module for packed AVCC conversations."""

import torch.distributed as dist
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader

from data.dataloaders.conversation import (
    ConversationCollator,
    PackedConversationDataset,
    SpeakerFrameBudgetSampler,
)
from preprocess.muvap.schema import EVENT_KIND, SEGMENT_KIND

#: Batches each worker keeps ahead of the model. On the raw path a batch is
#: hundreds of megabytes of face crops, and running ahead is what hides the read
#: behind the previous step's compute. Host memory in flight is
#: `num_workers x this x batch`, which is the reason it is not larger.
PREFETCH_BATCHES = 4


class MuVAPDataModule(LightningDataModule):
    """Segments for fitting, annotated turn events for testing.

    Both come out of the same pack format and the same reader, so a media pack
    and its extracted embedding pack are interchangeable here: `packed_muvap`
    simply points at whichever exists, and the task asks the dataset which one
    it got.
    """

    def __init__(self, cfg, gvap_projection, svap_projection):
        super().__init__()
        self.cfg = cfg
        self.gvap_projection = gvap_projection
        self.svap_projection = svap_projection

        muvap = cfg["muvap"]
        self.num_workers = muvap["num_workers"]
        # 30 s windows overlapping by 10 s at 25 Hz, the paper's segmentation.
        self.train_window = muvap.get("train_window", 750)
        self.train_overlap = muvap.get("train_overlap", 250)
        self.batch_size = muvap.get("batch_size")
        self.frame_budget = muvap.get("frame_budget", 4096)
        self.min_batch = muvap.get("min_batch", 1)
        self.max_batch = muvap.get("max_batch", 16)
        self.seed = muvap.get("seed", 0)
        # Validation and test are forward-only, so they afford a larger budget.
        self.val_frame_budget = muvap.get("val_frame_budget", 4 * self.frame_budget)
        self.val_max_batch = muvap.get("val_max_batch", 64)

    def _context(self, overlap):
        """Unscored frames read in front of each window.

        An overlapping grid already repeats the previous window's tail, so it
        carries its own history and needs none added. Without overlap a window
        starts cold, and the least it needs is the global projection window's
        own history - the span its first labels are computed from.
        """
        return 0 if overlap else self.gvap_projection.hist_frames

    def _replicas(self):
        if dist.is_available() and dist.is_initialized():
            return dist.get_world_size(), dist.get_rank()
        return 1, 0

    def _segments(self, roots, train: bool):
        """Windows, for fitting and for validating alike.

        Validation windows too, and it has to: a conversation is not a face
        track. The corpus runs to half-hour segments, and both stacks here are
        quadratic in their sequence - a 32-minute segment is 48k frames, whose
        ALiBi bias alone is larger than the card. Tiling it without overlap
        scores every frame exactly once, which is all running a segment whole
        ever bought, and each window reads the global projection window's
        history in front of it so no scored frame starts cold.
        """
        overlap = self.train_overlap if train else 0
        return PackedConversationDataset(
            roots,
            gvap_projection=self.gvap_projection,
            svap_projection=self.svap_projection,
            kind=SEGMENT_KIND,
            window=self.train_window,
            overlap=overlap,
            context=self._context(overlap),
        )

    def setup(self, stage: str | None = None):
        packed = self.cfg["packed_muvap"]
        if stage in ("fit", None):
            self.train_dataset = self._segments(packed["train"], train=True)
        if stage in ("fit", "validate", None):
            # Validation runs whole segments so every frame gets a score.
            self.val_dataset = self._segments(packed["val"], train=False)
        if stage in ("test", "predict", None):
            self.test_dataset = PackedConversationDataset(
                packed["test"], kind=EVENT_KIND
            )

    def _loader(self, dataset, frame_budget, max_batch, shuffle, pad, target=None):
        replicas, rank = self._replicas()
        sampler = SpeakerFrameBudgetSampler(
            dataset.chunk_lengths(),
            dataset.chunk_speakers(),
            frame_budget=frame_budget,
            batch_size=self.batch_size,
            min_batch=self.min_batch,
            max_batch=max_batch,
            shuffle=shuffle,
            drop_last=shuffle,
            seed=self.seed,
            num_replicas=replicas,
            rank=rank,
        )
        return DataLoader(
            dataset,
            batch_sampler=sampler,
            num_workers=self.num_workers,
            collate_fn=ConversationCollator(pad=pad, target=target),
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            prefetch_factor=PREFETCH_BATCHES if self.num_workers > 0 else None,
        )

    def train_dataloader(self):
        """Every training batch is one window long, whatever it holds.

        A segment's last window is short, and so is any segment shorter than the
        window itself. Padding those up rather than cropping the batch down
        keeps the frames and keeps the tensor shape constant across steps.
        """
        return self._loader(
            self.train_dataset,
            self.frame_budget,
            self.max_batch,
            shuffle=True,
            pad=True,
            target=self.train_window,
        )

    def val_dataloader(self):
        """Windows tiling every segment, grouped by length and speaker count.

        Evaluation needs every frame, so the batch is padded rather than cropped
        to its shortest member the way training's is. Grouping by length first
        keeps that padding small.
        """
        return self._loader(
            self.val_dataset,
            self.val_frame_budget,
            self.val_max_batch,
            shuffle=False,
            pad=True,
        )

    def test_dataloader(self):
        """Turn events, whose windows are already one fixed length."""
        return self._loader(
            self.test_dataset,
            self.val_frame_budget,
            self.val_max_batch,
            shuffle=False,
            pad=True,
        )
