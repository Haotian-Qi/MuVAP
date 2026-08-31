from pathlib import Path

from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader

from data.dataloaders.fisher import Fisher, FisherEvent
from models.vap import audio_channels


class VAPDataModule(LightningDataModule):
    """Fisher segments for fitting, annotated turn-taking events for testing."""

    def __init__(self, cfg, projection):
        super().__init__()
        vap_cfg = cfg["vap"]
        self.root = Path(cfg["fisher_path"])
        self.projection = projection
        self.batch_size = vap_cfg["batch_size"]
        self.num_workers = vap_cfg["num_workers"]
        self.channels = audio_channels(vap_cfg)
        self.swap_channels = bool(vap_cfg.get("swap_channels", False))
        self.source = str(vap_cfg.get("source", "npy")).lower()

    def _split(self, name):
        path = self.root / f"{name}.txt"
        if not path.exists():
            raise FileNotFoundError(
                f"missing Fisher split {path}; run preprocess/vap/00_prep_fisher.py first"
            )
        return path

    def _event_splits(self, names):
        paths = [self.root / f"{name}_events.txt" for name in names]
        found = [path for path in paths if path.exists()]
        if not found:
            raise FileNotFoundError(
                f"none of {[p.name for p in paths]} exist; run preprocess/vap/02_prep_event.py first"
            )
        return found

    def setup(self, stage=None):
        if stage in ("fit", None):
            self.train_dataset = self._segments("train", swap=self.swap_channels)
            self.val_dataset = self._segments("val")

        if stage in ("test", None):
            # The probe is fitted on held-out events and scored on the test events,
            # so the two sets must never share a conversation group.
            self.tune_dataset = FisherEvent(
                self.root,
                self._event_splits(["train", "val"]),
                channels=self.channels,
                source=self.source,
            )
            self.test_dataset = FisherEvent(
                self.root,
                self._event_splits(["test"]),
                channels=self.channels,
                source=self.source,
            )

    def _segments(self, name, swap=False):
        return Fisher(
            self.root,
            self._split(name),
            self.projection,
            channels=self.channels,
            frame_hz=self.projection.frame_hz,
            swap_channels=swap,
            source=self.source,
        )

    def _loader(self, dataset, shuffle=False):
        workers = self.num_workers
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=workers,
            pin_memory=True,
            persistent_workers=bool(workers),
            prefetch_factor=2 if workers else None,
            drop_last=shuffle,
        )

    def train_dataloader(self):
        return self._loader(self.train_dataset, shuffle=True)

    def val_dataloader(self):
        return self._loader(self.val_dataset)

    def tune_dataloader(self):
        return self._loader(self.tune_dataset)

    def test_dataloader(self):
        return self._loader(self.test_dataset)
