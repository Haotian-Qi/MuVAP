import argparse
from pathlib import Path

import torch
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import (
    Callback,
    LearningRateMonitor,
    ModelCheckpoint,
    RichProgressBar,
)

from config.setup import load_config
from data.muvap_dm import MuVAPDataModule
from models.release import load_weights, merge_config, resolve
from projection_window import ProjectionWindow
from tasks.muvap_task import MuVAPTask


class BatchSamplerEpoch(Callback):
    """Advance the train batch sampler's epoch so its shuffle changes.

    Lightning only forwards `set_epoch` to `dataloader.sampler` and
    `dataloader.batch_sampler.sampler`, so a bare batch sampler never hears
    about the epoch and would replay one fixed batch order forever.
    """

    def on_train_epoch_start(self, trainer, pl_module):
        sampler = getattr(trainer.train_dataloader, "batch_sampler", None)
        set_epoch = getattr(sampler, "set_epoch", None)
        if callable(set_epoch):
            set_epoch(trainer.current_epoch)


def parse_args():
    parser = argparse.ArgumentParser(description="Train or evaluate MuVAP")
    parser.add_argument(
        "--config", default=Path(__file__).resolve().parent / "config/yaml/muvap.yaml"
    )
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--checkpoint", help="Lightning .ckpt to resume or evaluate")
    parser.add_argument(
        "--weights",
        help="published release directory (or its weights.pt) to evaluate; "
        "the model architecture comes from the release, data paths from --config",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument(
        "--wandb-project",
        default="MuVAP",
        help="W&B project the run is logged to (with --wandb)",
    )
    parser.add_argument("--name", help="run name used for the checkpoint directory")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override one config value, e.g. --set muvap.frame_budget=8192",
    )
    return parser.parse_args()


def apply_overrides(cfg, overrides):
    """Apply `--set a.b.c=value` onto a loaded config, parsed as YAML scalars."""
    import yaml

    for override in overrides:
        key, _, raw = override.partition("=")
        if not _:
            raise SystemExit(f"--set expects KEY=VALUE, got {override!r}")
        target = cfg
        *path, leaf = key.split(".")
        for step in path:
            if step not in target:
                raise SystemExit(f"unknown config key in --set {override!r}")
            target = target[step]
        if leaf not in target:
            raise SystemExit(f"unknown config key in --set {override!r}")
        target[leaf] = yaml.safe_load(raw)
    return cfg


def build_checkpoints(muvap_cfg, checkpoint_dir):
    """Always keep `last.ckpt`; keep a selected `best.ckpt` only if asked to.

    Two callbacks, not one: a single `ModelCheckpoint` given both a monitor and
    `save_last=True` writes a `last.ckpt` that tracks the selected checkpoint
    rather than the final epoch, so the final epoch would never be kept.
    """
    last = ModelCheckpoint(dirpath=checkpoint_dir, save_top_k=0, save_last=True)
    monitor = muvap_cfg.get("checkpoint_monitor", "val/loss")
    if not monitor:
        return [last]
    best = ModelCheckpoint(
        monitor=monitor,
        mode=muvap_cfg.get("checkpoint_mode", "min"),
        save_top_k=1,
        dirpath=checkpoint_dir,
        filename="best",
    )
    return [best, last]


def build_trainer(cfg, logger, checkpoint_dir):
    """Trainer whose checkpoints and logs all land under `checkpoint_dir`."""
    muvap_cfg = cfg["muvap"]
    return Trainer(
        default_root_dir=checkpoint_dir,
        max_epochs=muvap_cfg["max_epochs"],
        logger=logger,
        callbacks=[
            *build_checkpoints(muvap_cfg, checkpoint_dir),
            BatchSamplerEpoch(),
            RichProgressBar(),
            LearningRateMonitor("step"),
        ],
        log_every_n_steps=100,
        accelerator="auto",
        devices="auto",
        precision=muvap_cfg.get("precision", "bf16-mixed"),
        gradient_clip_val=muvap_cfg.get("gradient_clip_val", 1.0),
        accumulate_grad_batches=muvap_cfg.get("accumulate_grad_batches", 1),
        # Both loaders shard themselves, the way the ASD module's do.
        use_distributed_sampler=False,
    )


def main():
    args = parse_args()
    if args.test and not (args.checkpoint or args.weights):
        raise SystemExit("--test needs --checkpoint (a .ckpt) or --weights (a release)")

    cfg = apply_overrides(load_config(args.config, "muvap"), args.set)
    weights_path = None
    if args.weights:
        weights_path, release_config = resolve(args.weights)
        cfg = merge_config(cfg, release_config, "muvap")
    seed_everything(args.seed, workers=True)
    torch.set_float32_matmul_precision(cfg["muvap"].get("matmul_precision", "high"))

    logger = None
    run_id = args.name or "local"
    if args.wandb:
        from lightning.pytorch.loggers import WandbLogger

        logger = WandbLogger(project=args.wandb_project, name=args.name)
        run_id = args.name or logger.experiment.id

    global_projection = ProjectionWindow(**cfg["muvap"]["gvap_projection_window"])
    speaker_projection = ProjectionWindow(**cfg["muvap"]["svap_projection_window"])
    print(f"MuVAP setup: source={cfg['muvap'].get('source', 'embeddings')}")
    print(f"  global   {global_projection}")
    print(f"  speaker  {speaker_projection}")

    data = MuVAPDataModule(cfg, global_projection, speaker_projection)
    model = MuVAPTask(cfg, global_projection)
    if weights_path is not None:
        load_weights(model.model, weights_path)
        print(f"loaded release weights from {weights_path}")
    trainer = build_trainer(cfg, logger, Path(cfg["checkpoint_dir"]) / run_id)

    if not args.test:
        trainer.fit(model, datamodule=data)
    # A release is already loaded into the model; a ckpt_path would replace it.
    if weights_path is not None:
        ckpt_path = None
    else:
        # Without a selected best there is only the final epoch to fall back on.
        fallback = (
            "best" if cfg["muvap"].get("checkpoint_monitor", "val/loss") else "last"
        )
        ckpt_path = args.checkpoint or fallback
    trainer.test(model, datamodule=data, ckpt_path=ckpt_path)


if __name__ == "__main__":
    main()
