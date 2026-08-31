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
from data.packed_asd_dm import PackedASDDataModule
from models.release import load_weights, merge_config, resolve
from projection_window import ProjectionWindow
from tasks.asd_task import ASDTask


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
    parser = argparse.ArgumentParser(description="Train or evaluate ASD")
    parser.add_argument(
        "--config", default=Path(__file__).resolve().parent / "config/yaml/asd.yaml"
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
        default="MuVAP-ASD",
        help="W&B project the run is logged to (with --wandb)",
    )
    parser.add_argument("--name", help="run name used for the checkpoint directory")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override one config value, e.g. --set asd.audio_encoder=mimi",
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


def build_checkpoints(asd_cfg, checkpoint_dir):
    """Always keep `last.ckpt`; keep a selected `best.ckpt` only if asked to.

    Two callbacks, not one: a single `ModelCheckpoint` given both a monitor and
    `save_last=True` writes a `last.ckpt` that tracks the selected checkpoint
    rather than the final epoch, so the final epoch would never be kept.

    `val/mAP_official` is the reported metric; `val/loss` mixes the projection
    and contrastive terms and does not track it closely. Setting
    `checkpoint_monitor` to nothing says there is no held-out split to select
    on, and the final epoch becomes the model the run produced.
    """
    last = ModelCheckpoint(dirpath=checkpoint_dir, save_top_k=0, save_last=True)
    monitor = asd_cfg.get("checkpoint_monitor", "val/mAP_official")
    if not monitor:
        return [last]
    best = ModelCheckpoint(
        monitor=monitor,
        mode=asd_cfg.get("checkpoint_mode", "max"),
        save_top_k=1,
        dirpath=checkpoint_dir,
        filename="best",
    )
    return [best, last]


def main():
    args = parse_args()
    if args.test and not (args.checkpoint or args.weights):
        raise SystemExit("--test needs --checkpoint (a .ckpt) or --weights (a release)")

    cfg = apply_overrides(load_config(args.config, "asd"), args.set)
    weights_path = None
    if args.weights:
        weights_path, release_config = resolve(args.weights)
        cfg = merge_config(cfg, release_config, "asd")

    seed_everything(args.seed, workers=True)
    asd_cfg = cfg["asd"]
    # Ampere and newer trade a little float32 mantissa for a large matmul speedup.
    torch.set_float32_matmul_precision(asd_cfg.get("matmul_precision", "high"))

    logger = None
    run_id = args.name or "local"
    if args.wandb:
        from lightning.pytorch.loggers import WandbLogger

        logger = WandbLogger(project=args.wandb_project, name=args.name)
        run_id = args.name or logger.experiment.id

    checkpoint_dir = Path(cfg["checkpoint_dir"]) / run_id
    callbacks = [
        *build_checkpoints(asd_cfg, checkpoint_dir),
        BatchSamplerEpoch(),
        RichProgressBar(),
        LearningRateMonitor("step"),
    ]
    projection = ProjectionWindow(**cfg["asd"]["projection_window"])
    encoder = asd_cfg.get("audio_encoder", "cpc")
    print(f"ASD setup: audio_encoder={encoder} {projection}")

    data = PackedASDDataModule(cfg, projection)
    model = ASDTask(cfg)
    if weights_path is not None:
        load_weights(model.model, weights_path)
        print(f"loaded release weights from {weights_path}")
    trainer = Trainer(
        # Checkpoints and Lightning's fallback CSV logs both land here rather
        # than as `lightning_logs/` inside the code checkout.
        default_root_dir=checkpoint_dir,
        max_epochs=asd_cfg["max_epochs"],
        logger=logger,
        callbacks=callbacks,
        log_every_n_steps=100,
        accelerator="auto",
        devices="auto",
        precision=asd_cfg.get("precision", "bf16-mixed"),
        gradient_clip_val=asd_cfg.get("gradient_clip_val", 1.0),
        accumulate_grad_batches=asd_cfg.get("accumulate_grad_batches", 1),
        # The train loader owns a frame-budget batch sampler that already
        # shards across ranks; Lightning must not wrap it in its own sampler.
        use_distributed_sampler=False,
    )
    if args.test:
        # A release is already in the model; a ckpt_path would overwrite it.
        trainer.validate(
            model,
            datamodule=data,
            ckpt_path=None if weights_path is not None else args.checkpoint,
        )
    else:
        trainer.fit(model, datamodule=data)


if __name__ == "__main__":
    main()
