import argparse
from pathlib import Path

import torch
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    RichProgressBar,
)

from config.setup import load_config
from data.vap_dm import VAPDataModule
from models.release import load_weights, merge_config, resolve
from projection_window import ProjectionWindow
from tasks.vap_task import VAPTask


def parse_args():
    parser = argparse.ArgumentParser(description="Train or evaluate VAP")
    parser.add_argument(
        "--config", default=Path(__file__).resolve().parent / "config/yaml/vap.yaml"
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
        default="VAP",
        help="W&B project the run is logged to (with --wandb)",
    )
    parser.add_argument("--name", help="run name used for the checkpoint directory")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override one config value, e.g. --set vap.temporal.pos_encoding=rope",
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


def build_checkpoint(vap_cfg, checkpoint_dir):
    """Always keep `last.ckpt`; keep a selected `best.ckpt` only if asked to.

    Selecting on a validation metric is only meaningful with a held-out split.
    Setting `checkpoint_monitor` to nothing says there is none, and the final
    epoch becomes the model the run produced.
    """
    monitor = vap_cfg.get("checkpoint_monitor", "val/loss")
    if not monitor:
        return ModelCheckpoint(dirpath=checkpoint_dir, save_top_k=0, save_last=True)
    return ModelCheckpoint(
        monitor=monitor,
        mode=vap_cfg.get("checkpoint_mode", "min"),
        save_top_k=1,
        dirpath=checkpoint_dir,
        filename="best",
        save_last=True,
    )


def build_trainer(cfg, logger, checkpoint_dir):
    """Trainer whose checkpoints and logs all land under `checkpoint_dir`."""
    vap_cfg = cfg["vap"]
    return Trainer(
        default_root_dir=checkpoint_dir,
        max_epochs=vap_cfg["max_epochs"],
        logger=logger,
        callbacks=[
            build_checkpoint(vap_cfg, checkpoint_dir),
            RichProgressBar(),
            LearningRateMonitor("step"),
        ],
        log_every_n_steps=100,
        accelerator="auto",
        devices="auto",
        precision=vap_cfg.get("precision", "bf16-mixed"),
        gradient_clip_val=vap_cfg.get("gradient_clip_val", 1.0),
        accumulate_grad_batches=vap_cfg.get("accumulate_grad_batches", 1),
    )


def main():
    args = parse_args()
    if args.test and not (args.checkpoint or args.weights):
        raise SystemExit("--test needs --checkpoint (a .ckpt) or --weights (a release)")

    cfg = apply_overrides(load_config(args.config, "vap"), args.set)
    weights_path = None
    if args.weights:
        weights_path, release_config = resolve(args.weights)
        cfg = merge_config(cfg, release_config, "vap")
    seed_everything(args.seed, workers=True)
    torch.set_float32_matmul_precision(cfg["vap"].get("matmul_precision", "high"))

    logger = None
    run_id = args.name or "local"
    if args.wandb:
        from lightning.pytorch.loggers import WandbLogger

        logger = WandbLogger(project=args.wandb_project, name=args.name)
        run_id = args.name or logger.experiment.id

    projection = ProjectionWindow(**cfg["vap"]["projection_window"])
    print(f"VAP setup: arch={cfg['vap'].get('arch', 'mono')} {projection}")

    data = VAPDataModule(cfg, projection)
    model = VAPTask(cfg, projection)
    if weights_path is not None:
        load_weights(model.model, weights_path)
        print(f"loaded release weights from {weights_path}")
    trainer = build_trainer(cfg, logger, Path(cfg["checkpoint_dir"]) / run_id)

    if not args.test:
        trainer.fit(model, datamodule=data)
    data.setup("test")
    # A release is already loaded into the model; a ckpt_path would replace it.
    if weights_path is not None:
        ckpt_path = None
    else:
        # Without a selected best there is only the final epoch to fall back on.
        fallback = "best" if cfg["vap"].get("checkpoint_monitor", "val/loss") else "last"
        ckpt_path = args.checkpoint or fallback
    trainer.test(
        model,
        dataloaders=[data.tune_dataloader(), data.test_dataloader()],
        ckpt_path=ckpt_path,
    )


if __name__ == "__main__":
    main()
