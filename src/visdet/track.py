"""Weights & Biases logging: one run per experiment and fold, in project visdrone-hw2.

torchvision runs log training losses, per-epoch validation mAP, final metrics and prediction
overlays into one run. Ultralytics runs keep the trainer's own run (losses, per-epoch mAP from
its evaluator) and add a second run in the same group with the final metrics and overlays.
Nothing is logged when WANDB_MODE=disabled or on non-zero DDP ranks.
"""

import os

PROJECT = "visdrone-hw2"
_run = None


def start(cfg: dict, job_type: str) -> None:
    global _run
    if os.environ.get("WANDB_MODE") == "disabled" or int(os.environ.get("RANK", "0")) != 0:
        return
    import wandb

    group = f"{cfg['name']}-fold{cfg['fold']}"
    name = group if job_type == "train" else f"{group}-{job_type}"
    _run = wandb.init(project=PROJECT, name=name, group=group, job_type=job_type, config=cfg)


def log(values: dict, step: int | None = None) -> None:
    if _run is not None:
        _run.log(values, step=step)


def summary(values: dict) -> None:
    if _run is not None:
        _run.summary.update(values)


def images(key: str, arrays: list, captions: list[str]) -> None:
    if _run is not None:
        import wandb

        _run.log({key: [wandb.Image(a, caption=c) for a, c in zip(arrays, captions, strict=True)]})


def finish() -> None:
    global _run
    if _run is not None:
        _run.finish()
        _run = None
