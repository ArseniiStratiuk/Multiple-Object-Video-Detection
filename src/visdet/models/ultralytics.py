"""YOLO (anchor-free) and RT-DETR (query-based) through the Ultralytics trainer."""

import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from ultralytics import RTDETR, YOLO

WANDB_PROJECT = "visdrone-hw2"
TRAIN_KEYS = (
    "epochs", "batch", "imgsz", "seed", "workers", "lr0", "lrf", "optimizer", "cos_lr",
    "mosaic", "mixup", "copy_paste", "scale", "close_mosaic", "box", "cls", "dfl", "patience",
    "warmup_epochs", "warmup_bias_lr", "momentum", "fliplr", "degrees",
)  # fmt: skip


def albumentations(specs: list[dict] | None) -> list | None:
    """Albumentations transforms from config entries, e.g. {Downscale: {scale_range: [0.35, 1]}}.

    Ultralytics applies them to the final training image (after mosaic and the affine step) and
    serialises them for the DDP workers. Without entries, its default photometric set is used.
    """
    if not specs:
        return None
    import albumentations as A

    def build(spec: dict):
        ((name, params),) = spec.items()
        params = dict(params or {})
        if "transforms" in params:  # composition, e.g. Sequential or OneOf
            params["transforms"] = [build(t) for t in params["transforms"]]
        return getattr(A, name)(**params)

    return [build(spec) for spec in specs]


def _absolute(name: str) -> str:
    """Weights in the working directory become absolute paths for the DDP workers."""
    return str(Path(name).resolve()) if Path(name).exists() else name


def _model(name: str):
    return RTDETR(name) if "rtdetr" in Path(name).name else YOLO(name)


def train(cfg: dict, data_yaml: Path, out_dir: Path, device: str) -> Path:
    """Train with the Ultralytics trainer; logs and weights go to out_dir/visdrone-hw2/<run>.

    The trainer resolves `project` relative to the working directory and uses it as the W&B
    project name, so the call runs from out_dir with the shared project name.
    """
    model = _model(_absolute(cfg["model"]))
    args = {k: cfg[k] for k in TRAIN_KEYS if k in cfg}
    if "pretrained" in cfg:
        args["pretrained"] = _absolute(cfg["pretrained"])
    augmentations = albumentations(cfg.get("augmentations"))
    if augmentations:
        args["augmentations"] = augmentations
    cwd = os.getcwd()
    os.chdir(out_dir)
    try:
        model.train(
            data=str(data_yaml),
            project=WANDB_PROJECT,
            name=f"{cfg['name']}-fold{cfg['fold']}",
            exist_ok=True,
            device=device,
            max_det=cfg["max_det"],
            plots=True,
            deterministic=False,
            **args,
        )
    finally:
        os.chdir(cwd)
    return Path(model.trainer.save_dir).resolve() / "weights" / "best.pt"


def predict(cfg: dict, weights: Path, images: pd.DataFrame, device: str) -> pd.DataFrame:
    model = _model(str(weights))
    parts = []
    paths = images["path"].tolist()
    ids = images["image_id"].tolist()
    chunk = cfg.get("predict_batch", 16)
    flip = cfg.get("flip", False)
    # Crop columns (visdet.infer.crops): predict that crop of each image, boxes in crop pixels.
    boxes = images[["x0", "y0", "x1", "y1"]].astype(int).values if "x0" in images else None
    for start in range(0, len(paths), chunk):
        batch = paths[start : start + chunk]
        if boxes is not None:
            crops = boxes[start : start + chunk]
            batch = [
                cv2.imread(p)[y0:y1, x0:x1]
                for p, (x0, y0, x1, y1) in zip(batch, crops, strict=True)
            ]
        if flip:  # test-time horizontal flip; boxes are mirrored back below
            batch = [(cv2.imread(p) if isinstance(p, str) else p)[:, ::-1].copy() for p in batch]
        results = model.predict(
            batch,
            imgsz=cfg["imgsz"],
            conf=cfg["predict_conf"],
            max_det=cfg["max_det"],
            device=device,
            half=True,
            verbose=False,
        )
        for image_id, r in zip(ids[start : start + chunk], results, strict=True):
            xyxy = r.boxes.xyxy.cpu().numpy()
            if flip:
                width = r.orig_shape[1]
                xyxy = np.stack([width - xyxy[:, 2], xyxy[:, 1], width - xyxy[:, 0], xyxy[:, 3]], 1)
            parts.append(
                pd.DataFrame(
                    {
                        "image_id": image_id,
                        "cls": r.boxes.cls.cpu().numpy().astype(int),
                        "score": r.boxes.conf.cpu().numpy(),
                        "x": xyxy[:, 0],
                        "y": xyxy[:, 1],
                        "w": xyxy[:, 2] - xyxy[:, 0],
                        "h": xyxy[:, 3] - xyxy[:, 1],
                    }
                )
            )
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
