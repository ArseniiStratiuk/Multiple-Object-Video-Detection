"""D-FINE (query-based, Peng et al. 2024) through the authors' code base.

Training runs the official train.py with a config from configs/dfine/;
this module turns a trained checkpoint into predictions in the shared DataFrame format.
The code base is a git checkout given by VISDET_DFINE (D-FINE imports itself as `src`).
"""

import os
import sys
from pathlib import Path

import pandas as pd


def register_ops() -> None:
    """Register visdet's extra transforms (visdet.models.dfine_ops) with D-FINE."""
    from visdet.models import dfine_ops  # noqa: F401


def _load(config: Path, checkpoint: Path, device: str, weights: str):
    import torch

    sys.path.insert(0, os.environ["VISDET_DFINE"])
    from src.core import YAMLConfig

    register_ops()  # configs may name visdet models such as DFINEEnhanced
    cfg = YAMLConfig(str(config))
    if "HGNetv2" in cfg.yaml_cfg:
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg.model.load_state_dict(state["ema"]["module"] if weights == "ema" else state["model"])
    model = cfg.model.deploy().to(device).eval()
    post = cfg.postprocessor.deploy().to(device).eval()
    return model, post, tuple(cfg.yaml_cfg["eval_spatial_size"])


def predict(
    config: Path,
    checkpoint: Path,
    images: pd.DataFrame,
    device: str = "cuda",
    batch: int = 16,
    weights: str = "ema",
) -> pd.DataFrame:
    """Boxes in pixels of each cached image (columns image_id, path); the input is stretched to
    the model's square size, as in D-FINE training. `weights` picks the checkpoint's "ema" or raw
    "model" state. With crop columns x0, y0, x1, y1 (visdet.infer.crops) each row is that crop
    of the image and boxes are in crop pixels."""
    import torch
    import torchvision.transforms.functional as F
    from PIL import Image

    model, post, (h, w) = _load(config, checkpoint, device, weights)
    parts = []
    crop = "x0" in images
    rows = images[["image_id", "path"] + (["x0", "y0", "x1", "y1"] if crop else [])].to_numpy()
    last = (None, None)  # consecutive crops of one image decode it once

    def read(row) -> "Image.Image":
        nonlocal last
        if last[0] != row[1]:
            last = (row[1], Image.open(row[1]).convert("RGB"))
        return last[1].crop(tuple(int(v) for v in row[2:])) if crop else last[1]

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        for start in range(0, len(rows), batch):
            chunk = rows[start : start + batch]
            pil = [read(row) for row in chunk]
            sizes = torch.tensor([im.size for im in pil], device=device)
            x = torch.stack([F.to_tensor(F.resize(im, [h, w])) for im in pil]).to(device)
            labels, boxes, scores = post(model(x), sizes)
            for row, lab, box, sc in zip(chunk, labels, boxes, scores, strict=True):
                box = box.float().cpu().numpy()
                parts.append(
                    pd.DataFrame(
                        {
                            "image_id": row[0],
                            "cls": lab.cpu().numpy().astype(int),
                            "score": sc.float().cpu().numpy(),
                            "x": box[:, 0],
                            "y": box[:, 1],
                            "w": box[:, 2] - box[:, 0],
                            "h": box[:, 3] - box[:, 1],
                        }
                    )
                )
    return pd.concat(parts, ignore_index=True)
