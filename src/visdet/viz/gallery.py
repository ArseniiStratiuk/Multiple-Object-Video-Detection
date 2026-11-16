"""Overlay galleries: ground truth vs predictions on the original frames.

Colours: green = matched object, yellow = missed object, red = false positive above the score
threshold, grey wash = ignored region. Matching is greedy per class at IoU 0.5, the AP50 rule.
"""

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from visdet import CLASSES
from visdet.data.index import Tables
from visdet.paths import data_root

GREEN, YELLOW, RED, GREY = (60, 200, 60), (0, 220, 255), (40, 40, 230), (160, 160, 160)


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ax2, ay2, bx2, by2 = a[:, 0] + a[:, 2], a[:, 1] + a[:, 3], b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]
    iw = np.clip(np.minimum(ax2[:, None], bx2) - np.maximum(a[:, None, 0], b[:, 0]), 0, None)
    ih = np.clip(np.minimum(ay2[:, None], by2) - np.maximum(a[:, None, 1], b[:, 1]), 0, None)
    inter = iw * ih
    return inter / (a[:, None, 2] * a[:, None, 3] + b[:, 2] * b[:, 3] - inter)


def match(gt: pd.DataFrame, pred: pd.DataFrame, iou: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """Return matched flags for ground truth rows and prediction rows of one image."""
    gt_hit, pred_hit = np.zeros(len(gt), bool), np.zeros(len(pred), bool)
    for c in range(len(CLASSES)):
        gi = np.flatnonzero(gt["cls"].to_numpy() == c)
        pi = np.flatnonzero(pred["cls"].to_numpy() == c)
        if not len(gi) or not len(pi):
            continue
        pi = pi[np.argsort(-pred["score"].to_numpy()[pi])]
        overlap = iou_matrix(
            pred[["x", "y", "w", "h"]].to_numpy()[pi], gt[["x", "y", "w", "h"]].to_numpy()[gi]
        )
        for row, p in enumerate(pi):
            free = np.where(gt_hit[gi], -1, overlap[row])
            best = free.argmax()
            if free[best] >= iou:
                gt_hit[gi[best]] = pred_hit[p] = True
    return gt_hit, pred_hit


def render(tables: Tables, pred: pd.DataFrame, image_id: str, threshold: float) -> np.ndarray:
    frame = tables.frames.set_index("image_id").loc[image_id]
    image = cv2.imread(str(data_root() / frame["file"]))
    wash = image.copy()
    for r in tables.ignore[tables.ignore["image_id"] == image_id].itertuples():
        cv2.rectangle(wash, (int(r.x), int(r.y)), (int(r.x + r.w), int(r.y + r.h)), GREY, -1)
    image = cv2.addWeighted(wash, 0.45, image, 0.55, 0)
    gt = tables.objects[tables.objects["image_id"] == image_id]
    shown = pred[(pred["image_id"] == image_id) & (pred["score"] >= threshold)]
    gt_hit, pred_hit = match(gt, shown)
    boxes = [(r, GREEN if h else YELLOW) for r, h in zip(gt.itertuples(), gt_hit, strict=True)]
    boxes += [(r, RED) for r, h in zip(shown.itertuples(), pred_hit, strict=True) if not h]
    for r, colour in boxes:
        cv2.rectangle(image, (int(r.x), int(r.y)), (int(r.x + r.w), int(r.y + r.h)), colour, 2)
    text = f"{image_id}  objects {len(gt)}  missed {(~gt_hit).sum()}  false pos {(~pred_hit).sum()}"
    cv2.putText(image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    return image


def write(tables: Tables, pred: pd.DataFrame, image_ids, out: Path, threshold: float = 0.3) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for image_id in image_ids:
        cv2.imwrite(str(out / f"{image_id}.jpg"), render(tables, pred, image_id, threshold))


def render_ground_truth(tables: Tables, image_id: str) -> np.ndarray:
    """Converted labels drawn on the original frame, coloured by class, ignored regions washed."""
    from visdet.viz.style import PALETTE

    frame = tables.frames.set_index("image_id").loc[image_id]
    image = cv2.imread(str(data_root() / frame["file"]))
    wash = image.copy()
    for r in tables.ignore[tables.ignore["image_id"] == image_id].itertuples():
        cv2.rectangle(wash, (int(r.x), int(r.y)), (int(r.x + r.w), int(r.y + r.h)), GREY, -1)
    image = cv2.addWeighted(wash, 0.45, image, 0.55, 0)
    for r in tables.objects[tables.objects["image_id"] == image_id].itertuples():
        rgb = tuple(int(PALETTE[r.cls][i : i + 2], 16) for i in (1, 3, 5))
        cv2.rectangle(image, (int(r.x), int(r.y)), (int(r.x + r.w), int(r.y + r.h)), rgb[::-1], 2)
    return image[:, :, ::-1]
