"""COCO detection metrics on a subset of labelled frames.

Ignored regions enter the ground truth as iscrowd annotations of every class, so detections
inside them are neither matched nor counted as false positives. The detection cap is 500 per
image, the limit the competition scores, instead of the COCO default of 100.

Predictions are a DataFrame with columns image_id, cls (0-based), score, x, y, w, h (pixels,
top-left corner).
"""

import numpy as np
import pandas as pd
from faster_coco_eval import COCO, COCOeval_faster

from visdet import CLASSES
from visdet.data.index import Tables

MAX_DETS = (1, 100, 500)
PRED_COLUMNS = ["image_id", "cls", "score", "x", "y", "w", "h"]
SUMMARY = ("mAP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large")


def _ground_truth(
    tables: Tables, image_ids: pd.Index, use_ignore: bool = True
) -> tuple[COCO, dict[str, int]]:
    frames = tables.frames.set_index("image_id").loc[image_ids]
    numeric = {name: i + 1 for i, name in enumerate(image_ids)}
    images = [
        {"id": numeric[name], "width": int(row.width), "height": int(row.height)}
        for name, row in frames.iterrows()
    ]
    objects = tables.objects[tables.objects["image_id"].isin(numeric)]
    ignore = tables.ignore[tables.ignore["image_id"].isin(numeric)]
    if not use_ignore:
        ignore = ignore.iloc[:0]
    annotations = [
        {
            "image_id": numeric[o.image_id],
            "category_id": int(o.cls) + 1,
            "bbox": [o.x, o.y, o.w, o.h],
            "area": o.w * o.h,
            "iscrowd": 0,
        }
        for o in objects.itertuples()
    ]
    annotations += [
        {
            "image_id": numeric[r.image_id],
            "category_id": c + 1,
            "bbox": [r.x, r.y, r.w, r.h],
            "area": r.w * r.h,
            "iscrowd": 1,
        }
        for r in ignore.itertuples()
        for c in range(len(CLASSES))
    ]
    for i, a in enumerate(annotations, start=1):
        a["id"] = i
    categories = [{"id": i + 1, "name": n} for i, n in enumerate(CLASSES)]
    gt = COCO()
    gt.dataset = {"images": images, "annotations": annotations, "categories": categories}
    gt.createIndex()
    return gt, numeric


def top_per_image(pred: pd.DataFrame, limit: int = MAX_DETS[-1]) -> pd.DataFrame:
    ranked = pred.sort_values("score", ascending=False)
    return ranked.groupby("image_id", sort=False).head(limit)


def evaluate(
    pred: pd.DataFrame, tables: Tables, image_ids, max_dets: int = MAX_DETS[-1],
    use_ignore: bool = True,
) -> dict:  # fmt: skip
    """Return summary metrics and per-class AP for the given labelled frames.

    max_dets and use_ignore exist to test which scoring rule the leaderboard follows.
    """
    image_ids = pd.Index(image_ids)
    gt, numeric = _ground_truth(tables, image_ids, use_ignore)
    pred = top_per_image(pred[pred["image_id"].isin(numeric)], max_dets)
    detections = np.column_stack(
        [
            pred["image_id"].map(numeric).to_numpy(float),
            pred[["x", "y", "w", "h", "score"]].to_numpy(float),
            pred["cls"].to_numpy(float) + 1,
        ]
    )
    if len(detections) == 0:
        return {name: 0.0 for name in SUMMARY} | {f"AP_{c}": 0.0 for c in CLASSES}
    dt = gt.loadRes(detections)
    ev = COCOeval_faster(gt, dt, "bbox", print_function=lambda *_: None)
    ev.params.maxDets = [1, min(100, max_dets), max_dets]
    ev.evaluate()
    ev.accumulate()
    ev.summarize()
    metrics = {name: float(ev.stats[i]) for i, name in enumerate(SUMMARY)}
    # precision: [iou thresholds, recall points, classes, area ranges, max dets]
    precision = ev.eval["precision"][:, :, :, 0, -1]
    for k, name in enumerate(CLASSES):
        p = precision[:, :, k]
        metrics[f"AP_{name}"] = float(p[p > -1].mean()) if (p > -1).any() else float("nan")
    return metrics
