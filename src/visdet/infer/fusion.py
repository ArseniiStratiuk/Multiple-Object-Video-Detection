"""Weighted Boxes Fusion of several prediction sets (TTA views or different models).

WBF (Solovyev et al., 2021) averages the coordinates of overlapping boxes of one class weighted
by score, instead of keeping only the highest-scoring box as NMS does. Coordinates are
normalised by the original frame size for the ensemble-boxes implementation.

WBF expects each input set to be one model's final detections, i.e. after NMS. Query-based models
(RT-DETR, D-FINE) emit every query, so one object can carry dozens of low-score duplicates from the
same model; WBF would count them as votes and average the score down. With `nms_per_set` each set
is first reduced by class-wise NMS at the fusion IoU.
"""

import numpy as np
import pandas as pd
from ensemble_boxes import nms, weighted_boxes_fusion

from visdet.data.index import Tables
from visdet.eval.coco import PRED_COLUMNS


def fuse(
    sets: list[pd.DataFrame], tables: Tables, weights=None, iou: float = 0.6,
    skip: float = 0.001, conf_type: str = "avg", nms_per_set: bool = False,
) -> pd.DataFrame:  # fmt: skip
    size = tables.frames.set_index("image_id")[["width", "height"]]
    sets = [s[(s["w"] > 0.5) & (s["h"] > 0.5)] for s in sets]  # degenerate boxes carry no area
    groups = [dict(tuple(s.groupby("image_id"))) for s in sets]
    image_ids = sorted(set().union(*[g.keys() for g in groups]))
    out = []
    for image_id in image_ids:
        w, h = size.loc[image_id]
        boxes, scores, labels = [], [], []
        for g in groups:
            p = g.get(image_id)
            if p is None or not len(p):
                boxes.append(np.zeros((0, 4)))
                scores.append(np.zeros(0))
                labels.append(np.zeros(0))
                continue
            xyxy = np.column_stack([p.x / w, p.y / h, (p.x + p.w) / w, (p.y + p.h) / h]).clip(0, 1)
            score, label = p.score.to_numpy(), p.cls.to_numpy()
            if nms_per_set:
                xyxy, score, label = nms([xyxy], [score], [label], iou_thr=iou)
            boxes.append(xyxy)
            scores.append(score)
            labels.append(label)
        b, s, c = weighted_boxes_fusion(
            boxes, scores, labels, weights=weights, iou_thr=iou, skip_box_thr=skip,
            conf_type=conf_type,
        )  # fmt: skip
        out.append(
            pd.DataFrame(
                {
                    "image_id": image_id, "cls": c.astype(int), "score": s,
                    "x": b[:, 0] * w, "y": b[:, 1] * h,
                    "w": (b[:, 2] - b[:, 0]) * w, "h": (b[:, 3] - b[:, 1]) * h,
                }
            )
        )  # fmt: skip
    return pd.concat(out, ignore_index=True)[PRED_COLUMNS]
