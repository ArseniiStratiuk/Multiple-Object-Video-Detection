"""Sanity checks that must pass before any training run is trusted.

Each check compares the pipeline against a case with a known answer: ground truth scored as a
prediction, empty predictions, false positives hidden inside ignored regions, a submission
round trip, and fold leakage.
"""

import numpy as np
import pandas as pd

from visdet.data.folds import load_folds
from visdet.data.index import Tables, load_tables
from visdet.eval.coco import PRED_COLUMNS, evaluate
from visdet.submit import parse_rows, to_rows


def _ground_truth_as_prediction(tables: Tables, image_ids) -> pd.DataFrame:
    gt = tables.objects[tables.objects["image_id"].isin(set(image_ids))]
    return gt.assign(score=1.0)[PRED_COLUMNS]


def _check(name: str, ok: bool, detail: str) -> bool:
    print(f"{'PASS' if ok else 'FAIL'}  {name}: {detail}")
    return ok


def run(n_frames: int = 300) -> bool:
    tables = load_tables()
    folds = load_folds(tables)
    rng = np.random.default_rng(0)
    scored = folds.loc[folds["scored"] & (folds["fold"] == 0), "image_id"]
    ids = rng.choice(scored.to_numpy(), size=min(n_frames, len(scored)), replace=False)
    perfect = _ground_truth_as_prediction(tables, ids)
    results = []

    m = evaluate(perfect, tables, ids)
    results.append(_check("ground truth scores 1", m["mAP"] > 0.999, f"mAP={m['mAP']:.4f}"))

    m = evaluate(perfect.iloc[:0], tables, ids)
    results.append(_check("empty scores 0", m["mAP"] == 0.0, f"mAP={m['mAP']:.4f}"))

    ignore = tables.ignore[tables.ignore["image_id"].isin(set(ids))]
    inside = ignore.assign(
        cls=0,
        score=0.99,
        x=ignore["x"] + ignore["w"] * 0.25,
        y=ignore["y"] + ignore["h"] * 0.25,
        w=ignore["w"] * 0.5,
        h=ignore["h"] * 0.5,
    )[PRED_COLUMNS]
    m = evaluate(pd.concat([perfect, inside]), tables, ids)
    results.append(
        _check(
            "false positives in ignored regions are free",
            m["mAP"] > 0.999,
            f"{len(inside)} boxes, mAP={m['mAP']:.4f}",
        )
    )

    shifted = perfect.assign(x=perfect["x"] + perfect["w"] * 0.5)
    m = evaluate(shifted, tables, ids)
    results.append(_check("half-width shift is penalised", m["mAP"] < 0.5, f"mAP={m['mAP']:.4f}"))

    back = parse_rows(to_rows(perfect, ids))
    m = evaluate(back, tables, ids)
    results.append(
        _check("submission round trip keeps the score", m["mAP"] > 0.999, f"mAP={m['mAP']:.4f}")
    )

    per_video = folds.groupby("video")["fold"].nunique()
    fold_sizes = folds["fold"].value_counts().sort_index().tolist()
    results.append(
        _check(
            "each video in exactly one fold",
            bool((per_video == 1).all()) and bool((folds["fold"] >= 0).all()),
            f"{len(per_video)} videos, frames per fold {fold_sizes}",
        )
    )
    return all(results)
