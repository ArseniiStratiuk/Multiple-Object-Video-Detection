"""Run comparison tables, training curves and error breakdowns for the model notebooks."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from visdet import CLASSES
from visdet.data.index import Tables
from visdet.viz.gallery import iou_matrix, match
from visdet.viz.style import MUTED, PALETTE

SIZE_EDGES = [0, 16, 32, 64, 96, 1e9]
SIZE_LABELS = ["<16", "16-32", "32-64", "64-96", ">=96"]
METRICS = ["mAP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large", "ms_per_image"]


def run_table(runs: Path, names: list[str], fold: int = 0) -> pd.DataFrame:
    rows = {}
    for name in names:
        path = runs / name / f"fold{fold}" / "metrics.json"
        if path.exists():
            m = json.loads(path.read_text())
            rows[name] = {k: m.get(k) for k in METRICS}
    table = pd.DataFrame(rows).T
    return table.astype(float).round({k: 4 for k in METRICS[:-1]} | {"ms_per_image": 1})


def class_table(runs: Path, names: list[str], fold: int = 0) -> pd.DataFrame:
    rows = {}
    for name in names:
        path = runs / name / f"fold{fold}" / "metrics.json"
        if path.exists():
            m = json.loads(path.read_text())
            rows[name] = {c: m.get(f"AP_{c}") for c in CLASSES}
    return pd.DataFrame(rows).T.astype(float).round(3)


def ultralytics_curve(runs: Path, name: str, fold: int = 0) -> pd.DataFrame:
    """Per-epoch validation mAP logged by the Ultralytics trainer (its own evaluator)."""
    found = list((runs / name / f"fold{fold}").rglob("results.csv"))
    if not found:
        return pd.DataFrame()
    df = pd.read_csv(found[0])
    df.columns = df.columns.str.strip()
    return df[["epoch", "metrics/mAP50-95(B)", "metrics/mAP50(B)"]].rename(
        columns={"metrics/mAP50-95(B)": "mAP", "metrics/mAP50(B)": "AP50"}
    )


def plot_curves(curves: dict[str, pd.DataFrame], title: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 3.6))
    for i, (name, df) in enumerate(curves.items()):
        if len(df):
            ax.plot(df["epoch"], df["mAP"], color=PALETTE[i], label=name, marker="o", ms=4)
    ax.set_xlabel("epoch")
    ax.set_ylabel("validation mAP (trainer evaluator)")
    ax.set_title(title)
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def error_breakdown(tables: Tables, pred: pd.DataFrame, threshold: float = 0.3) -> pd.DataFrame:
    """Per-size recall and the kind of each false positive at one score threshold.

    A false positive is a duplicate if it overlaps a matched object of its class at IoU >= 0.5,
    a class confusion if it overlaps an object of another class at IoU >= 0.5, a localisation
    error if it overlaps any object at 0.1-0.5, and background otherwise.
    """
    shown = pred[pred["score"] >= threshold]
    by_image = dict(tuple(shown.groupby("image_id")))
    gt_all = tables.objects[tables.objects["image_id"].isin(pred["image_id"].unique())]
    recall, kinds = [], []
    for image_id, gt in gt_all.groupby("image_id"):
        p = by_image.get(image_id, shown.iloc[:0])
        gt_hit, pred_hit = match(gt, p)
        side = np.sqrt(gt["w"] * gt["h"]).to_numpy()
        recall.append(pd.DataFrame({"side": side, "hit": gt_hit}))
        fp = p[~pred_hit]
        if len(fp):
            iou = iou_matrix(
                fp[["x", "y", "w", "h"]].to_numpy(), gt[["x", "y", "w", "h"]].to_numpy()
            )
            same = fp["cls"].to_numpy()[:, None] == gt["cls"].to_numpy()[None, :]
            best_same = np.where(same, iou, 0).max(1)
            best_other = np.where(~same, iou, 0).max(1)
            kind = np.select(
                [best_same >= 0.5, best_other >= 0.5, iou.max(1) >= 0.1],
                ["duplicate", "class confusion", "localisation"],
                "background",
            )
            kinds.append(pd.Series(kind))
    r = pd.concat(recall)
    bins = pd.cut(r["side"], SIZE_EDGES, labels=SIZE_LABELS)
    sizes = r.groupby(bins, observed=False)["hit"].agg(["mean", "size"])
    sizes.columns = ["recall", "objects"]
    fp_kinds = pd.concat(kinds).value_counts() if kinds else pd.Series(dtype=int)
    return sizes.round(3), fp_kinds


def worst_frames(tables: Tables, pred: pd.DataFrame, n: int = 3, threshold: float = 0.3):
    """Frames with the most missed plus false-positive boxes, one frame per clip."""
    shown = pred[pred["score"] >= threshold]
    by_image = dict(tuple(shown.groupby("image_id")))
    scores = []
    for image_id, gt in tables.objects[tables.objects["image_id"].isin(pred["image_id"])].groupby(
        "image_id"
    ):
        gt_hit, pred_hit = match(gt, by_image.get(image_id, shown.iloc[:0]))
        scores.append((image_id, (~gt_hit).sum() + (~pred_hit).sum(), len(gt)))
    df = pd.DataFrame(scores, columns=["image_id", "errors", "objects"])
    df["clip"] = df["image_id"].str.rsplit("_", n=1).str[0]
    df = df.sort_values("errors", ascending=False).drop_duplicates("clip")
    return df.head(n)


def plot_size_recall(breakdowns: dict[str, pd.DataFrame]) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 3.6))
    width = 0.8 / max(1, len(breakdowns))
    for i, (name, sizes) in enumerate(breakdowns.items()):
        x = np.arange(len(sizes)) + i * width
        ax.bar(x, sizes["recall"], width=width * 0.92, color=PALETTE[i], label=name)
    ax.set_xticks(np.arange(len(sizes)) + width * (len(breakdowns) - 1) / 2, sizes.index)
    ax.set_xlabel("object side, px (original frame)")
    ax.set_ylabel("recall at score 0.3")
    ax.legend(frameon=False, fontsize=8)
    ax.tick_params(axis="x", colors=MUTED)
    fig.tight_layout()
    return fig


def recall_by_size(tables: Tables, boxes: pd.DataFrame, top_n: int | None = None, iou=0.5):
    """Share of objects covered by any box at IoU >= iou, by object size (class-agnostic).

    Used to compare the proposal stage (top-N proposals) with the final detections.
    """
    if top_n is not None:
        boxes = boxes.sort_values("score", ascending=False).groupby("image_id").head(top_n)
    by_image = dict(tuple(boxes.groupby("image_id")))
    rows = []
    gt_all = tables.objects[tables.objects["image_id"].isin(boxes["image_id"].unique())]
    for image_id, gt in gt_all.groupby("image_id"):
        b = by_image[image_id][["x", "y", "w", "h"]].to_numpy()
        g = gt[["x", "y", "w", "h"]].to_numpy()
        hit = iou_matrix(g, b).max(1) >= iou if len(b) else np.zeros(len(g), bool)
        rows.append(pd.DataFrame({"side": np.sqrt(g[:, 2] * g[:, 3]), "hit": hit}))
    r = pd.concat(rows)
    bins = pd.cut(r["side"], SIZE_EDGES, labels=SIZE_LABELS)
    out = r.groupby(bins, observed=False)["hit"].mean()
    out["all"] = r["hit"].mean()
    return out.round(3)


def density_table(tables: Tables, runs: Path, names: list[str], fold: int = 0) -> pd.DataFrame:
    """mAP and AP_small on scored frames split by the number of annotated objects."""
    from visdet.data.folds import load_folds
    from visdet.eval.coco import evaluate

    folds = load_folds(tables)
    ids = folds.loc[folds["scored"] & (folds["fold"] == fold), "image_id"]
    count = tables.objects.groupby("image_id").size().reindex(ids).fillna(0)
    groups = pd.cut(count, [-1, 30, 80, 1e9], labels=["<=30 objects", "31-80", ">80"])
    rows = {}
    for name in names:
        pred = pd.read_parquet(runs / name / f"fold{fold}" / "val_predictions.parquet")
        rows[name] = {
            g: evaluate(pred, tables, count.index[groups == g])["mAP"]
            for g in groups.cat.categories
        }
    table = pd.DataFrame(rows).T.round(4)
    table.columns.name = f"frames: {groups.value_counts().reindex(groups.cat.categories).tolist()}"
    return table
