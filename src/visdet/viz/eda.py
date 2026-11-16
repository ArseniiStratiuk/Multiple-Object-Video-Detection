"""Dataset statistics and figures used by the data notebook."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from visdet import CLASSES
from visdet.data.index import Tables
from visdet.viz.style import MUTED, PALETTE

COCO_SMALL, COCO_MEDIUM = 32, 96


def objects_with_frames(tables: Tables) -> pd.DataFrame:
    frames = tables.frames[["image_id", "clip", "video", "width", "height"]]
    o = tables.objects.merge(frames, on="image_id")
    return o.assign(side=np.sqrt(o["w"] * o["h"]), long_side=np.maximum(o["width"], o["height"]))


def split_summary(tables: Tables) -> pd.DataFrame:
    f = tables.frames
    per_image = tables.objects.groupby("image_id").size()
    rows = {}
    for split, g in f.groupby("split"):
        n = g["image_id"].map(per_image).fillna(0)
        rows[split] = {
            "clips": g["clip"].nunique(),
            "source videos": g["video"].nunique(),
            "frames": len(g),
            "frames per clip (median)": int(g.groupby("clip").size().median()),
            "objects": int(n.sum()) if split != "test" else None,
            "objects per frame (median)": float(n.median()) if split != "test" else None,
        }
    return pd.DataFrame(rows).T


def plot_classes_and_sizes(o: pd.DataFrame) -> plt.Figure:
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 3.8))
    share = o["cls"].value_counts(normalize=True).reindex(range(len(CLASSES))).fillna(0)
    a.barh(CLASSES, share.values * 100, color=PALETTE[0], height=0.6)
    a.invert_yaxis()
    a.set_xlabel("share of objects, %")
    a.set_title("Class balance")
    bins = np.logspace(np.log10(4), np.log10(600), 60)
    b.hist(o["side"], bins=bins, color=PALETTE[0])
    b.set_xscale("log")
    for edge, name in ((COCO_SMALL, "small < 32"), (COCO_MEDIUM, "large >= 96")):
        b.axvline(edge, color=MUTED, lw=1, ls="--")
        b.text(edge * 1.05, b.get_ylim()[1] * 0.9, name, color=MUTED, fontsize=8)
    b.set_xlabel("object side sqrt(w*h), px at native resolution")
    b.set_title("Object size")
    fig.tight_layout()
    return fig


def size_table(o: pd.DataFrame) -> pd.DataFrame:
    g = o.groupby("cls")["side"]
    t = pd.DataFrame({"objects": g.size(), "median side": g.median(), "p10 side": g.quantile(0.1)})
    return t.set_axis(CLASSES).round(1)


def plot_clip_density(tables: Tables, folds: pd.DataFrame) -> plt.Figure:
    per_image = tables.objects.groupby("image_id").size()
    f = folds.assign(n=folds["image_id"].map(per_image).fillna(0))
    clips = (
        f.groupby("clip").agg(density=("n", "mean"), fold=("fold", "first")).sort_values("density")
    )
    fig, ax = plt.subplots(figsize=(12, 3.6))
    colors = [PALETTE[k] for k in clips["fold"]]
    ax.bar(range(len(clips)), clips["density"], color=colors, width=0.8)
    ax.set_xticks([])
    ax.set_xlabel("labelled clips, sorted by density")
    ax.set_ylabel("mean objects per frame")
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=PALETTE[k]) for k in range(folds["fold"].max() + 1)
    ]
    ax.legend(handles, [f"fold {k}" for k in range(len(handles))], ncol=5, frameon=False)
    ratio = clips["density"].max() / clips["density"].min()
    ax.set_title(f"Mean density varies {ratio:.0f}x between clips")
    fig.tight_layout()
    return fig


def resolution_shift(tables: Tables) -> pd.DataFrame:
    f = tables.frames.assign(long_side=np.maximum(tables.frames["width"], tables.frames["height"]))
    f["group"] = np.where(f["split"] == "test", "test", "labelled")
    bucket = pd.cut(f["long_side"], [0, 700, 1000, 1400, 2000, 4000],
                    labels=["680", "960", "1344-1360", "1904-1920", "2688-3840"])  # fmt: skip
    return (pd.crosstab(bucket, f["group"], normalize="columns") * 100).round(1)


def temporal_overlap(tables: Tables, steps=(1, 2, 5, 10, 20)) -> pd.DataFrame:
    """IoU of the same tracked object between frame t and t+step."""
    o = tables.objects.merge(tables.frames[["image_id", "clip", "frame"]], on="image_id")
    a = o[["clip", "track_id", "frame", "x", "y", "w", "h"]]
    rows = []
    for k in steps:
        m = a.merge(
            a.assign(frame=a["frame"] - k), on=["clip", "track_id", "frame"], suffixes=("", "_n")
        )
        ix = (np.minimum(m.x + m.w, m.x_n + m.w_n) - np.maximum(m.x, m.x_n)).clip(lower=0)
        iy = (np.minimum(m.y + m.h, m.y_n + m.h_n) - np.maximum(m.y, m.y_n)).clip(lower=0)
        iou = ix * iy / (m.w * m.h + m.w_n * m.h_n - ix * iy)
        rows.append({"step": k, "median IoU": iou.median(), "share IoU < 0.5": (iou < 0.5).mean()})
    return pd.DataFrame(rows).set_index("step").round(3)


def sharpness(path) -> float:
    """Variance of the Laplacian of the grey image: higher means more fine detail."""
    import cv2

    grey = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(grey, cv2.CV_64F).var())


def sharpness_table(tables: Tables, cache_root, per_bucket: int = 40) -> pd.DataFrame:
    """Mean sharpness of test frames vs labelled frames resized to the same long side.

    "resized" are the eval-cache frames (clean INTER_AREA downscale of labelled frames);
    "degraded" are the evalb frames after the blur and JPEG fitted to the test frames.
    """
    from pathlib import Path

    from visdet.paths import data_root

    root = Path(cache_root)
    test = tables.frames[tables.frames["split"] == "test"]
    test = test.assign(side=np.maximum(test["width"], test["height"]))
    rows = {}
    for bucket in (680, 960, 1360, 1920):
        t = test[(test["side"] - bucket).abs() <= 30]
        t = t.sample(min(per_bucket, len(t)), random_state=0)
        row = {"test": np.mean([sharpness(data_root() / f) for f in t["file"]])}
        for variant, name in (("eval", "labelled, resized"), ("evalb", "labelled, degraded")):
            index = pd.read_parquet(root / f"{variant}_index.parquet")
            index = index[(index["cached_width"] - bucket).abs() <= 30]
            index = index.sample(min(per_bucket, len(index)), random_state=0)
            row[name] = np.mean([sharpness(p) for p in index["path"]])
        rows[bucket] = row
    out = pd.DataFrame(rows).T.round(0)
    out.index.name = "long side, px"
    return out
