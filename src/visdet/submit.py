"""Kaggle submission file: one row per test image, "class score x y w h" groups in pixels."""

from pathlib import Path

import pandas as pd

from visdet.eval.coco import MAX_DETS, PRED_COLUMNS, top_per_image


def to_rows(pred: pd.DataFrame, image_ids) -> pd.DataFrame:
    pred = top_per_image(pred)
    text = (
        pred["cls"].astype(int).astype(str)
        + " "
        + pred["score"].map("{:.4f}".format)
        + " "
        + pred[["x", "y", "w", "h"]].round(1).astype(str).agg(" ".join, axis=1)
    )
    joined = text.groupby(pred["image_id"]).agg(" ".join)
    ids = pd.Index(image_ids, name="image_id")
    return pd.DataFrame(
        {"image_id": ids, "PredictionString": joined.reindex(ids).fillna("").values}
    )


def parse_rows(rows: pd.DataFrame) -> pd.DataFrame:
    records = []
    for image, text in zip(rows["image_id"], rows["PredictionString"].fillna(""), strict=True):
        values = text.split()
        for i in range(0, len(values), 6):
            c, s, x, y, w, h = values[i : i + 6]
            records.append((image, int(c), float(s), float(x), float(y), float(w), float(h)))
    return pd.DataFrame(records, columns=PRED_COLUMNS)


def validate(rows: pd.DataFrame, test_frames: pd.DataFrame) -> None:
    """Fail on any mismatch with the expected test image list or the scored format."""
    if list(rows.columns) != ["image_id", "PredictionString"]:
        raise ValueError(f"unexpected columns {list(rows.columns)}")
    if set(rows["image_id"]) != set(test_frames["image_id"]) or rows["image_id"].duplicated().any():
        raise ValueError("image ids differ from test_images.csv")
    pred = parse_rows(rows)
    if not pred["cls"].between(0, 7).all() or not pred["score"].between(0, 1).all():
        raise ValueError("class or score out of range")
    if (pred.groupby("image_id").size() > MAX_DETS[-1]).any():
        raise ValueError("more than 500 detections in an image")
    size = test_frames.set_index("image_id").loc[pred["image_id"], ["width", "height"]].to_numpy()
    if (pred["x"].to_numpy() > size[:, 0]).any() or (pred["y"].to_numpy() > size[:, 1]).any():
        raise ValueError("box origin outside the image")


def clip_to_frame(pred: pd.DataFrame, frames: pd.DataFrame) -> pd.DataFrame:
    """Cut boxes that extend past the frame edge back to the visible part."""
    size = frames.set_index("image_id").loc[pred["image_id"], ["width", "height"]].to_numpy()
    x1, y1 = pred["x"].clip(lower=0), pred["y"].clip(lower=0)
    x2 = (pred["x"] + pred["w"]).clip(upper=size[:, 0])
    y2 = (pred["y"] + pred["h"]).clip(upper=size[:, 1])
    out = pred.assign(x=x1, y=y1, w=x2 - x1, h=y2 - y1)
    return out[(out["w"] > 0) & (out["h"] > 0)]


def write(pred: pd.DataFrame, test_frames: pd.DataFrame, path: Path) -> Path:
    rows = to_rows(clip_to_frame(pred, test_frames), test_frames["image_id"])
    validate(rows, test_frames)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows.to_csv(path, index=False)
    return path
