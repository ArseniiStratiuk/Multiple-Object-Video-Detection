"""Prediction on image crops, mapped back to frame pixels: SAHI slices and split-screen panels.

A crop table has one row per crop: image_id, path, and the crop box x0, y0, x1, y1 in frame
pixels. The model families crop the image when these columns are present and return boxes in
crop pixels; `predict` shifts them back by the crop origin.

SAHI (Akyon et al. 2022): the full frame plus 640x640 slices with 25% overlap
(sahi.slicing.get_slice_bboxes), each stretched to the model input, merged by class-wise NMS.
Panels: every frame of a segment that the layout detector marks as a split screen
(visdet.infer.layout) is cut into its rows x cols panels, so each panel is seen as one video.
"""

from collections.abc import Callable

import pandas as pd
import torch
from PIL import Image
from torchvision.ops import batched_nms

BOX = ["x0", "y0", "x1", "y1"]
COLUMNS = ["image_id", "cls", "score", "x", "y", "w", "h"]


def sahi_tiles(images: pd.DataFrame, size: int = 640, overlap: float = 0.25) -> pd.DataFrame:
    """Crop table of the full frame (tile 0) and the SAHI slices of each image."""
    from sahi.slicing import get_slice_bboxes

    rows = []
    for image_id, path in images[["image_id", "path"]].to_numpy():
        w, h = Image.open(path).size
        boxes = [[0, 0, w, h], *get_slice_bboxes(h, w, size, size, overlap, overlap)]
        rows += [(image_id, path, t, *b) for t, b in enumerate(boxes)]
    return pd.DataFrame(rows, columns=["image_id", "path", "tile", *BOX])


def split_frames(frames: pd.DataFrame, panels: pd.DataFrame) -> pd.DataFrame:
    """Frames (rows of Tables.frames) inside split-screen segments, with their rows and cols."""
    parts = []
    for p in panels[(panels["rows"] > 1) | (panels["cols"] > 1)].itertuples():
        f = frames[(frames["clip"] == p.clip) & frames["frame"].between(p.first, p.last)]
        parts.append(f.assign(rows=p.rows, cols=p.cols))
    return pd.concat(parts, ignore_index=True)


def panel_tiles(images: pd.DataFrame, split: pd.DataFrame) -> pd.DataFrame:
    """Crop table of the panels of each split-screen frame (`split` from split_frames)."""
    rows = []
    layout = split.set_index("image_id")[["rows", "cols", "width", "height"]]
    images = images[images["image_id"].isin(layout.index)]
    for image_id, path in images[["image_id", "path"]].to_numpy():
        n_rows, n_cols, w, h = layout.loc[image_id]
        ph, pw = h // n_rows, w // n_cols
        for i in range(n_rows):
            for j in range(n_cols):
                tile = i * n_cols + j
                rows.append((image_id, path, tile, j * pw, i * ph, (j + 1) * pw, (i + 1) * ph))
    return pd.DataFrame(rows, columns=["image_id", "path", "tile", *BOX])


def predict(
    predict_fn: Callable[[pd.DataFrame], pd.DataFrame], tiles: pd.DataFrame, min_score: float = 0.01
) -> pd.DataFrame:
    """Predict every crop and return boxes in frame pixels with a `tile` column."""
    keyed = tiles.assign(image_id=tiles["image_id"] + "|" + tiles["tile"].astype(str))
    pred = predict_fn(keyed)
    pred = pred[pred["score"] >= min_score]
    origin = keyed.set_index("image_id")[["x0", "y0"]]
    pred = pred.assign(x=pred["x"] + pred["image_id"].map(origin["x0"]))
    pred = pred.assign(y=pred["y"] + pred["image_id"].map(origin["y0"]))
    image_id, tile = pred["image_id"].str.split("|", expand=True).T.values
    return pred.assign(image_id=image_id, tile=tile.astype(int)).reset_index(drop=True)


def merge(pred: pd.DataFrame, iou: float = 0.5) -> pd.DataFrame:
    """Class-wise NMS per image over all tiles (SAHI's NMS postprocess)."""
    keep = []
    for _, g in pred.groupby("image_id", sort=False):
        xyxy = torch.tensor(g[["x", "y", "w", "h"]].to_numpy(), dtype=torch.float32)
        xyxy[:, 2:] += xyxy[:, :2]
        score = torch.tensor(g["score"].to_numpy(), dtype=torch.float32)
        k = batched_nms(xyxy, score, torch.tensor(g["cls"].to_numpy()), iou)
        keep.append(g.iloc[k.numpy()])
    return pd.concat(keep, ignore_index=True)[COLUMNS]
