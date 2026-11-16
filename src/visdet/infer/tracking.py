"""Temporal rescoring with a tracker: link detections across frames, then rescore by track.

1. Segments. A clip is split where frames are not consecutive (scored validation windows) and at
   scene cuts found by PySceneDetect's AdaptiveDetector (several test clips join two videos).
2. Tracks. Ultralytics BoT-SORT (Aharon et al. 2022) links the detections of each segment, with
   its global motion compensation estimating camera movement between frames. It tracks objects
   regardless of class; detections below its low threshold stay unlinked.
   With a panel table (split-screen layout per segment, see the layout detector), each panel of a
   split frame is tracked separately on its own crop, so motion compensation sees one camera.
3. Rescoring. Within a track, only detections of the same class are pooled. Each linked score
   becomes alpha * own score + (1 - alpha) * mean of the top_k scores of its class in the track,
   so a weak detection of an object seen confidently in nearby frames is raised and a one-frame
   blip is lowered (the Seq-NMS idea, Han et al. 2016). Unlinked detections keep their score.

Predictions are a DataFrame with image_id (<clip>_<frame>), cls, score, x, y, w, h in pixels of
the original frame; `frames` maps image_id to the frame file (Tables.frames).
"""

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from visdet.paths import data_root


class _Detections:
    """The minimal Boxes interface that the Ultralytics trackers read."""

    def __init__(self, xywh: np.ndarray, conf: np.ndarray, cls: np.ndarray):
        self.xywh, self.conf, self.cls = xywh, conf, cls

    def __len__(self) -> int:
        return len(self.conf)

    def __getitem__(self, mask):
        return _Detections(self.xywh[mask], self.conf[mask], self.cls[mask])

    @property
    def xyxy(self) -> np.ndarray:
        xyxy = self.xywh.copy()
        xyxy[:, :2] -= xyxy[:, 2:] / 2
        xyxy[:, 2:] += xyxy[:, :2]
        return xyxy


def _split_id(image_ids: pd.Series) -> tuple[pd.Series, pd.Series]:
    parts = image_ids.str.rsplit("_", n=1, expand=True)
    return parts[0], parts[1].astype(int)


def scene_starts(clip_files: list[Path]) -> set[int]:
    """Positions (0-based, in clip_files order) where PySceneDetect starts a new scene."""
    from scenedetect import AdaptiveDetector, detect

    pattern = str(clip_files[0].parent / f"%0{len(clip_files[0].stem)}d{clip_files[0].suffix}")
    return {start.get_frames() for start, _ in detect(pattern, AdaptiveDetector())[1:]}


def segments(pred: pd.DataFrame, frames: pd.DataFrame) -> pd.Series:
    """Segment label per prediction row: consecutive frames of one clip and one scene."""
    clip, number = _split_id(pred["image_id"])
    label = pd.Series("", index=pred.index)
    for name, rows in pred.groupby(clip):
        files = frames[frames["clip"] == name].sort_values("frame")
        paths = [data_root() / f for f in files["file"]]
        position = dict(zip(files["frame"], range(len(files)), strict=True))
        cuts = scene_starts(paths)
        present = np.sort(number[rows.index].unique())
        seg, out = 0, {}
        for i, f in enumerate(present):
            if i and (f != present[i - 1] + 1 or position[f] in cuts):
                seg += 1
            out[f] = f"{name}/{seg}"
        label[rows.index] = number[rows.index].map(out)
    return label


def panel_of(pred: pd.DataFrame, panels: pd.DataFrame | None, size: tuple[int, int]) -> pd.Series:
    """Panel (row, col, rows, cols) of each prediction by box centre; (0, 0, 1, 1) = whole."""
    out = pd.Series([(0, 0, 1, 1)] * len(pred), index=pred.index, dtype=object)
    if panels is None or pred.empty:
        return out
    clip, number = _split_id(pred["image_id"])
    w, h = size
    for p in panels[(panels["rows"] > 1) | (panels["cols"] > 1)].itertuples():
        m = (clip == p.clip) & number.between(p.first, p.last)
        cx = pred.loc[m, "x"] + pred.loc[m, "w"] / 2
        cy = pred.loc[m, "y"] + pred.loc[m, "h"] / 2
        col = np.clip((cx * p.cols // w).astype(int), 0, p.cols - 1)
        row = np.clip((cy * p.rows // h).astype(int), 0, p.rows - 1)
        out[m] = [(r, c, p.rows, p.cols) for r, c in zip(row, col, strict=True)]
    return out


def link(
    pred: pd.DataFrame, frames: pd.DataFrame, tracker: dict | None = None,
    panels: pd.DataFrame | None = None,
) -> pd.Series:  # fmt: skip
    """Track id per prediction row (-1 when BoT-SORT did not link it)."""
    from ultralytics.utils import ROOT, YAML

    args = YAML.load(ROOT / "cfg/trackers/botsort.yaml") | (tracker or {})
    frames = frames.reset_index() if "image_id" not in frames else frames
    file_of = frames.set_index("image_id")["file"]
    size_of = frames.set_index("image_id")[["width", "height"]]
    seg = segments(pred, frames)
    _, number = _split_id(pred["image_id"])
    track = pd.Series(-1, index=pred.index)
    offset = 0
    for _, seg_rows in pred.groupby(seg):
        w, h = size_of.loc[seg_rows["image_id"].iloc[0]]
        panel = panel_of(seg_rows, panels, (int(w), int(h)))
        for (r, c, nr, nc), rows in seg_rows.groupby(panel):
            x0, x1 = round(c * w / nc), round((c + 1) * w / nc)
            y0, y1 = round(r * h / nr), round((r + 1) * h / nr)
            offset = _track_panel(rows, number, file_of, args, track, offset, (x0, y0, x1, y1))
    return track


def _track_panel(rows, number, file_of, args, track, offset, box) -> int:
    """BoT-SORT over the frames of one segment panel; writes ids into track, returns next offset."""
    from ultralytics.trackers.bot_sort import BOTSORT
    from ultralytics.utils import IterableSimpleNamespace

    x0, y0, x1, y1 = box
    bot = BOTSORT(IterableSimpleNamespace(**args))
    for _, det in rows.groupby(number[rows.index], sort=True):
        xywh = det[["x", "y", "w", "h"]].to_numpy(np.float32).copy()
        xywh[:, :2] += xywh[:, 2:] / 2 - np.array([x0, y0], np.float32)
        image = cv2.imread(str(data_root() / file_of[det["image_id"].iloc[0]]))[y0:y1, x0:x1]
        found = bot.update(
            _Detections(xywh, det["score"].to_numpy(np.float32), det["cls"].to_numpy(np.float32)),
            image,
        )
        if len(found):
            track[det.index[found[:, 7].astype(int)]] = found[:, 4].astype(int) + offset
    return int(track.max()) + 1


def rescore(
    pred: pd.DataFrame, frames: pd.DataFrame, alpha: float = 0.25, top_k: int = 5,
    tracker: dict | None = None, panels: pd.DataFrame | None = None,
) -> pd.DataFrame:  # fmt: skip
    """Blend each linked score with the mean top-k score of its class within its track."""
    track = link(pred, frames, tracker, panels)
    linked = track >= 0
    group = [track[linked], pred.loc[linked, "cls"]]
    top = pred.loc[linked, "score"].groupby(group).transform(lambda s: s.nlargest(top_k).mean())
    out = pred.copy()
    out.loc[linked, "score"] = alpha * pred.loc[linked, "score"] + (1 - alpha) * top
    return out
