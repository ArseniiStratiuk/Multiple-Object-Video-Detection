"""One experiment on one fold: train, predict the held-out scored frames, evaluate, save.

Artifacts per run in <runs>/<name>/fold<k>/: config.yaml, val_predictions.parquet (pixels of the
original frames), metrics.json (COCO metrics and inference time per image), plus framework logs.
"""

import json
import os
import time
from pathlib import Path

import pandas as pd
import yaml

from visdet import models, submit, track
from visdet.data.cache import load_index
from visdet.data.export import fold_frames, write_fold_dataset
from visdet.data.folds import load_folds
from visdet.data.index import load_tables
from visdet.eval.coco import evaluate
from visdet.paths import data_root, runs_root
from visdet.viz import gallery

EPOCH_VAL_FRAMES = 300


def to_original_pixels(pred: pd.DataFrame, index: pd.DataFrame) -> pd.DataFrame:
    scale = pred["image_id"].map(index.set_index("image_id")["scale"])
    return pred.assign(**{c: pred[c] * scale for c in ("x", "y", "w", "h")})


def run(cfg: dict, cache_root: Path, device: str, limit: int | None = None) -> dict:
    out = runs_root() / cfg["name"] / f"fold{cfg['fold']}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))

    tables = load_tables()
    folds = load_folds(tables)
    if limit:
        folds = folds[folds["clip"].isin(folds.groupby("fold")["clip"].first())]
    tag = "-smoke" if limit else ""
    variant = cfg.get("eval_variant", "eval")
    if int(os.environ.get("RANK", "0")) == 0:
        data_yaml = write_fold_dataset(folds, cfg["fold"], cfg["step"], cache_root, tag, variant)
    else:
        suffix = "" if variant == "eval" else f"-{variant}"
        name = f"fold{cfg['fold']}-step{cfg['step']}{tag}{suffix}"
        data_yaml = cache_root / "folds" / name / "data.yaml"
        while not data_yaml.exists():
            time.sleep(1)
    adapter = models.get(cfg["framework"])
    _, val_ids = fold_frames(folds, cfg["fold"], cfg["step"])
    index = load_index(cache_root, variant)
    images = index.set_index("image_id").loc[val_ids].reset_index()[["image_id", "path"]]
    curve = []

    def on_epoch_end(predict_fn, epoch: int) -> None:
        """Validation mAP on a fixed subset of scored frames after every epoch."""
        subset = images.sample(min(EPOCH_VAL_FRAMES, len(images)), random_state=0)
        m = evaluate(to_original_pixels(predict_fn(subset), index), tables, subset["image_id"])
        curve.append({"epoch": epoch + 1} | {k: m[k] for k in ("mAP", "AP50", "AP_small")})
        pd.DataFrame(curve).to_csv(out / "val_curve.csv", index=False)
        track.log({f"val_subset/{k}": v for k, v in curve[-1].items()})

    torchvision_run = cfg["framework"] == "torchvision"
    if torchvision_run:
        track.start(cfg, "train")
        weights = adapter.train(cfg, data_yaml, out, device, on_epoch_end=on_epoch_end)
    else:
        weights = adapter.train(cfg, data_yaml, out, device)
    if weights is None:  # non-zero DDP rank: rank 0 predicts and evaluates
        return {}
    if not torchvision_run:
        track.start(cfg, "eval")

    for view in cfg.get("views", []):
        predict_view(
            cfg,
            weights,
            view,
            test_images(),
            None,
            out / f"test_predictions_{view}.parquet",
            device,
        )
    if cfg["fold"] < 0:
        metrics = {"weights": str(weights)}
        (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
        track.finish()
        return metrics
    start = time.time()
    pred = adapter.predict(cfg, weights, images, device)
    ms_per_image = 1000 * (time.time() - start) / len(images)
    pred = to_original_pixels(pred, index)
    pred.to_parquet(out / "val_predictions.parquet", index=False)

    metrics = evaluate(pred, tables, val_ids) | {
        "ms_per_image": ms_per_image,
        "val_images": len(images),
        "weights": str(weights),
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    for view in cfg.get("views", []):
        predict_view(
            cfg, weights, view, images, index, out / f"val_predictions_{view}.parquet", device
        )
    track.summary({f"val/{k}": v for k, v in metrics.items() if isinstance(v, float)})
    _log_overlays(tables, pred, val_ids)
    track.finish()
    return metrics


def parse_view(view: str) -> dict:
    """ "1536" -> inference at long side 1536; "flip1536" -> the same on mirrored images."""
    flip = view.startswith("flip")
    return {"imgsz": int(view.removeprefix("flip")), "flip": flip}


def test_images() -> pd.DataFrame:
    frames = load_tables().frames
    test = frames[frames["split"] == "test"]
    return pd.DataFrame({"image_id": test["image_id"], "path": test["file"].map(_data_path)})


def predict_view(cfg, weights, view, images, index, path: Path, device: str) -> None:
    """Predict one TTA view and save it in original-frame pixels (test frames are unscaled)."""
    pred = models.get(cfg["framework"]).predict(cfg | parse_view(view), weights, images, device)
    if index is not None:
        pred = to_original_pixels(pred, index)
    pred.to_parquet(path, index=False)


def _log_overlays(tables, pred: pd.DataFrame, val_ids, n: int = 8) -> None:
    """Overlays of one scored frame per held-out clip: green hit, yellow miss, red FP."""
    ids = pd.Series(list(val_ids))
    picks = ids.groupby(ids.str.rsplit("_", n=1).str[0]).first().head(n)
    arrays = [gallery.render(tables, pred, i, threshold=0.3)[:, :, ::-1] for i in picks]
    track.images("val/predictions", arrays, list(picks))


def predict_test(run_dir: Path, device: str, weights: Path | None = None) -> Path:
    """Predict the test frames at their original size and write a checked submission file."""
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
    if weights is None:
        weights = Path(json.loads((run_dir / "metrics.json").read_text())["weights"])
    tables = load_tables()
    test = tables.frames[tables.frames["split"] == "test"]
    images = pd.DataFrame({"image_id": test["image_id"], "path": test["file"].map(_data_path)})
    pred = models.get(cfg["framework"]).predict(cfg, weights, images, device)
    pred.to_parquet(run_dir / "test_predictions.parquet", index=False)
    return submit.write(pred, test, run_dir / "submission.csv")


def _data_path(file: str) -> str:
    return str(data_root() / file)


def predict_val(
    run_dir: Path, cache_root: Path, variant: str, views: list[str], device: str,
    weights: Path | None = None,
) -> list[Path]:  # fmt: skip
    """Re-predict a finished fold run on another validation variant (e.g. the degraded evalb).

    Files are named <variant>_predictions_<view>.parquet, in original-frame pixels.
    """
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
    if weights is None:
        weights = Path(json.loads((run_dir / "metrics.json").read_text())["weights"])
    tables = load_tables()
    folds = load_folds(tables)
    _, val_ids = fold_frames(folds, cfg["fold"], cfg["step"])
    index = load_index(cache_root, variant)
    images = index.set_index("image_id").loc[val_ids].reset_index()[["image_id", "path"]]
    paths = []
    for view in views:
        path = run_dir / f"{variant}_predictions_{view}.parquet"
        predict_view(cfg, weights, view, images, index, path, device)
        paths.append(path)
    return paths
