"""Replace the test predictions of split-screen frames by per-panel predictions, for every fold
model of a run (D-FINE or Ultralytics). Each panel crop is predicted at the full model input size
and mapped back; other frames and the validation files are taken from the source run.

Usage:
    python scripts/panel_predict.py ultralytics <run> <out_run>
    python scripts/panel_predict.py dfine <run> <out_run> --config "<cfg with {k}>" \
        --checkpoint "<checkpoint with {k}>"
The panel table defaults to tables_dir()/test_panels.csv (`visdet layout`).
"""

import argparse
import shutil

import pandas as pd
import yaml

from visdet.data.index import load_tables
from visdet.infer import crops
from visdet.paths import runs_root, tables_dir
from visdet.run import test_images

parser = argparse.ArgumentParser()
parser.add_argument("family", choices=["dfine", "ultralytics"])
parser.add_argument("run")
parser.add_argument("out_run")
parser.add_argument("--panels", default=None)
parser.add_argument("--config")
parser.add_argument("--checkpoint")
parser.add_argument("--folds", default="0,1,2,3,4")
args = parser.parse_args()

tables = load_tables()
panels = pd.read_csv(args.panels or tables_dir() / "test_panels.csv")
split = crops.split_frames(tables.frames[tables.frames["split"] == "test"], panels)
tiles = crops.panel_tiles(test_images(), split)
print("split-screen frames", len(split), "panels", len(tiles), flush=True)
for k in map(int, args.folds.split(",")):
    src, out = runs_root() / args.run / f"fold{k}", runs_root() / args.out_run / f"fold{k}"
    out.mkdir(parents=True, exist_ok=True)
    if args.family == "dfine":
        from visdet.models.dfine import predict

        cfg, ckpt = args.config.format(k=k), args.checkpoint.format(k=k)

        def model(images, cfg=cfg, ckpt=ckpt):
            return predict(cfg, ckpt, images, weights="ema")
    else:
        from visdet.models.ultralytics import predict

        cfg = yaml.safe_load((src / "config.yaml").read_text())
        weights = next(src.glob(f"runs/detect/*/{args.run}-fold{k}/weights/best.pt"))

        def model(images, cfg=cfg, weights=weights):
            return predict(cfg, weights, images, "0")

    per_panel = crops.predict(model, tiles, min_score=0)[crops.COLUMNS]
    base = pd.read_parquet(src / "test_predictions_1280.parquet")
    keep = base[~base["image_id"].isin(split["image_id"])]
    pred = pd.concat([keep, per_panel], ignore_index=True)
    pred.to_parquet(out / "test_predictions_1280.parquet", index=False)
    for f in src.glob("val_predictions*.parquet"):
        shutil.copy(f, out / f.name)
    print("fold", k, len(base), "->", len(pred), flush=True)
