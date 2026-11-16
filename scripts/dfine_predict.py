"""Predictions of one D-FINE fold model, in original frame pixels, as an ensemble member run.

Writes val_predictions.parquet (the fold's scored windows of the blur-matched "evalb" cache) and
test_predictions_1280.parquet. With --sahi, every image is predicted as the full frame plus SAHI
slices (visdet.infer.crops) and merged by class-wise NMS; the unmerged boxes are kept in
*_sahi_raw.parquet with their tile index.

Usage (D-FINE checkout in VISDET_DFINE, cache root in --cache):
    python scripts/dfine_predict.py <config> <checkpoint> <out_dir> --fold K --cache <root> [--sahi]
"""

import argparse
from pathlib import Path

import pandas as pd

from visdet.data.cache import load_index
from visdet.data.folds import load_folds
from visdet.data.index import load_tables
from visdet.infer import crops
from visdet.models.dfine import predict
from visdet.run import test_images, to_original_pixels

parser = argparse.ArgumentParser()
parser.add_argument("config", type=Path)
parser.add_argument("checkpoint", type=Path)
parser.add_argument("out", type=Path)
parser.add_argument("--fold", type=int, required=True)
parser.add_argument("--cache", type=Path, required=True)
parser.add_argument("--weights", default="ema", choices=["ema", "model"])
parser.add_argument("--sahi", action="store_true")
args = parser.parse_args()
args.out.mkdir(parents=True, exist_ok=True)


def model(images: pd.DataFrame) -> pd.DataFrame:
    return predict(args.config, args.checkpoint, images, weights=args.weights)


def run(images: pd.DataFrame, split: str) -> pd.DataFrame:
    if not args.sahi:
        return model(images)
    raw = crops.predict(model, crops.sahi_tiles(images))
    raw.to_parquet(args.out / f"{split}_sahi_raw.parquet", index=False)
    return crops.merge(raw)


tables = load_tables()
folds = load_folds(tables)
scored = folds.loc[(folds["fold"] == args.fold) & folds["scored"], "image_id"]
index = load_index(args.cache, "evalb").set_index("image_id").loc[scored].reset_index()
val = to_original_pixels(run(index, "val"), index)
val.to_parquet(args.out / "val_predictions.parquet", index=False)
run(test_images(), "test").to_parquet(args.out / "test_predictions_1280.parquet", index=False)
