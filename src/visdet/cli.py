"""Command line entry point: `visdet <command>`."""

import argparse
import json
import sys
from pathlib import Path

from visdet import checks
from visdet.data import cache, export
from visdet.data.folds import N_FOLDS, build_folds, load_folds
from visdet.data.index import build_tables, load_tables


def main() -> None:
    parser = argparse.ArgumentParser(prog="visdet")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("index", help="build frame, object, ignore and fold tables")
    commands.add_parser("check", help="run the evaluation and split sanity checks")
    commands.add_parser("layout", help="train the split-screen detector, write test_panels.csv")
    c = commands.add_parser("cache", help="write resized images, labels and fold datasets")
    c.add_argument("--root", type=Path, required=True)
    c.add_argument("--step", type=int, default=5)
    c.add_argument("--variants", default="train,eval,evalb")
    r = commands.add_parser("run", help="train and evaluate one config on one fold")
    r.add_argument("config", type=Path)
    r.add_argument("--cache", type=Path, required=True)
    r.add_argument("--device", default="0")
    r.add_argument("--fold", type=int)
    r.add_argument("--epochs", type=int)
    r.add_argument("--name")
    r.add_argument("--workers", type=int)
    r.add_argument("--step", type=int)
    r.add_argument("--smoke", action="store_true", help="one clip per fold, for pipeline checks")
    t = commands.add_parser("predict-test", help="write a submission from a finished run")
    t.add_argument("run_dir", type=Path)
    t.add_argument("--device", default="0")
    t.add_argument("--weights", type=Path, help="local copy when trained on another machine")
    v = commands.add_parser("predict-val", help="re-predict a fold run on a validation variant")
    v.add_argument("run_dir", type=Path)
    v.add_argument("--cache", type=Path, required=True)
    v.add_argument("--variant", default="evalb")
    v.add_argument("--views", default="1280")
    v.add_argument("--device", default="0")
    v.add_argument("--weights", type=Path)
    e = commands.add_parser("ensemble", help="out-of-fold score and submission of an ensemble")
    e.add_argument("spec", type=Path)
    e.add_argument("--no-test", action="store_true")
    g = commands.add_parser("gallery", help="overlay predictions of a run on sample frames")
    g.add_argument("run_dir", type=Path)
    g.add_argument("--frames", type=int, default=12)
    g.add_argument("--threshold", type=float, default=0.3)
    args = parser.parse_args()

    if args.command == "index":
        tables = build_tables()
        folds = build_folds(tables)
        print(
            f"frames {len(tables.frames)}, objects {len(tables.objects)}, "
            f"ignored regions {len(tables.ignore)}, scored frames {folds['scored'].sum()}"
        )
    elif args.command == "check":
        sys.exit(0 if checks.run() else 1)
    elif args.command == "cache":
        tables = load_tables()
        folds = load_folds(tables)
        variants = args.variants.split(",")
        train_ids = folds.loc[folds["position"] % args.step == 0, "image_id"]
        if "train" in variants:
            cache.build(tables, train_ids, "train", args.root)
        for variant in ("eval", "evalb"):
            if variant in variants:
                cache.build(tables, folds.loc[folds["scored"], "image_id"], variant, args.root)
        for k in range(N_FOLDS):
            print(export.write_fold_dataset(folds, k, args.step, args.root))
    elif args.command == "run":
        from visdet import config, run

        cfg = config.load(
            args.config,
            fold=args.fold,
            epochs=args.epochs,
            name=args.name,
            workers=args.workers,
            step=args.step,
        )
        metrics = run.run(cfg, args.cache, args.device, limit=1 if args.smoke else None)
        print(json.dumps(metrics, indent=2))
    elif args.command == "predict-test":
        from visdet import run

        print(run.predict_test(args.run_dir, args.device, args.weights))
    elif args.command == "predict-val":
        from visdet import run

        views = args.views.split(",")
        print(
            run.predict_val(
                args.run_dir, args.cache, args.variant, views, args.device, args.weights
            )
        )
    elif args.command == "layout":
        from visdet.infer import layout
        from visdet.paths import data_root, tables_dir

        frames = load_tables().frames
        clf = layout.train(frames, data_root())
        held_out = [s for seed in (4, 5) for s in layout.synthetic(frames, data_root(), 30, seed)]
        wrong, total = layout.frame_errors(clf, held_out)
        print(f"held-out synthetic frames wrong: {wrong} / {total}")
        panels = layout.detect_clips(clf, frames[frames["split"] == "test"], data_root())
        panels.to_csv(tables_dir() / "test_panels.csv", index=False)
        print(panels[(panels["rows"] > 1) | (panels["cols"] > 1)].to_string())
    elif args.command == "ensemble":
        from visdet import ensemble

        print(json.dumps(ensemble.run(args.spec, test=not args.no_test), indent=2))
    elif args.command == "gallery":
        import pandas as pd

        from visdet.viz import gallery

        pred = pd.read_parquet(args.run_dir / "val_predictions.parquet")
        ids = pd.Series(pred["image_id"].unique())
        ids = ids.groupby(ids.str.rsplit("_", n=1).str[0]).first()
        ids = ids.sample(min(args.frames, len(ids)), random_state=0)
        gallery.write(load_tables(), pred, ids, args.run_dir / "gallery", args.threshold)


if __name__ == "__main__":
    main()
