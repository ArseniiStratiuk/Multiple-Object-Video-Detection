"""Out-of-fold evaluation and test submission of a multi-model, multi-view ensemble.

A spec (YAML) lists member runs, the TTA views of each, optional model weights and temporal
rescoring. All member views are fused in one WBF step. Validation uses each fold model on its
own held-out fold, so the out-of-fold score covers all 63 clips once.

Example spec:
    name: ens-a
    members:
      - {run: rtdetr-x-final, views: [base, "1536", flip1536], weight: 2}
      - {run: yolo11s-final, views: [base, "1536"]}
    folds: [0, 1, 2, 3, 4]
    fusion: {conf_type: max}  # optional WBF arguments (visdet.infer.fusion.fuse)
    tracking: {alpha: 0.25}   # optional BoT-SORT track rescoring (visdet.infer.tracking);
                              # add panels: true to track split-screen panels separately
                              # (layout table tables_dir()/test_panels.csv, `visdet layout`)

A member may set test_folds: [-1] to predict the test set with its full-data model while its
validation still comes from the fold models.
"""

import json
from pathlib import Path

import pandas as pd
import yaml

from visdet import submit
from visdet.data.folds import load_folds
from visdet.data.index import load_tables
from visdet.eval.coco import evaluate
from visdet.infer import fusion, tracking
from visdet.paths import runs_root, tables_dir

MIN_SCORE = 0.01  # boxes below this barely move AP and slow the fusion down


def _load(run_dir: Path, split: str, view: str) -> pd.DataFrame:
    name = (
        f"{split}_predictions.parquet" if view == "base" else f"{split}_predictions_{view}.parquet"
    )  # split is "test", "val" (eval cache) or a variant name such as "evalb"
    pred = pd.read_parquet(run_dir / name)
    return pred[pred["score"] >= MIN_SCORE]


def _fuse(sets: list[pd.DataFrame], tables, weights=None, **fuse_args) -> pd.DataFrame:
    if len(sets) == 1 and not fuse_args.get("nms_per_set"):
        return sets[0]
    return fusion.fuse(sets, tables, weights=weights, **fuse_args)


def _member_views(
    spec_member: dict, split: str, folds: list[int], tables, fuse_args: dict
) -> list[pd.DataFrame]:
    """One prediction set per view of a member.

    Validation folds hold different frames, so the fold predictions of a view are concatenated.
    On the test set the fold models of a view are fused first (a seed-like ensemble of one
    model and view), which keeps one set per view as on validation.
    """
    run = runs_root() / spec_member["run"]
    if split != "test":
        split = spec_member.get("val_split", split)  # members may store validation differently
    if split == "test":
        folds = spec_member.get("test_folds", folds)  # e.g. [-1]: the full-data model
    sets = []
    for view in spec_member["views"]:
        per_fold = [_load(run / f"fold{k}", split, view) for k in folds]
        if split == "test":
            sets.append(_fuse(per_fold, tables, **fuse_args))
        else:
            sets.append(pd.concat(per_fold, ignore_index=True))
    return sets


def _combine(spec: dict, split: str, tables) -> pd.DataFrame:
    """Flat WBF over every member view; a two-level fusion lost 0.01 mAP on fold 0."""
    sets, weights = [], []
    for member in spec["members"]:
        views = _member_views(member, split, spec["folds"], tables, spec.get("fusion", {}))
        sets += views
        weights += [member.get("weight", 1)] * len(views)
    pred = _fuse(sets, tables, weights=weights, **spec.get("fusion", {}))
    if spec.get("tracking") is not None:
        args = dict(spec["tracking"])
        if args.get("panels"):  # split-screen layout per segment, from the layout detector
            args["panels"] = pd.read_csv(tables_dir() / "test_panels.csv")
        pred = tracking.rescore(pred, tables.frames, **args)
    return pred


def run(spec_path: Path, test: bool = True) -> dict:
    spec = yaml.safe_load(Path(spec_path).read_text())
    out = runs_root() / "ensembles" / spec["name"]
    out.mkdir(parents=True, exist_ok=True)
    (out / "spec.yaml").write_text(yaml.safe_dump(spec, sort_keys=False))
    tables = load_tables()
    folds = load_folds(tables)
    scored = folds.loc[folds["scored"] & folds["fold"].isin(spec["folds"]), "image_id"]
    oof = _combine(spec, spec.get("val_split", "val"), tables)
    metrics = evaluate(oof, tables, scored)
    per_fold = {
        f"mAP_fold{k}": evaluate(
            oof, tables, folds.loc[folds["scored"] & (folds["fold"] == k), "image_id"]
        )["mAP"]
        for k in spec["folds"]
    }
    metrics |= per_fold
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    if test:
        pred = _combine(spec, "test", tables)
        test_frames = tables.frames[tables.frames["split"] == "test"]
        submit.write(pred, test_frames, out / "submission.csv")
    return metrics
