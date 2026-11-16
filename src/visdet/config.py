"""Experiment configs: a YAML file that names a base file and overrides some of its keys."""

from pathlib import Path

import yaml


def load(path: Path, **overrides) -> dict:
    path = Path(path)
    cfg = yaml.safe_load(path.read_text())
    base = cfg.pop("base", None)
    if str(cfg.get("model", "")).endswith(".yaml"):
        cfg["model"] = str((path.parent / cfg["model"]).resolve())
    merged = (load(path.parent / base) if base else {}) | cfg
    return merged | {k: v for k, v in overrides.items() if v is not None}
