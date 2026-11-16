"""Locations of the competition data, derived tables and run artifacts.

Each machine sets VISDET_DATA (the unzipped competition root that holds train/, val/, test/)
and optionally VISDET_RUNS. Derived tables live next to the data so every machine builds its own.
"""

import os
from pathlib import Path


def data_root() -> Path:
    return Path(os.environ.get("VISDET_DATA", "data")).expanduser().resolve()


def tables_dir() -> Path:
    """Derived tables; VISDET_TABLES overrides when the data folder is read-only."""
    return Path(os.environ.get("VISDET_TABLES", data_root() / "tables")).expanduser().resolve()


def runs_root() -> Path:
    return Path(os.environ.get("VISDET_RUNS", "runs")).expanduser().resolve()
