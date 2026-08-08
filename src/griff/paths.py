"""Filesystem anchors.

Assets travel with the package so `griff` works from an installed wheel.
Everything else is resolved relative to the repository, so a `pip install -e .`
checkout and a plain `python -m griff.cli` both find the same files.

Two of those locations can be redirected by environment variable:

    GRIFF_DATASETS   where recorded episodes and their decoded image caches live
    GRIFF_RUNS       where training checkpoints and logs are written

Both hold large, regenerable artefacts -- a 60-episode dataset is tens of
thousands of PNGs plus a few hundred megabytes of cache -- and both are the
things you want *off* a cloud-synced folder. Leaving them inside one is not a
tidiness problem: on the machine this was developed on, OneDrive indexing the
recorded frames took more CPU than training did, and a run that took 7 minutes
alone took over an hour with the sync service competing for cores.

`results/`, `calibration/` and `datasets/demo-*` are small and committed, and
stay in the repository regardless.
"""

from __future__ import annotations

import os
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
ASSETS = PACKAGE_ROOT / "assets"

# src/griff/ -> src/ -> repo root
REPO_ROOT = PACKAGE_ROOT.parent.parent


def _override(variable: str, default: Path) -> Path:
    value = os.environ.get(variable)
    return Path(value).expanduser().resolve() if value else default


CONFIGS = REPO_ROOT / "configs"
DATASETS = _override("GRIFF_DATASETS", REPO_ROOT / "datasets")
RUNS = _override("GRIFF_RUNS", REPO_ROOT / "runs")
RESULTS = REPO_ROOT / "results"
CALIBRATION = REPO_ROOT / "calibration"


def scene(task: str) -> Path:
    """Path to the MJCF scene for a task name."""
    path = ASSETS / f"task_{task}.xml"
    if not path.exists():
        available = sorted(p.stem.removeprefix("task_") for p in ASSETS.glob("task_*.xml"))
        raise FileNotFoundError(f"no scene for task {task!r}; available: {available}")
    return path
