"""Load config.yaml and resolve its paths."""

from __future__ import annotations

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str | Path | None = None) -> dict:
    """Read the YAML config. Paths under `paths:` are resolved relative to the
    config file's own directory, so the repo works from any working directory."""
    path = Path(path) if path else REPO_ROOT / "config.yaml"
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    base = path.resolve().parent
    cfg["paths"] = {key: (base / value).resolve() for key, value in cfg["paths"].items()}
    return cfg


def dataset_dir(cfg: dict, dataset: str) -> Path:
    return cfg["paths"]["data_root"] / dataset


def features_dir(cfg: dict, dataset: str, limit: int | None = None) -> Path:
    """Where cached features for a dataset live. Subset runs get their own
    directory so a debug run can never be mistaken for the full extraction."""
    name = dataset if limit is None else f"{dataset}_n{limit}"
    return cfg["paths"]["artifacts"] / "features" / name


def results_dir(cfg: dict, variable: str) -> Path:
    return cfg["paths"]["artifacts"] / "results" / variable
