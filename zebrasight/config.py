"""Config loading. Paths resolve from the repo root with pathlib, so this works on any OS."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from omegaconf import DictConfig, OmegaConf

from zebrasight.types import ViewCond

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = REPO_ROOT / "configs"


def load_config(name: str) -> DictConfig:
    """Load ``configs/<name>.yaml``."""
    return OmegaConf.load(CONFIG_DIR / f"{name}.yaml")


def distances(grid: DictConfig | None = None) -> np.ndarray:
    grid = grid if grid is not None else load_config("grid")
    d = grid.distances_m
    assert d.spacing == "log"
    return np.geomspace(d.start, d.stop, d.n)


def make_cond(cond_id: str, distance_m: float, grid: DictConfig | None = None, **overrides) -> ViewCond:
    """Build a ViewCond from a condition id in grid.yaml (c0-c9)."""
    grid = grid if grid is not None else load_config("grid")
    c = grid.conditions[cond_id]
    fields = dict(
        distance_m=float(distance_m),
        cond_id=cond_id,
        light=c.light,
        luminance_cd_m2=float(c.luminance_cd_m2),
        beta_haze=float(c.beta_haze),
        azimuth_view_deg=float(c.azimuth_view_deg),
    )
    fields.update(overrides)
    return ViewCond(**fields)
