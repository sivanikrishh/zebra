import math

import pytest
import torch
from omegaconf import OmegaConf

from zebrasight.config import load_config, make_cond
from zebrasight.pipeline import Pipeline


def eye_cfg(**overrides):
    """eye.yaml with dotted overrides, e.g. eye_cfg(**{"eye_stages.noise": False})."""
    cfg = load_config("eye")
    for k, v in overrides.items():
        OmegaConf.update(cfg, k, v, merge=False)
    return cfg


NO_NOISE = {"eye_stages.noise": False}


def pipe(stages=("geometry", "eye"), device=None, **overrides) -> Pipeline:
    return Pipeline(list(stages), eye_cfg=eye_cfg(**overrides), device=device)


def stripes(h: int, w: int, period_px: float, lo: float = 0.0, hi: float = 1.0, vertical: bool = True) -> torch.Tensor:
    """(1,3,H,W) square-wave stripes in sRGB [0,1], 50/50 duty cycle."""
    n = w if vertical else h
    phase = (torch.arange(n) % period_px) < period_px / 2
    line = torch.where(phase, torch.tensor(hi), torch.tensor(lo))
    img = line.view(1, 1, 1, n).expand(1, 3, h, n) if vertical else line.view(1, 1, n, 1).expand(1, 3, n, w)
    return img.clone()


def ones_mask(img: torch.Tensor) -> torch.Tensor:
    return torch.ones(img.shape[0], 1, *img.shape[-2:])


def cond(cond_id="c0", distance_m=20.0, **kw):
    return make_cond(cond_id, distance_m, **kw)


def srgb_to_linear(v):
    return v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4


def michelson(a, b):
    return abs(a - b) / (a + b)




