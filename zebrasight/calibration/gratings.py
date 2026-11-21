"""Synthetic stripe gratings and stripe-contrast measurement for calibration."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from omegaconf import DictConfig, OmegaConf

from zebrasight.config import load_config, make_cond
from zebrasight.pipeline import Pipeline

LIGHT_TO_COND = {"day": "c0", "dusk": "c8", "night": "c9"}
PX_PER_PERIOD = 40
N_PERIODS = 60
HEIGHT_PERIODS = 6


@dataclass
class Grating:
    img: torch.Tensor  # (1,3,H,W) LINEAR rgb
    cm_per_px: float
    period_cm: float
    mean_level: float


def grating(width_cm: float, contrast: float = 1.0, wavy: bool = False, n_periods: int = N_PERIODS) -> Grating:
    """Square-wave black/white stripes, stripe width w (period 2w), linear Michelson ``contrast``.

    wavy: stripes are phase-modulated down the image (x shift = 0.15 period x sin(2 pi y / 3 periods)),
    so every row is the same grating along x, shifted; stripe tilt is at most ~17 deg.
    """
    period_px = PX_PER_PERIOD
    h, w = HEIGHT_PERIODS * period_px, n_periods * period_px
    yy, xx = torch.meshgrid(torch.arange(h, dtype=torch.float64), torch.arange(w, dtype=torch.float64), indexing="ij")
    shift = 0.15 * period_px * torch.sin(2 * math.pi * yy / (3 * period_px)) if wavy else 0.0
    white = ((xx + shift) % period_px) < period_px / 2
    hi, lo = 1.0, (1.0 - contrast) / (1.0 + contrast)  # Michelson (hi - lo) / (hi + lo) = contrast
    img = torch.where(white, torch.tensor(hi, dtype=torch.float64), torch.tensor(lo, dtype=torch.float64))
    img = img.float().expand(1, 3, h, w).clone()
    return Grating(img=img, cm_per_px=2 * width_cm / period_px, period_cm=2 * width_cm, mean_level=(hi + lo) / 2)


def calibration_cfg(full_eye: bool = False, **overrides) -> DictConfig:
    """Acuity-only: geometry + dichromacy + streak (acuity map at field 0,0) + peli CSF; motion, noise off."""
    cfg = load_config("eye")
    if not full_eye:
        OmegaConf.update(cfg, "eye_stages.motion", False, merge=False)
        OmegaConf.update(cfg, "eye_stages.noise", False, merge=False)
    for k, v in overrides.items():
        OmegaConf.update(cfg, k, v, merge=False)
    return cfg


def make_pipe(full_eye: bool = False, **overrides) -> Pipeline:
    return Pipeline(["geometry", "eye"], eye_cfg=calibration_cfg(full_eye, **overrides))


def central(t: torch.Tensor, frac: float = 0.5) -> torch.Tensor:
    h, w = t.shape[-2:]
    sh, sw = max(1, round(h * frac)), max(1, round(w * frac))
    return t[..., (h - sh) // 2 : (h - sh) // 2 + sh, (w - sw) // 2 : (w - sw) // 2 + sw]


def fundamental_contrast(y: torch.Tensor, cycles_per_px: float, item: int = 0) -> float:
    """Contrast (amplitude / mean) of the grating fundamental along x, central region.

    Hann-weighted least-squares fit of a + b cos + c sin at the known frequency, per row (the
    window suppresses leakage from edges and slow trends); contrast = mean per-row amplitude
    over the mean level. Per row, so wavy (row-shifted) stripes count fully.
    """
    yc = central(y[item, 0]).double()
    n = yc.shape[-1]
    x = torch.arange(n, dtype=torch.float64, device=yc.device)
    basis = torch.stack([torch.ones_like(x), torch.cos(2 * math.pi * cycles_per_px * x),
                         torch.sin(2 * math.pi * cycles_per_px * x)], dim=1)  # (n,3)
    sw = torch.sqrt(torch.hann_window(n, periodic=False, dtype=torch.float64, device=yc.device)).clamp_min(1e-6)
    coef = torch.linalg.lstsq(basis * sw[:, None], (yc * sw).T).solution  # (3, rows)
    amp = torch.sqrt(coef[1] ** 2 + coef[2] ** 2)
    return float(amp.mean() / coef[0].mean().abs().clamp_min(1e-12))


def stripe_contrasts(pipe: Pipeline, g: Grating, light: str, distance_m: float, image_id: str = "cal",
                     n_seeds: int = 1) -> tuple[list[float], dict]:
    """Contrast for ``n_seeds`` copies of the grating in one batch. Each copy has its own image_id,
    hence its own noise seed (noise stage hashes image_id)."""
    cond = make_cond(LIGHT_TO_COND[light], distance_m)
    img = g.img.expand(n_seeds, -1, -1, -1).contiguous()
    ids = [f"{image_id}-s{i}" for i in range(n_seeds)] if n_seeds > 1 else image_id
    p = pipe.run(img, g.cm_per_px, torch.ones(n_seeds, 1, *g.img.shape[-2:]), cond,
                 meta={"image_id": ids, "region": "grating"}, img_is_linear=True)
    period_deg = math.degrees(math.atan(g.period_cm / 100.0 / distance_m))
    f_px = p.deg_per_px / period_deg
    info = {"deg_per_px": p.deg_per_px, "f_cycles_per_px": f_px, "f_cpd": 1 / period_deg,
            "size": tuple(p.y.shape[-2:]), "display_limited": p.display_limited, "source_limited": p.source_limited}
    if f_px >= 0.5:  # not representable on the stored grid
        return [0.0] * n_seeds, info
    return [fundamental_contrast(p.y, f_px, i) for i in range(n_seeds)], info


def stripe_contrast(pipe: Pipeline, g: Grating, light: str, distance_m: float, image_id: str = "cal") -> tuple[float, dict]:
    c, info = stripe_contrasts(pipe, g, light, distance_m, image_id, 1)
    return c[0], info


def blank_grating(g: Grating) -> Grating:
    return Grating(torch.full_like(g.img, g.mean_level), g.cm_per_px, g.period_cm, g.mean_level)


def blank_floor(pipe: Pipeline, g: Grating, light: str, distance_m: float, image_id: str = "blank") -> float:
    """Same measurement on a uniform field at the grating's mean level: the noise floor."""
    c, _ = stripe_contrast(pipe, blank_grating(g), light, distance_m, image_id)
    return c
