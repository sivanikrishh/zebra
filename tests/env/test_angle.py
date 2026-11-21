"""Viewing angle (geometry foreshortening)."""

import math

import torch
from omegaconf import OmegaConf

from zebrasight.config import load_config, make_cond
from zebrasight.pipeline import Pipeline


def _pipe() -> Pipeline:
    eye = load_config("eye")
    OmegaConf.update(eye, "eye_stages.noise", False, merge=False)
    return Pipeline(["geometry", "eye"], eye_cfg=eye)


def _stripe_period_px(y: torch.Tensor) -> float:
    prof = y[0, 0].mean(dim=0)
    prof = prof - prof.mean()
    spec = torch.fft.rfft(prof).abs()
    return prof.numel() / (int(spec[1:].argmax()) + 1)


def test_cos60_halves_stripe_width():
    w = 480  # vertical stripes, 48 px period at 0.1 cm/px; well resolved at 5 m, under the 512 px cap
    x = torch.arange(w)
    img = ((x % 48) < 24).float().view(1, 1, 1, w).expand(1, 3, 64, w).clone()
    m = torch.ones(1, 1, 64, w)
    pipe = _pipe()
    side = pipe.run(img, 0.1, m, make_cond("c0", 5.0))
    oblique = pipe.run(img, 0.1, m, make_cond("c5", 5.0))  # 60 deg
    geo = next(e for e in oblique.log if e["stage"] == "geometry")
    assert abs(geo["foreshortening_factor"] - 0.5) < 1e-9
    ps = _stripe_period_px(side.y) * side.deg_per_px
    po = _stripe_period_px(oblique.y) * oblique.deg_per_px
    assert abs(po / ps - math.cos(math.radians(60))) < 0.03  # stripe width halves
    assert oblique.y.shape[-2] == side.y.shape[-2]  # height unchanged


def test_c0_has_no_foreshortening():
    img = torch.rand(1, 3, 64, 96)
    p = _pipe().run(img, 0.5, torch.ones(1, 1, 64, 96), make_cond("c0", 20.0))
    geo = next(e for e in p.log if e["stage"] == "geometry")
    assert geo["foreshortening_factor"] == 1.0 and geo["view_azimuth_deg"] == 0.0
