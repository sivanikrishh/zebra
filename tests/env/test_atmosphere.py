"""Atmosphere stage: Koschmieder + Angstrom (small synthetic images, CPU)."""

import math

import pytest
import torch

from zebrasight.config import load_config, make_cond
from zebrasight.env.atmosphere import Atmosphere, airlight_rgb, channel_betas, koschmieder
from zebrasight.types import ViewState

ENV = load_config("env")
WL = list(ENV.atmosphere.wavelengths_nm.value)


def _stage():
    from omegaconf import OmegaConf

    return Atmosphere(OmegaConf.merge(load_config("eye"), {"env": ENV}))


def _state(cond, value=0.2):
    rgb = torch.full((1, 3, 8, 8), value)
    return ViewState(rgb=rgb, mask=torch.ones(1, 1, 8, 8), cond=cond, cm_per_px=0.1)


def _contrast_vs_sky(out, a):
    return (out[0, :, 0, 0] - torch.tensor(a)) / torch.tensor(a)


@pytest.mark.parametrize("beta,d", [(0.002, 300.0), (0.01, 100.0), (0.03, 50.0)])
def test_contrast_follows_exp_minus_beta_d(beta, d):
    a = [0.9, 0.9, 0.9]
    rgb = torch.full((1, 3, 4, 4), 0.1)
    out = koschmieder(rgb, d, [beta] * 3, a)
    c0 = (0.1 - 0.9) / 0.9
    ratio = _contrast_vs_sky(out, a) / c0
    assert torch.allclose(ratio, torch.full((3,), math.exp(-beta * d)), rtol=0.02)


@pytest.mark.parametrize("beta", [0.002, 0.01, 0.03])
def test_two_percent_at_koschmieder_range(beta):
    a = [1.0, 1.0, 1.0]
    black = torch.zeros(1, 3, 2, 2)  # C0 = -1
    out = koschmieder(black, 3.912 / beta, [beta] * 3, a)
    assert torch.allclose(_contrast_vs_sky(out, a).abs(), torch.full((3,), 0.02), rtol=0.01)


def test_blue_fades_faster_under_haze_not_dust():
    haze = channel_betas(0.002, 1.3, WL)  # R, G, B
    dust = channel_betas(0.01, 0.2, WL)
    assert haze[2] / haze[0] > 1.4  # (464/611)^-1.3 = 1.43
    assert 1.0 < dust[2] / dust[0] < 1.07  # (464/611)^-0.2 = 1.06


def test_airlight_colour_and_dusk_dimming():
    st = _stage()
    day = st(_state(make_cond("c1", 100.0)))
    log_day = next(e for e in day.log if e["stage"] == "atmosphere")
    a = log_day["airlight_rgb_rel"]
    assert a[2] > a[1] > a[0]  # haze airlight is bluish
    assert sum(w * c for w, c in zip((0.2126, 0.7152, 0.0722), a)) == pytest.approx(0.90)
    dust = next(e for e in st(_state(make_cond("c2", 100.0))).log if e["stage"] == "atmosphere")["airlight_rgb_rel"]
    assert dust[0] > dust[1] > dust[2]  # yellow-brown
    dusk = st(_state(make_cond("c1", 100.0, light="dusk", luminance_cd_m2=0.16)))
    log_dusk = next(e for e in dusk.log if e["stage"] == "atmosphere")
    assert log_dusk["airlight_luminance_cd_m2"] == pytest.approx(log_day["airlight_luminance_cd_m2"] / 100)
    torch.testing.assert_close(dusk.rgb, day.rgb)  # relative image: same haze/scene ratio


def test_clear_condition_untouched():
    st = _stage()
    s = _state(make_cond("c0", 200.0))
    out = st(s)
    assert torch.equal(out.rgb, s.rgb)
    assert next(e for e in out.log if e["stage"] == "atmosphere")["applied"] is False


def test_airlight_scaling():
    a = airlight_rgb([1.0, 0.8, 0.55], 0.7)
    assert sum(w * c for w, c in zip((0.2126, 0.7152, 0.0722), a)) == pytest.approx(0.7)
