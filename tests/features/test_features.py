"""Stripe features on synthetic percepts with known geometry."""

import math

import numpy as np
import pytest

from zebrasight.features import FEATURES, extract


def stripes(period_px: float, h=96, w=160, angle_deg=0.0, duty=0.5):
    yy, xx = np.mgrid[0:h, 0:w].astype(float)
    a = math.radians(angle_deg)
    u = xx * math.cos(a) + yy * math.sin(a)
    black = (u % period_px) < duty * period_px
    y = np.where(black, 0.1, 0.9)
    return y


def percept(y, dpp, mask=None, cond="c0"):
    mask = np.ones_like(y, bool) if mask is None else mask
    rho = np.full_like(y, 0.5)
    return {"y": y.astype(np.float32), "mask": mask, "rho": rho, "deg_per_px": dpp, "cond": cond, "computed_as": cond}


def dpp_for(cm_per_px, d):
    return math.degrees(math.atan(cm_per_px / 100.0 / d))


@pytest.mark.parametrize("d,period_px", [(10.0, 20.0), (40.0, 5.0)])
def test_f7_period_in_cm_at_two_distances(d, period_px):
    # the same coat: 10 cm period (5 cm stripes); at 4x the distance it is 4x fewer pixels
    cm_per_px = 10.0 / period_px
    p = percept(stripes(period_px, h=int(30 * period_px / 5) + 40, w=int(60 * period_px / 5) + 60), dpp_for(cm_per_px, d))
    f = extract(p, d)
    assert f["F7_period_cm"] == pytest.approx(10.0, rel=0.12)
    assert f["F7_fft_peak_cpcm"] == pytest.approx(0.1, rel=0.15)
    assert f["F1_width_cm_median"] == pytest.approx(5.0, rel=0.3)


def test_too_small_percept_is_all_nan():
    p = percept(stripes(4, h=10, w=10), 0.05)
    f = extract(p, 50.0)
    assert len(f) > 40 and all(math.isnan(v) for v in f.values())


def test_degenerate_binary_is_nan():
    y = np.full((64, 64), 0.5)
    y[0, 0] = 0.4  # nearly constant region
    f = extract(percept(y, 0.02), 20.0)
    assert math.isnan(f["F1_width_cm_median"]) and math.isnan(f["F7_period_cm"])


def test_orientation_and_density_outputs():
    p = percept(stripes(12, angle_deg=0), 0.02)
    f = extract(p, 20.0)
    # vertical stripes (intensity varies along x): stripe direction is vertical -> cos2a ~ -1
    assert f["F4_cos2a"] == pytest.approx(-1.0, abs=0.1)
    assert f["F4_coherence"] > 0.5
    assert f["F5_junctions_per100cm2"] == pytest.approx(0.0, abs=0.5)  # straight stripes: no Y-junctions
    for name in FEATURES:
        assert any(k.startswith(name + "_") for k in f), name
    assert sum(k.startswith("L_") and not k.startswith("L_rawotsu") for k in f) == 16 * 24
    assert sum(k.startswith("L_rawotsu_") for k in f) == 16 * 24 and sum(k.startswith("M_") for k in f) == 48


def test_lighting_correction_removes_shading():
    # stripes under a strong top-to-bottom lighting gradient: the raw single Otsu turns the dark
    # bottom solid black, the corrected binary keeps ~50% black everywhere (D1)
    y = stripes(16, h=160, w=200)
    shade = np.linspace(1.0, 0.25, 160)[:, None]
    p = percept(y * shade, dpp_for(0.5, 20.0))
    p["region"] = "flank"
    from zebrasight.features.common import prepare

    c = prepare(p, 20.0)
    bottom = slice(120, 160)
    assert c.black_raw[bottom].mean() > 0.85  # raw: shaded part all "black"
    assert abs(c.black[bottom].mean() - 0.5) < 0.1  # corrected: stripes kept
    f = extract(p, 20.0)
    assert abs(f["F2_rho_mean"] - 0.5) < 0.1 and f["F2_rawotsu_rho_mean"] != f["F2_rho_mean"]


def test_prune_length_rule():
    from zebrasight.features.common import prepare

    p = percept(stripes(40, h=120, w=200), dpp_for(0.1, 2.0))  # close range: acuity cycle < 1 px
    c = prepare(p, 2.0)
    assert c.prune_px >= 3 and c.prune_px >= round(0.5 * 20) - 1  # >= half the ~20 px stripe width
