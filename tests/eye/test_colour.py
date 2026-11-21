"""Dichromacy: spectral model and RGB -> [L, S]."""

import json

import numpy as np
import pytest
import torch

from helpers import NO_NOISE, cond, michelson, ones_mask, pipe, srgb_to_linear
from zebrasight.config import REPO_ROOT
from zebrasight.eye import spectral
from zebrasight.eye.dichromacy import cone_matrix

L_PEAK, S_PEAK = 545.0, 430.0


def chroma_of(wl, l_peak=L_PEAK):
    """Dichromat chromaticity s/(l+s) of a monochromatic light (independent of intensity)."""
    return float(spectral.chroma_coord(spectral.mono_catch(wl, l_peak), spectral.mono_catch(wl, S_PEAK)))


@pytest.mark.parametrize("l_peak", [545.0, 539.0])
def test_neutral_point_near_480(l_peak):
    assert abs(spectral.neutral_point_nm(l_peak, S_PEAK) - 480.0) <= 10.0


def test_480nm_looks_neutral():
    # white/grey has chroma 0.5 by construction; 480 nm should be close to it
    assert abs(chroma_of(480.0) - 0.5) < 0.15
    assert abs(chroma_of(spectral.neutral_point_nm(L_PEAK, S_PEAK)) - 0.5) < 1e-3


def test_red_green_confused_blue_yellow_not():
    # matched L catch: scale each light so L = 1; then compare the S signals
    def s_at_matched_l(wl):
        return float(spectral.mono_catch(wl, S_PEAK) / spectral.mono_catch(wl, L_PEAK))

    assert abs(s_at_matched_l(620.0) - s_at_matched_l(550.0)) < 0.01  # both ~0: indistinguishable
    assert abs(chroma_of(620.0) - chroma_of(550.0)) < 0.01
    assert abs(chroma_of(450.0) - chroma_of(580.0)) > 0.5  # distinguishable
    assert s_at_matched_l(450.0) > 5.0 and s_at_matched_l(580.0) < 0.01


def test_matrix_white_and_grey():
    m = cone_matrix(L_PEAK, S_PEAK)
    torch.testing.assert_close(m.sum(dim=1), torch.ones(2), atol=1e-6, rtol=0)  # white -> L = S = 1


@pytest.mark.parametrize("level", [0.1, 0.5, 0.9])
def test_grey_stays_grey(level):
    img = torch.full((1, 3, 64, 64), level)
    p = pipe(**NO_NOISE).run(img, 0.5, ones_mask(img), cond("c0", 20.0))
    torch.testing.assert_close(p.s, p.y, atol=1e-5, rtol=1e-4)  # S/L = 1


def test_night_is_rod_only():
    img = torch.rand(1, 3, 32, 32)
    p = pipe(**NO_NOISE).run(img, 0.5, ones_mask(img), cond("c9", 5.0))
    assert p.s is None
    dusk = pipe(**NO_NOISE).run(img, 0.5, ones_mask(img), cond("c8", 5.0))
    assert dusk.s is not None  # 0.16 cd/m2 is above the 0.02 colour threshold


def _patch_contrast(rgb_target, rgb_bg=(1.0, 1.0, 1.0)):
    """Michelson contrast of a patch on a background for zebra L and human Y (linear sRGB)."""
    m = cone_matrix(L_PEAK, S_PEAK).numpy()
    t = np.array([srgb_to_linear(v) for v in rgb_target])
    b = np.array([srgb_to_linear(v) for v in rgb_bg])
    y_w = np.array(spectral.SRGB_Y)
    return michelson(m[0] @ t, m[0] @ b), michelson(y_w @ t, y_w @ b)


def test_brown_on_white_report(capsys):
    zebra, human = _patch_contrast((120 / 255, 80 / 255, 50 / 255))
    with capsys.disabled():
        print(f"\n[report] brown (120,80,50) on white, Michelson contrast: zebra L {zebra:.3f}, human Y {human:.3f}")
    assert 0 < zebra < 1 and 0 < human < 1


def test_vr_matrix_file_matches_model():
    path = REPO_ROOT / "vr" / "shaders" / "dichromat_matrix.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    np.testing.assert_allclose(data["rgb_to_LS"], spectral.rgb_to_receptor_matrix((L_PEAK, S_PEAK)), atol=1e-12)
    sim = np.array(data["rgb_to_rgb_simulated"])
    np.testing.assert_allclose(sim @ [0, 0, 1], [0, 0, 1], atol=1e-9)  # Vienot anchors preserved
    np.testing.assert_allclose(sim @ [1, 1, 0], [1, 1, 0], atol=1e-9)
    np.testing.assert_allclose(sim @ [1, 1, 1], [1, 1, 1], atol=1e-9)
