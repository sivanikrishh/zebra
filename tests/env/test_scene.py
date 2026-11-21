"""Scene stage: shadows, occlusion, background (small synthetic images)."""

import pytest
import torch
from omegaconf import OmegaConf

from zebrasight.config import load_config, make_cond
from zebrasight.env.scene import Scene, apply_shadow, shadow_map
from zebrasight.types import ViewState

ENV = load_config("env")
SRGB_Y = torch.tensor([0.2126, 0.7152, 0.0722]).view(1, 3, 1, 1)


def _stage():
    return Scene(OmegaConf.merge(load_config("eye"), {"env": ENV}))


def _state(cond, h=96, w=160, meta=None):
    rgb = torch.full((1, 3, h, w), 0.5)
    mask = torch.zeros(1, 1, h, w)
    mask[..., 10:80, 10:150] = 1.0
    return ViewState(rgb=rgb, mask=mask, cond=cond, cm_per_px=0.1,
                     meta=meta or {"image_id": "img1", "region": "flank"})


@pytest.mark.parametrize("darkening", [0.4, 0.55, 0.7])
def test_full_shadow_darkens_by_set_amount(darkening):
    rgb = torch.full((1, 3, 4, 4), 0.6)
    out = apply_shadow(rgb, torch.ones(1, 1, 4, 4), darkening, list(ENV.scene.shadow.tint_rgb.value))
    y_in, y_out = (rgb * SRGB_Y).sum(1), (out * SRGB_Y).sum(1)
    assert torch.allclose(y_out / y_in, torch.full_like(y_in, 1.0 - darkening), rtol=1e-5)
    assert out[0, 2, 0, 0] > out[0, 0, 0, 0]  # skylight-only shadow is bluish


@pytest.mark.parametrize("cid,frac", [("s25", 0.25), ("s50", 0.50)])
def test_shadow_fraction_and_darkening_in_stage(cid, frac):
    st = _state(make_cond(cid, 20.0))
    out = _stage()(st)
    log = next(e for e in out.log if e["stage"] == "scene")
    assert abs(log["shadow_frac_measured"] - frac) < 0.05
    y = (out.rgb * SRGB_Y).sum(1, keepdim=True)
    inside = st.mask > 0.5
    darkest = y[inside].min()
    assert darkest == pytest.approx(0.5 * (1 - float(ENV.scene.shadow.darkening.value)), rel=0.02)


def test_shadow_same_place_at_every_distance():
    a = shadow_map(torch.ones(1, 1, 64, 64), 0.3, 7, 0.25, 0.04)
    b = shadow_map(torch.ones(1, 1, 128, 128), 0.3, 7, 0.25, 0.04)
    b_small = torch.nn.functional.interpolate(b, size=(64, 64), mode="area")
    assert ((a >= 0.5) == (b_small >= 0.5)).float().mean() > 0.95


@pytest.mark.parametrize("cid,frac", [("o20", 0.20), ("o40", 0.40), ("o60", 0.60)])
def test_occlusion_fraction_and_mask_update(cid, frac):
    st = _state(make_cond(cid, 20.0))
    out = _stage()(st)
    before = (st.mask > 0.5).sum()
    after = (out.mask > 0.5).sum()
    assert abs(float(1 - after / before) - frac) < 0.05  # visible coat shrinks by the set fraction
    hidden = (st.mask > 0.5) & (out.mask < 0.5)
    grass = torch.tensor(ENV.scene.occlusion.grass_rgb.value).view(1, 3, 1, 1)
    col = out.rgb.permute(0, 2, 3, 1)[hidden[:, 0]].mean(0)
    assert torch.allclose(col, grass.view(3), rtol=0.25)  # occluded pixels are grass-coloured
    assert torch.equal(out.rgb[(out.mask > 0.5).expand_as(out.rgb)], st.rgb[(out.mask > 0.5).expand_as(st.rgb)])


def test_occlusion_reproducible_and_id_dependent():
    a = _stage()(_state(make_cond("o40", 20.0))).mask
    b = _stage()(_state(make_cond("o40", 50.0))).mask  # distance does not change the pattern
    c = _stage()(_state(make_cond("o40", 20.0), meta={"image_id": "img2", "region": "flank"})).mask
    assert torch.equal(a, b)
    assert not torch.equal(a, c)


@pytest.mark.parametrize("kind", ["short_green", "tall_dry", "bare_dust"])
def test_background_plate_composite(kind):
    st = _state(make_cond("c0", 20.0, background=kind))
    out = _stage()(st)
    inside = (st.mask > 0.5).expand_as(st.rgb)
    assert torch.equal(out.rgb[inside], st.rgb[inside])  # the zebra region is untouched
    outside = out.rgb.permute(0, 2, 3, 1)[(st.mask < 0.5)[:, 0]].mean(0)
    mean = torch.tensor(list(ENV.scene.background.plates[kind].value))
    assert torch.allclose(outside, mean, rtol=0.15)  # plate colour behind the zebra


def test_plate_standalone_for_vr():
    from zebrasight.env.scene import make_plate

    a = make_plate(64, 96, "tall_dry", [0.33, 0.27, 0.12], 0.18, seed=3)
    b = make_plate(128, 192, "tall_dry", [0.33, 0.27, 0.12], 0.18, seed=3)
    assert a.shape == (1, 3, 64, 96) and (a >= 0).all()
    b_small = torch.nn.functional.interpolate(b, size=(64, 96), mode="area")
    assert torch.corrcoef(torch.stack([a.flatten(), b_small.flatten()]))[0, 1] > 0.9  # same pattern at any size


def test_core_conditions_untouched():
    for cid in ("c0", "c1", "c4", "c8", "c9"):
        st = _state(make_cond(cid, 20.0))
        out = _stage()(st)
        assert torch.equal(out.rgb, st.rgb) and torch.equal(out.mask, st.mask)
