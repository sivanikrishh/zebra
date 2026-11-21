"""Geometry, streak, csf, motion, noise, percept."""

import pytest
import torch

from helpers import NO_NOISE, cond, michelson, ones_mask, pipe, srgb_to_linear, stripes
from zebrasight.eye.csf import csf_params, csf_sensitivity
from zebrasight.eye.motion import line_kernel
from zebrasight.pipeline import PIPELINE_ORDER
from zebrasight.types import COMPUTE_DTYPE, STORAGE_DTYPE


def test_shapes_dtypes_downsampled():
    img = torch.rand(2, 3, 600, 900)
    p = pipe().run(img, 0.2, ones_mask(img), cond("c0", 100.0))  # fine source -> resampled
    assert not p.source_limited
    assert max(p.y.shape[-2:]) <= 256
    for t in (p.y, p.s, p.rho, p.mask):
        assert t.shape[0] == 2 and t.shape[1] == 1 and t.shape[-2:] == p.y.shape[-2:]
        assert t.dtype == COMPUTE_DTYPE
    assert p.for_storage().y.dtype == STORAGE_DTYPE
    geo = next(e for e in p.log if e["stage"] == "geometry")
    assert geo["resampled"] and abs(geo["deg_per_px"] - 0.01) < 1e-4


def test_close_range_is_source_limited():
    img = torch.rand(1, 3, 100, 100)
    p = pipe().run(img, 0.5, ones_mask(img), cond("c0", 2.0))
    geo = next(e for e in p.log if e["stage"] == "geometry")
    assert p.source_limited and not geo["resampled"]


def test_cpu_equals_cuda(cuda_available):
    torch.manual_seed(1)
    img = torch.rand(1, 3, 300, 400)
    mask = ones_mask(img)
    c = cond("c0", 40.0, speed_mps=3.0, field_el_deg=2.0)
    a = pipe(device="cpu").run(img, 0.3, mask, c)
    b = pipe(device="cuda").run(img, 0.3, mask, c)
    torch.testing.assert_close(b.y.cpu(), a.y, atol=1e-4, rtol=1e-3)
    torch.testing.assert_close(b.s.cpu(), a.s, atol=1e-4, rtol=1e-3)
    assert (b.rho.cpu() - a.rho).abs().mean() < 1e-3


def test_deterministic_with_seed():
    img = torch.rand(1, 3, 128, 128)
    mask = ones_mask(img)
    p = pipe()
    a = p.run(img, 0.5, mask, cond("c9", 5.0, seed=7))
    b = p.run(img, 0.5, mask, cond("c9", 5.0, seed=7))
    c = p.run(img, 0.5, mask, cond("c9", 5.0, seed=8))
    assert torch.equal(a.y, b.y)
    assert not torch.equal(a.y, c.y)


def test_night_noisier_than_day():
    img = torch.full((1, 3, 128, 128), 0.5)
    mask = ones_mask(img)
    p = pipe(**{"noise.neural_sigma.value": 0.0})
    day = p.run(img, 0.5, mask, cond("c0", 20.0)).y
    night = p.run(img, 0.5, mask, cond("c9", 20.0)).y
    assert night.std() / night.mean() > 10 * day.std() / day.mean()


def test_stripe_contrast_matches_human_luminance():
    lo, hi = 0.1, 0.9
    img = stripes(128, 512, period_px=128, lo=lo, hi=hi)  # 0.27 cpd at 1 cm/px, 20 m: well below CSF peak
    p = pipe(**NO_NOISE).run(img, 1.0, ones_mask(img), cond("c0", 20.0))
    y = p.y[0, 0].mean(dim=0)  # profile across stripes
    period = y.shape[-1] / 4
    q = int(period / 8)
    centres_hi = [int(period * (k + 0.25)) for k in range(4)]
    centres_lo = [int(period * (k + 0.75)) for k in range(4)]
    y_hi = torch.stack([y[c - q : c + q].mean() for c in centres_hi[1:3]]).mean()
    y_lo = torch.stack([y[c - q : c + q].mean() for c in centres_lo[1:3]]).mean()
    zebra = michelson(float(y_hi), float(y_lo))
    human = michelson(srgb_to_linear(hi), srgb_to_linear(lo))
    assert abs(zebra - human) / human < 0.05


def test_csf_constraint_and_cutoff():
    b, c, w = csf_params(23.3, 0.10, 50.0)
    s = csf_sensitivity(torch.tensor([b, 23.3]), b, c, w)
    torch.testing.assert_close(s, torch.tensor([50.0, 1.0]), rtol=1e-4, atol=1e-4)

    def grating(cpd, dpp=0.0115, h=64, w=256):  # sRGB sinusoid with Michelson ~1 in linear light
        x = torch.arange(w, dtype=torch.float32)
        g = 0.5 + 0.5 * torch.sin(2 * torch.pi * cpd * dpp * x)
        return g.view(1, 1, 1, w).expand(1, 3, h, w).clone()

    # 1 cm/px at 50 m -> 0.0115 deg/px: source grid kept, Nyquist 44 cpd
    fine = grating(30.0)  # above the 23.3 cpd acuity: removed
    coarse = grating(1.0)  # below the peak b = 2.33 cpd: passed
    p_fine = pipe(**NO_NOISE).run(fine, 1.0, ones_mask(fine), cond("c0", 50.0))
    p_coarse = pipe(**NO_NOISE).run(coarse, 1.0, ones_mask(coarse), cond("c0", 50.0))
    assert p_fine.y.std() / p_fine.y.mean() < 0.01
    assert p_coarse.y.std() / p_coarse.y.mean() > 0.5


def test_stripes_above_horizon_blur_more():
    img = stripes(256, 512, period_px=58)  # ~6 cpd after resampling to 0.01 deg/px
    mask = ones_mask(img)
    p = pipe(**NO_NOISE)
    on = p.run(img, 0.1, mask, cond("c0", 20.0, field_el_deg=0.0)).y
    above = p.run(img, 0.1, mask, cond("c0", 20.0, field_el_deg=10.0)).y
    c_on = on.std() / on.mean()
    c_above = above.std() / above.mean()
    assert c_on > 0.1
    assert c_above < 0.5 * c_on


def test_rho_of_half_black_stripes():
    img = stripes(128, 512, period_px=64)
    p = pipe(**NO_NOISE).run(img, 1.0, ones_mask(img), cond("c0", 20.0))
    inner = p.rho[..., 8:-8, 8:-8]
    assert abs(float(inner.mean()) - 0.5) < 0.05


def test_mask_survives_resampling():
    img = torch.rand(1, 3, 600, 800)
    mask = torch.zeros(1, 1, 600, 800)
    mask[..., 150:450, 200:600] = 1.0
    p = pipe().run(img, 0.2, mask, cond("c0", 100.0))
    assert set(p.mask.unique().tolist()) <= {0.0, 1.0}
    assert abs(float(p.mask.mean()) - float(mask.mean())) < 0.02
    labels = torch.zeros(1, 1, 600, 800)
    labels[..., :, :300], labels[..., :, 300:600], labels[..., :, 600:] = 1.0, 2.0, 3.0
    pl = pipe().run(img, 0.2, labels, cond("c0", 100.0))
    assert set(pl.mask.unique().tolist()) == {1.0, 2.0, 3.0}


def test_stage_flags():
    img = stripes(128, 256, period_px=10)
    mask = ones_mask(img)
    c = cond("c0", 20.0, speed_mps=5.0)
    full = pipe(**NO_NOISE).run(img, 0.5, mask, c)
    no_csf = pipe(**NO_NOISE, **{"eye_stages.csf": False}).run(img, 0.5, mask, c)
    assert not torch.allclose(full.y, no_csf.y)
    assert "csf" not in [e["stage"] for e in no_csf.log]

    bare = pipe(**{f"eye_stages.{n}": False for n in ("motion", "streak", "csf", "noise")})
    p = bare.run(img, 0.5, mask, c)
    assert [e["stage"] for e in p.log if e["stage"] not in ("input", "geometry", "eye", "percept")] == ["dichromacy"]

    with pytest.raises(ValueError):
        pipe(**{"eye_stages.dichromacy": False})

    # scene and atmosphere are no-ops
    all_on = pipe(PIPELINE_ORDER, **NO_NOISE).run(img, 0.5, mask, c)
    torch.testing.assert_close(all_on.y, full.y)


def test_motion_blur_length_and_direction():
    k = line_kernel(9.0, 0.0)
    assert k.shape == (11, 11)
    assert torch.isclose(k.sum(), torch.tensor(1.0))
    assert k[5].sum() > 0.99  # horizontal line in the centre row
    kv = line_kernel(9.0, 90.0)
    assert kv[:, 5].sum() > 0.99

    img = stripes(64, 256, period_px=16)  # vertical stripes: horizontal motion blurs them
    mask = ones_mask(img)
    still = pipe(**NO_NOISE).run(img, 0.5, mask, cond("c0", 20.0)).y
    moving = pipe(**NO_NOISE).run(img, 0.5, mask, cond("c0", 20.0, speed_mps=10.0)).y
    along = pipe(**NO_NOISE).run(img, 0.5, mask, cond("c0", 20.0, speed_mps=10.0, motion_dir_deg=90.0)).y
    assert moving.std() < 0.5 * still.std()
    torch.testing.assert_close(along, still, atol=1e-3, rtol=1e-3)  # motion along the stripes: no blur


def test_provenance_log_has_parameters():
    img = torch.rand(1, 3, 64, 64)
    p = pipe().run(img, 0.5, ones_mask(img), cond("c0", 20.0))
    by_stage = {e["stage"]: e for e in p.log}
    assert by_stage["dichromacy"]["L_peak_nm"] == 545
    assert by_stage["csf"]["b_cpd"] == pytest.approx(2.33)
    assert by_stage["streak"]["sigma_el_deg"] == 5.0
    assert "rho_window_px" in by_stage["percept"]
