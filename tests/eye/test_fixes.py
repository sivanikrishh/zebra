"""Peli CSF (crop-size independence), adaptive storage (moire, display cap), per-image noise seeds."""

import math

import torch

from helpers import NO_NOISE, cond, ones_mask, pipe


def wavy(h, w, period=50.0, amp=25.0, lo=0.1, hi=0.9):
    yy, xx = torch.meshgrid(torch.arange(h, dtype=torch.float32), torch.arange(w, dtype=torch.float32), indexing="ij")
    coat = (torch.sin(2 * math.pi * (xx + amp * torch.sin(yy / 60.0)) / period) > 0).float()
    return (lo + (hi - lo) * coat).expand(1, 3, h, w).clone()


def centre(t, size):
    h, w = t.shape[-2:]
    sh, sw = size
    return t[..., (h - sh) // 2 : (h - sh) // 2 + sh, (w - sw) // 2 : (w - sw) // 2 + sw]


def test_peli_crop_size_independent():
    big = wavy(1024, 1024)
    small = centre(big, (512, 512))  # same stripes, same scale, half the crop
    for cid, d in (("c0", 40.0), ("c8", 20.0)):
        res = {}
        for mode in ("peli", "global"):
            p = pipe(**NO_NOISE, **{"csf.mode.value": mode})
            ys = p.run(small, 0.1, ones_mask(small), cond(cid, d)).y
            yb = p.run(big, 0.1, ones_mask(big), cond(cid, d)).y
            region = (ys.shape[-2] // 2, ys.shape[-1] // 2)
            cs, cb = [float(centre(y, region).std() / centre(y, region).mean()) for y in (ys, yb)]
            res[mode] = abs(cs - cb) / cb
        assert res["peli"] < 0.05, (cid, res)


def test_wavy_close_to_unthresholded():
    """Peli keeps wavy stripes that the global per-coefficient rule wiped out."""
    img = wavy(600, 1200)
    peli = pipe(**NO_NOISE).run(img, 0.1, ones_mask(img), cond("c8", 20.0)).y
    glob = pipe(**NO_NOISE, **{"csf.mode.value": "global"}).run(img, 0.1, ones_mask(img), cond("c8", 20.0)).y
    c_peli, c_glob = float(peli.std() / peli.mean()), float(glob.std() / glob.mean())
    assert c_peli > 0.07 and c_peli > 1.2 * c_glob


def _dominant_cpd(y, dpp):
    prof = y[0, 0].mean(dim=0)
    prof = prof - prof.mean()
    spec = torch.fft.rfft(prof).abs()
    k = int(spec[1:].argmax()) + 1
    return k / prof.numel() / dpp


def test_no_moire_near_limit():
    # linear grating at 0.9 x acuity, source at 0.004 deg/px (0.1 cm/px at 14.3 m)
    dpp_src = math.degrees(math.atan(0.001 / 14.3))
    f = 0.9 * 23.3
    x = torch.arange(2048, dtype=torch.float32)
    g = (0.5 + 0.5 * torch.sin(2 * math.pi * f * dpp_src * x)).view(1, 1, 1, -1).expand(1, 3, 64, 2048).clone()
    p = pipe(**NO_NOISE).run(g, 0.1, ones_mask(g), cond("c0", 14.3), img_is_linear=True)
    assert abs(_dominant_cpd(p.y, p.deg_per_px) - f) / f < 0.03  # right frequency, not an alias


def test_antialias_suppresses_above_nyquist():
    # with the CSF off, a grating above the storage Nyquist must not alias into a visible pattern
    from zebrasight.eye.ops import gaussian_resize

    x = torch.arange(1000, dtype=torch.float32)
    f_px = 0.3  # cycles/px; downsample 2.5x -> new Nyquist 0.2 cycles/px; naive sampling aliases it to 0.05
    g = (0.5 + 0.5 * torch.sin(2 * math.pi * f_px * x)).view(1, 1, 1, -1).expand(1, 1, 16, 1000).clone()
    out = gaussian_resize(g, (16, 400))
    naive = torch.nn.functional.interpolate(g, size=(16, 400), mode="bilinear", align_corners=False)
    c_out = float(out[..., 20:-20].std() / out.mean())
    c_naive = float(naive[..., 20:-20].std() / naive.mean())
    assert c_naive > 0.3 and c_out < 0.3 * c_naive


def test_display_limited_flag():
    img = torch.rand(1, 3, 400, 4000)  # 4 m wide at 0.1 cm/px, seen from 5 m: ~43 deg
    p = pipe(**NO_NOISE).run(img, 0.1, ones_mask(img), cond("c0", 5.0))
    assert p.display_limited and max(p.y.shape[-2:]) <= 1024
    log = next(e for e in p.log if e["stage"] == "percept")
    assert log["effective_cpd"] < 23.3
    small = pipe(**NO_NOISE).run(img[..., :400], 0.1, ones_mask(img[..., :400]), cond("c0", 5.0))
    assert not small.display_limited


def test_storage_grid_adapts_to_regime():
    img = torch.rand(1, 3, 300, 300)
    day = pipe(**NO_NOISE).run(img, 0.1, ones_mask(img), cond("c0", 20.0))
    night = pipe(**NO_NOISE).run(img, 1.0, ones_mask(img), cond("c9", 20.0))  # ~35 px output
    assert abs(day.deg_per_px - 1 / (2.5 * 23.3)) / day.deg_per_px < 0.02
    assert abs(night.deg_per_px - 1 / (2.5 * 1.61)) / night.deg_per_px < 0.03


def _noise_of(meta, seed=0):
    # dusk, 256 px at 0.5 cm/px from 10 m -> ~230 x 230 output pixels (enough for |r| < 0.05)
    img = torch.full((1, 3, 256, 256), 0.5)
    on = pipe().run(img, 0.5, ones_mask(img), cond("c8", 10.0, seed=seed), meta=meta).y
    off = pipe(**NO_NOISE).run(img, 0.5, ones_mask(img), cond("c8", 10.0, seed=seed), meta=meta).y
    return (on - off).flatten()


def test_noise_differs_between_images_but_is_reproducible():
    a = _noise_of({"image_id": "img_001", "region": "flank"})
    a2 = _noise_of({"image_id": "img_001", "region": "flank"})
    b = _noise_of({"image_id": "img_002", "region": "flank"})
    c = _noise_of({"image_id": "img_001", "region": "face"})
    assert torch.equal(a, a2)
    for other in (b, c):
        r = torch.corrcoef(torch.stack([a, other]))[0, 1]
        assert abs(float(r)) < 0.05


def test_noise_seed_per_batch_item():
    img = torch.full((2, 3, 64, 64), 0.5)
    p = pipe().run(img, 0.5, ones_mask(img), cond("c9", 10.0), meta={"image_id": ["a", "b"], "region": "flank"})
    assert not torch.equal(p.y[0], p.y[1])
    seeds = next(e for e in p.log if e["stage"] == "noise")["seeds"]
    assert seeds[0] != seeds[1]
