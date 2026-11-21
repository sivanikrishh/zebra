"""Batched distances (Pipeline.run_distances) vs one call per distance."""

import math

import pytest
import torch

from helpers import NO_NOISE, cond, ones_mask, pipe

DISTANCES = [2.0, 5.024, 12.619, 31.698, 79.621, 200.0]


def coat(h=300, w=420):
    yy, xx = torch.meshgrid(torch.arange(h, dtype=torch.float32), torch.arange(w, dtype=torch.float32), indexing="ij")
    c = (torch.sin(2 * math.pi * (xx + 12 * torch.sin(yy / 30)) / 24) > 0).float()
    return (0.1 + 0.8 * c).expand(1, 3, h, w).clone()


def centre(t, frac=0.5):
    h, w = t.shape[-2:]
    sh, sw = max(1, round(h * frac)), max(1, round(w * frac))
    return t[..., (h - sh) // 2 : (h - sh) // 2 + sh, (w - sw) // 2 : (w - sw) // 2 + sw]


def test_batched_matches_looped_in_the_centre():
    img = coat()
    m = ones_mask(img)
    p = pipe(**NO_NOISE)
    for cid in ("c0", "c9"):
        looped = [p.run(img, 0.1, m, cond(cid, d)) for d in DISTANCES]
        batched = p.run_distances(img, 0.1, m, cond(cid, DISTANCES[0]), DISTANCES)
        for d, a, b in zip(DISTANCES, looped, batched):
            assert a.y.shape == b.y.shape, (cid, d)
            assert a.display_limited == b.display_limited and a.source_limited == b.source_limited
            ca, cb = centre(a.y), centre(b.y)
            rel = float((ca - cb).abs().max() / ca.abs().mean().clamp_min(1e-9))
            assert rel < 0.02, (cid, d, rel)
            if a.s is not None:
                rs = float((centre(a.s) - centre(b.s)).abs().max() / centre(a.s).abs().mean().clamp_min(1e-9))
                assert rs < 0.02, (cid, d, rs)


def test_batch_of_one_is_exact():
    img = coat(120, 160)
    m = ones_mask(img)
    p = pipe(**NO_NOISE)
    a = p.run(img, 0.1, m, cond("c0", 12.619))
    (b,) = p.run_distances(img, 0.1, m, cond("c0", 99.0), [12.619])
    assert torch.equal(a.y, b.y) and torch.equal(a.s, b.s) and torch.equal(a.rho, b.rho)


def test_working_cap_flags_and_logs():
    img = torch.rand(1, 3, 300, 1400)  # source-limited at 2 m and wider than the 512 px working cap
    p = pipe(**NO_NOISE).run(img, 0.2, ones_mask(img), cond("c0", 2.0))
    geo = next(e for e in p.log if e["stage"] == "geometry")
    assert geo["display_limited"] and max(geo["size_out"]) <= 512
    assert geo["effective_cpd"] == pytest.approx(1 / (2.5 * geo["deg_per_px"]))
    assert p.display_limited


def test_batched_with_noise_is_reproducible():
    img = coat(120, 160)
    m = ones_mask(img)
    p = pipe()
    meta = {"image_id": "x", "region": "flank"}
    a = p.run_distances(img, 0.1, m, cond("c9", 2.0), DISTANCES[:3], meta=meta)
    b = p.run_distances(img, 0.1, m, cond("c9", 2.0), DISTANCES[:3], meta=meta)
    assert all(torch.equal(x.y, y.y) for x, y in zip(a, b))
