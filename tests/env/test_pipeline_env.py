"""Environment stages inside the full pipeline: order, c0 unchanged, effects end to end."""

import pytest
import torch

from zebrasight.config import load_config, make_cond
from zebrasight.pipeline import Pipeline, check_order

ALL = list(load_config("grid").all_stages)
META = {"image_id": "img1", "region": "flank"}


def coat(h=96, w=192):
    x = torch.arange(w)
    return ((x % 24) < 12).float().mul(0.8).add(0.1).view(1, 1, 1, w).expand(1, 3, h, w).clone()


def test_stage_order_enforced():
    assert ALL == ["geometry", "scene", "atmosphere", "eye"]
    for bad in (["geometry", "atmosphere", "scene", "eye"], ["scene", "geometry", "eye"], ["geometry", "eye", "scene"]):
        with pytest.raises(ValueError):
            check_order(bad)


def test_executed_in_physical_order():
    p = Pipeline(ALL).run(coat(), 0.1, torch.ones(1, 1, 96, 192), make_cond("c1", 50.0), meta=META)
    order = [e["stage"] for e in p.log if e["stage"] in ("geometry", "scene", "atmosphere", "eye")]
    assert order == ["geometry", "scene", "atmosphere", "eye"]


@pytest.mark.parametrize("d", [2.0, 23.318, 200.0])
def test_c0_unchanged_by_environment_stages(d):
    img, m = coat(), torch.ones(1, 1, 96, 192)
    base = Pipeline(["geometry", "eye"]).run(img, 0.1, m, make_cond("c0", d), meta=META)
    full = Pipeline(ALL).run(img, 0.1, m, make_cond("c0", d), meta=META)
    assert torch.equal(base.y, full.y) and torch.equal(base.s, full.s) and torch.equal(base.rho, full.rho)


def test_haze_angle_occlusion_end_to_end():
    img, m = coat(), torch.ones(1, 1, 96, 192)
    pipe = Pipeline(ALL)
    out = {c: pipe.run(img, 0.1, m, make_cond(c, 50.0), meta=META) for c in ("c0", "c3", "c5", "o40")}

    def contrast(p):
        y = p.y[0, 0]
        return float(y.std() / y.mean())

    assert contrast(out["c3"]) < 0.5 * contrast(out["c0"])  # dust storm at 50 m: t = exp(-1.5)
    assert out["c5"].y.shape[-1] < out["c0"].y.shape[-1]  # 60 deg: narrower
    assert out["o40"].mask.mean() < 0.7 * out["c0"].mask.mean()  # occlusion removes visible coat
