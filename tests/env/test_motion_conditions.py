"""Motion conditions m2 / m5 (existing eye motion stage, exposed as condition ids)."""

import math

import pytest
import torch
from omegaconf import OmegaConf

from zebrasight.config import load_config, make_cond
from zebrasight.pipeline import Pipeline


@pytest.mark.parametrize("cid,speed", [("m2", 2.0), ("m5", 5.0)])
def test_condition_speed(cid, speed):
    c = make_cond(cid, 20.0)
    assert c.speed_mps == speed and c.light == "day" and c.aerosol == "clear"


def test_motion_blur_grows_with_speed():
    eye = load_config("eye")
    OmegaConf.update(eye, "eye_stages.noise", False, merge=False)
    pipe = Pipeline(["geometry", "eye"], eye_cfg=eye)
    w = 400
    x = torch.arange(w)
    img = ((x % 40) < 20).float().view(1, 1, 1, w).expand(1, 3, 64, w).clone()  # vertical stripes
    m = torch.ones(1, 1, 64, w)
    out = {cid: pipe.run(img, 0.1, m, make_cond(cid, 10.0)) for cid in ("c0", "m2", "m5")}
    blur = {cid: next(e for e in p.log if e["stage"] == "motion")["blur_px"] for cid, p in out.items()}
    assert blur["c0"] == 0 and 0 < blur["m2"] < blur["m5"]
    t = float(eye.temporal.integration_ms_day.value) / 1000
    expected_deg = math.degrees(5.0 * t / 10.0)
    work_dpp = next(e for e in out["m5"].log if e["stage"] == "geometry")["deg_per_px"]  # blur acts on the working grid
    assert blur["m5"] * work_dpp == pytest.approx(expected_deg, rel=0.02)
    def central_contrast(p):  # long blur kernels create edge ramps: judge the centre
        y = p.y[0, 0]
        mid = y[:, y.shape[1] // 4 : 3 * y.shape[1] // 4]
        return float(mid.std() / mid.mean())

    c = {cid: central_contrast(p) for cid, p in out.items()}
    assert c["c0"] > c["m2"] > c["m5"]
