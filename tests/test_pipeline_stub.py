"""End-to-end run of the stub pipeline on random images: shapes, day/night switch, provenance."""

import pytest
import torch

from zebrasight.config import load_config, make_cond
from zebrasight.pipeline import PIPELINE_ORDER, Pipeline, check_order
from zebrasight.types import COMPUTE_DTYPE, STORAGE_DTYPE

B, H, W = 2, 64, 96


@pytest.fixture
def batch():
    g = torch.Generator().manual_seed(0)
    img = torch.rand(B, 3, H, W, generator=g)
    mask = (torch.rand(B, 1, H, W, generator=g) > 0.5).float()
    meta = {"zebra_id": ["z01", "z02"], "image_id": ["i1", "i2"], "session_id": "s1", "region": "flank"}
    return img, mask, meta


@pytest.mark.parametrize("phase_stages", [None, list(PIPELINE_ORDER)], ids=["phase1", "all_stages"])
@pytest.mark.parametrize("cond_id", ["c0", "c8", "c9"])
def test_end_to_end_shapes(batch, phase_stages, cond_id):
    img, mask, meta = batch
    pipe = Pipeline.from_grid() if phase_stages is None else Pipeline(phase_stages)
    cond = make_cond(cond_id, distance_m=23.318)
    p = pipe.run(img, cm_per_px=0.5, mask=mask, cond=cond, meta=meta)

    assert p.y.shape == (B, 1, H, W)
    assert p.y.dtype == COMPUTE_DTYPE
    assert p.y.device.type == pipe.device.type
    assert p.mask.shape == (B, 1, H, W)
    assert p.rho.shape == (B, 1, H, W)
    if cond_id == "c9":  # night: below the 0.02 cd/m2 colour threshold -> rod only
        assert p.s is None
    else:  # dusk 0.16 cd/m2 is above the threshold: colour stays on
        assert p.s.shape == (B, 1, H, W)
    assert p.meta == meta
    assert p.deg_per_px > 0

    stages_logged = [e["stage"] for e in p.log]
    for name in pipe.stage_names + ["dichromacy", "streak", "csf", "motion", "noise"]:
        assert name in stages_logged
    assert stages_logged[0] == "input"

    stored = p.for_storage()
    assert stored.y.dtype == STORAGE_DTYPE


def test_phase1_is_geometry_plus_eye():
    assert Pipeline.from_grid().stage_names == ["geometry", "eye"]


def test_field_maps_centre_and_orientation(batch):
    img, mask, _ = batch
    pipe = Pipeline(["geometry", "eye"])
    cond = make_cond("c0", 10.0, field_az_deg=20.0, field_el_deg=-3.0)
    state_log = pipe.run(img, 0.5, mask, cond).log
    geo = next(e for e in state_log if e["stage"] == "geometry")
    dpp = geo["deg_per_px"]
    from zebrasight.eye.geometry import field_maps

    az, el = field_maps(1, H, W, dpp, 20.0, -3.0, torch.device("cpu"))
    assert torch.isclose(az[0, 0].mean(), torch.tensor(20.0), atol=1e-4)
    assert torch.isclose(el[0, 0].mean(), torch.tensor(-3.0), atol=1e-4)
    assert az[0, 0, 0, -1] > az[0, 0, 0, 0]  # azimuth grows to the right
    assert el[0, 0, 0, 0] > el[0, 0, -1, 0]  # elevation positive up


def test_source_limited_flag(batch):
    img, mask, _ = batch
    pipe = Pipeline(["geometry", "eye"])
    near = pipe.run(img, 0.5, mask, make_cond("c0", 2.0))  # 0.5 cm px at 2 m: ~0.14 deg/px
    far = pipe.run(img, 0.5, mask, make_cond("c0", 200.0))  # ~0.0014 deg/px
    assert near.source_limited and not far.source_limited


@pytest.mark.parametrize("bad", [["eye", "geometry"], ["geometry", "scene"], ["geometry", "warp", "eye"]])
def test_stage_order_enforced(bad):
    with pytest.raises(ValueError):
        check_order(bad)


def test_grid_config():
    grid = load_config("grid")
    assert len(grid.distances_m["values"]) == 16
    assert len(grid.conditions) == 10
    assert list(grid.regions) == ["flank", "face", "rump"]
