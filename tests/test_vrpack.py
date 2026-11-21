"""The VR shader JSONs must match the model: regenerated values == files on disk == model functions."""

import json
import math

import numpy as np
import pytest

from zebrasight.config import REPO_ROOT, load_config
from zebrasight.env.atmosphere import channel_betas
from zebrasight.eye import spectral
from zebrasight.vrpack import params

OUT = REPO_ROOT / "vr" / "export" / "params"


def _load(name):
    path = OUT / name
    if not path.exists():
        pytest.skip(f"{path} not generated yet (python -m zebrasight.vrpack --params-only)")
    return json.loads(path.read_text(encoding="utf-8"))


def _close(a, b):
    if isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a:
            _close(a[k], b[k])
    elif isinstance(a, list):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            _close(x, y)
    elif isinstance(a, float) or isinstance(b, float):
        assert a == pytest.approx(b, rel=1e-9, abs=1e-12)
    else:
        assert a == b


@pytest.mark.parametrize("name", list(params.ALL))
def test_files_match_generator(name):
    _close(_load(name), json.loads(json.dumps(params.ALL[name]())))


def test_shader_copy_in_sync():
    on_disk = json.loads((REPO_ROOT / "vr" / "shaders" / "dichromat_matrix.json").read_text(encoding="utf-8"))
    _close(on_disk["rgb_to_LS"], _load("dichromat_matrix.json")["rgb_to_LS"])
    _close(on_disk["LS_to_rgb"], _load("dichromat_matrix.json")["LS_to_rgb"])


def test_values_match_model():
    eye, env, grid = load_config("eye"), load_config("env"), load_config("grid")
    dm = params.dichromat()
    np.testing.assert_allclose(dm["rgb_to_LS"], spectral.rgb_to_receptor_matrix((545.0, 430.0)))
    d = np.array(dm["LS_to_rgb"])
    np.testing.assert_allclose(np.array(dm["rgb_to_LS_display_inverse"]) @ d, np.eye(2), atol=1e-9)
    hz = params.haze()["conditions"]
    wl = list(env.atmosphere.wavelengths_nm.value)
    assert list(hz["c2"]["beta_rgb_per_m"].values()) == pytest.approx(channel_betas(0.01, 0.2, wl))
    assert list(hz["c1"]["beta_rgb_per_m"].values()) == pytest.approx(channel_betas(0.002, 1.3, wl))
    assert hz["c3"]["koschmieder_range_m"] == pytest.approx(3.912 / 0.03, rel=1e-3)
    assert hz["c0"]["applied_in_model_textures"] is False and hz["c1"]["applied_in_model_textures"] is True
    lt = params.light()["regimes"]
    assert lt["day"]["acuity_cpd"] == eye.acuity.photopic.value and lt["night"]["colour"] is False
    assert lt["dusk"]["colour"] is True and params.light()["aliases"] == {"dawn": "dusk"}
    assert params.loupe()["magnification"] == pytest.approx(23.3 / 12.5)
    fov = params.field_of_view()["acuity_map"]
    assert fov["sigma_el_deg"] == eye.streak.sigma_el_deg.value and fov["floor"] == eye.streak.periphery_floor.value
    assert set(hz) == set(grid.core_conditions)
    assert math.isinf(params.haze()["conditions"]["c0"]["koschmieder_range_m"]) is False


def test_textures_index_consistent():
    path = REPO_ROOT / "vr" / "export" / "textures" / "textures.json"
    if not path.exists():
        pytest.skip("textures not exported")
    tex = json.loads(path.read_text(encoding="utf-8"))
    assert len(tex["distance_bins_m"]) == 16
    assert {s["condition"] for s in tex["sets"]} == {"c0", "c8", "c9", "c1", "c2", "c3"}
    for s in tex["sets"]:
        assert len(s["layers"]) == 16 and 0 < s["exposure"]
        assert all((path.parent / lay["file"]).exists() for lay in s["layers"])
