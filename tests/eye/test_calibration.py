"""Fast calibration checks (the full run is python -m zebrasight.calibration.melin)."""

import pytest

from zebrasight.calibration.gratings import grating, make_pipe
from zebrasight.calibration.melin import acuity_check, detection_distance
from zebrasight.config import load_config


@pytest.fixture(scope="module")
def pipe():
    return make_pipe()


def test_day_acuity_23_visible_26_gone(pipe):
    c = acuity_check(pipe)
    assert c[23.3] > 1e-3
    assert c[26.0] <= 1e-3


@pytest.mark.parametrize("light", ["day", "dusk", "night"])
def test_plains_widest_within_5pct_of_melin(pipe, light):
    row = load_config("melin_table2").rows[0]
    melin = float(row[f"{light}_m"])
    res = detection_distance(pipe, grating(float(row.width_cm)), light, melin, k_hi=8, k_lo=-8)
    assert abs(res["distance_m"] / melin - 1.0) <= 0.05
