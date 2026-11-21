"""Consistency checks between the literature configs."""

import math

import numpy as np
import pytest

from zebrasight.config import distances, load_config

GRADES = {"A", "B", "C"}
# grade-C values deliberately not swept in phase 1 (listed on the limits slide)
FIXED_C = {"working_grid.deg_per_px", "temporal.integration_ms_night", "cones.rod_peak", "luminance.dawn",
           "percept.max_side", "percept.samples_per_cycle"}


def test_distances_log_spaced():
    grid = load_config("grid")
    d = distances(grid)
    assert d[0] == 2.0 and d[-1] == pytest.approx(200.0)
    np.testing.assert_allclose(d, list(grid.distances_m["values"]), atol=5e-4)
    ratios = d[1:] / d[:-1]
    np.testing.assert_allclose(ratios, ratios[0])


def _params(node, path=""):
    for k, v in node.items():
        if k == "refs":
            continue
        if hasattr(v, "keys") and "grade" in v:
            yield f"{path}{k}", v
        elif hasattr(v, "keys"):
            yield from _params(v, f"{path}{k}.")


def test_every_parameter_has_grade_and_citation_rule():
    eye = load_config("eye")
    refs = eye.refs
    params = list(_params(eye))
    assert len(params) > 20
    for name, p in params:
        assert p.grade in GRADES, name
        cites = p.cite if p.cite is None or isinstance(p.cite, str) else list(p.cite)
        cites = [] if cites is None else [cites] if isinstance(cites, str) else cites
        if p.grade in {"A", "B"}:
            assert cites, f"{name}: grade {p.grade} needs a citation"
        for c in cites:
            assert c in refs, f"{name}: unknown ref {c}"
        if p.grade == "C" and p.get("sweep") is None and isinstance(p.value, (int, float)) and not isinstance(p.value, bool):
            # numeric grade-C values must be swept, except the documented fixed choices
            assert name in FIXED_C, name


def test_mesopic_is_melin_midpoint():
    a = load_config("eye").acuity
    assert a.mesopic.value == pytest.approx((a.photopic.value + a.scotopic.value) / 2, abs=0.01)


def test_melin_table2_matches_acuities():
    """d = stripe period / tan(1 / acuity): period = 2 x stripe width."""
    t = load_config("melin_table2")
    eye = load_config("eye").acuity
    acuity = {"day": eye.photopic.value, "dusk": eye.mesopic.value, "night": eye.scotopic.value}
    assert acuity == dict(t.acuity_cpd)
    for row in t.rows:
        period_m = 2 * row.width_cm / 100
        for light, cpd in acuity.items():
            d = period_m / math.tan(math.radians(1 / cpd))
            assert d == pytest.approx(row[f"{light}_m"], abs=0.01), (row.species, row.stripe, light)
