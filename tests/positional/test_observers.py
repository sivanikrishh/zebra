import numpy as np
import pytest

from tests.positional import synth
from zebrasight.positional.observers import (aligned_minutiae, gabor_magnitude, gabor_phase, gabor_pooled,
                                             orientation_field, orientation_pooled)
from zebrasight.positional.observers import gabor_bank as gb

MASK = synth.ellipse()


def V(y):
    return synth.view(y, MASK)


def test_gabor_phase_identical_is_one():
    y = synth.stripes()
    assert gabor_phase.score(V(y), V(y), device="cpu")["score"] > 0.99


def test_gabor_phase_shift_without_search():
    a = synth.stripes()
    quarter = gabor_phase.score(V(a), V(synth.stripes(phase=np.pi / 2)), device="cpu", shift_search=False)
    half = gabor_phase.score(V(a), V(synth.stripes(phase=np.pi)), device="cpu", shift_search=False)
    # at the stripe scale (24 cpbl): a quarter-period shift is chance, a half-period shift INVERTS the phase
    assert abs(quarter["per_scale"][24.0] - 0.5) < 0.15
    assert half["per_scale"][24.0] < 0.2


def test_gabor_phase_residual_shift_recovered():
    a = synth.stripes()
    shift_px = 0.01 * synth.BL                      # on the search grid (+/- 2% of body length, 5 steps)
    b = synth.stripes(phase=-2 * np.pi * shift_px / synth.PERIOD)
    no = gabor_phase.score(V(a), V(b), device="cpu", shift_search=False)
    yes = gabor_phase.score(V(a), V(b), device="cpu", shift_search=True)
    # the synthetic coat has energy only at the stripe scale: recovery is judged there; coarse scales
    # carry little real signal and stay nearer chance
    assert yes["per_scale"][24.0] > 0.9 and yes["per_scale"][48.0] > 0.9
    assert yes["score"] > no["score"] + 0.3


def test_gabor_phase_different_pattern_is_chance():
    s = [gabor_phase.score(V(synth.stripes(seed=1)), V(synth.stripes(seed=k)), device="cpu", shift_search=False)["score"]
         for k in (2, 3, 4)]
    assert abs(np.mean(s) - 0.5) < 0.1


def test_magnitude_and_pooled_ignore_half_period_shift():
    a, b = synth.stripes(), synth.stripes(phase=np.pi)
    assert gabor_pooled.score(V(a), V(b), device="cpu")["score"] > 0.99
    assert gabor_magnitude.score(V(a), V(b), device="cpu")["score"] > 0.9


def test_orientation_field():
    a = synth.stripes(wobble=0.0)
    assert orientation_field.score(V(a), V(a))["score"] > 0.99
    rot = orientation_field.score(V(a), V(synth.stripes(angle_deg=30, wobble=0.0)))["score"]
    assert abs(rot - np.cos(np.radians(60))) < 0.1          # 30 deg -> cos(2 x 30 deg)
    assert orientation_field.score(V(a), V(synth.stripes(phase=np.pi, wobble=0.0)))["score"] > 0.95
    assert orientation_pooled.score(V(a), V(synth.stripes(phase=np.pi, wobble=0.0)))["score"] > 0.99


def test_minutiae_matching():
    pts = np.array([[10.0, 10.0], [50.0, 40.0], [90.0, 20.0]])
    i, j = aligned_minutiae.match_points(pts, pts + 1.0, tol=3.0)
    assert len(i) == 3
    i, _ = aligned_minutiae.match_points(pts, pts + 5.0, tol=3.0)
    assert len(i) == 0
    m = {"J": pts, "E": pts[:2]}
    assert aligned_minutiae.score_points(m, m, 1.0)[0] == 1.0
    far = {"J": pts + 10, "E": pts[:2] + 10}
    assert aligned_minutiae.score_points(m, far, 3.0)[0] == 0.0


def test_minutiae_end_to_end_identical():
    y = synth.stripes(seed=5)
    r = aligned_minutiae.score(V(y), V(y))
    assert r["score"] == pytest.approx(1.0) and r["n_minutiae"] > 0


@pytest.mark.skipif(not __import__("torch").cuda.is_available(), reason="no GPU")
def test_gabor_cpu_vs_gpu():
    y = synth.stripes(seed=2)
    a = gb.responses(y, MASK, [18.3, 36.7], 6, 1.0, device="cpu")
    b = gb.responses(y, MASK, [18.3, 36.7], 6, 1.0, device="cuda")
    assert np.abs(a - b).max() < 1e-3 * np.abs(a).max()
    h = gb.responses(y, MASK, [18.3], 6, 1.0, device="cuda", half=True)
    assert h.dtype == np.float16 and np.abs(h[:, :, 0].astype(np.float32) - a[:1].real).max() < 1e-2 * np.abs(a).max()
