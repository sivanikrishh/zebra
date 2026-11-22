import cv2
import numpy as np
from scipy import ndimage as ndi

from tests.positional import synth
from zebrasight.positional.align import PairAlign, align_pair, apply_tps, fit_tps, oracle_warp
from zebrasight.positional.frame import apply, fit_frame, part_mask, percept_to_canonical, warp_to_canonical

CELL = 440 / 100  # px per canonical u-unit at body length 440


def _affine_about_centre(angle, scale, flip=False, shift=(0, 0)):
    M = cv2.getRotationMatrix2D((synth.CX, synth.CY), angle, scale)
    if flip:
        F = np.array([[-1, 0, 2 * synth.CX], [0, 1, 0], [0, 0, 1.0]])
        M = (np.vstack([M, [0, 0, 1]]) @ F)[:2]
    M[:, 2] += shift
    return M


def test_canonical_frame_rotation_flip_scale():
    mask = synth.ellipse()
    base = fit_frame(mask, (0, 0), head_left=True)
    pts = np.array([[60.0, 110.0], [240.0, 90.0], [400.0, 150.0], [300.0, 180.0]])
    for angle, scale, flip in [(20, 1.0, False), (-35, 0.8, False), (10, 1.2, True), (0, 0.6, True)]:
        M = _affine_about_centre(angle, scale, flip, shift=(210, 180))   # centred on a canvas big enough
        W2, H2 = 900, 600
        m2 = cv2.warpAffine(mask.astype(np.uint8), M, (W2, H2), flags=cv2.INTER_NEAREST) > 0
        assert not (m2[:, 0].any() or m2[:, -1].any() or m2[0].any() or m2[-1].any())  # body not cut
        fr = fit_frame(m2, (0, 0), head_left=not flip)   # a mirrored animal faces the other way
        a = apply(base.A, pts)
        b = apply(fr.A, apply(M, pts))
        assert np.abs(a - b).max() < 1.0, (angle, scale, flip, np.abs(a - b).max())


def test_percept_maps_share_the_frame():
    mask = synth.ellipse()
    fr = fit_frame(mask, (0, 0), True)
    bbox = [0, 0, synth.W, synth.H]
    for f in (1, 2, 4, 8):
        ph, pw = synth.H // f, synth.W // f
        M = percept_to_canonical(fr, bbox, (synth.H, synth.W), (ph, pw))
        # the percept pixel covering photo point (240, 120) maps to the same canonical point
        p = np.array([[(240 + 0.5) / f - 0.5, (120 + 0.5) / f - 0.5]])
        assert np.abs(apply(M, p) - apply(fr.A, [[240, 120]])).max() < 1e-6


def test_part_masks_and_canonical_warp():
    assert part_mask("rump").sum() > 0 and not (part_mask("head") & part_mask("rump")).any()
    mask = synth.ellipse()
    fr = fit_frame(mask, (0, 0), True)
    c = warp_to_canonical(mask.astype(np.float32), fr.A, order="nearest")
    assert c.shape == (60, 100) and c.mean() > 0.7     # the ellipse fills its own frame (pi/4 of the box)


def _texture(seed=0):
    rng = np.random.default_rng(seed)
    t = ndi.gaussian_filter(rng.standard_normal((synth.H, synth.W)), 3)
    return (0.5 * synth.stripes() + 0.5 * (t > 0)).astype(np.float32)


def test_known_affine_recovered_by_sift_and_tps():
    g = _texture()
    mask = synth.ellipse(a=200, b=95)
    A = _affine_about_centre(8, 0.9, shift=(6, -4))      # query = g warped by A (gallery px -> query px)
    q = cv2.warpAffine(g, A, (synth.W, synth.H), flags=cv2.INTER_LINEAR)
    qm = cv2.warpAffine(mask.astype(np.uint8), A, (synth.W, synth.H), flags=cv2.INTER_NEAREST) > 0
    rgb = lambda x: np.repeat(x[..., None], 3, axis=-1)
    pa = align_pair(rgb(q), qm, rgb(g), mask)
    assert pa.kind in ("tps", "affine") and pa.inliers >= 12
    test = np.array([[150.0, 100.0], [300.0, 140.0], [240.0, 80.0]])
    assert np.abs(pa.g2q(test) - apply(A, test)).max() < 1.5
    assert pa.aligned_ok and pa.overlap > 0.8
    # TPS on exactly affine data reproduces the affine
    src = np.random.default_rng(1).uniform(50, 400, (20, 2))
    t = fit_tps(src, apply(A, src), lam=0.0)
    assert np.abs(apply_tps(t, test) - apply(A, test)).max() < 1e-6


def test_oracle_alignment_keeps_cells_aligned_at_all_distances():
    # coarse texture (blobs ~12 px) so that content survives 8x downsampling; this tests ALIGNMENT,
    # not how much detail a coarse grid keeps
    g = ndi.gaussian_filter(np.random.default_rng(3).standard_normal((synth.H, synth.W)), 12).astype(np.float32)
    A = _affine_about_centre(8, 0.9, shift=(6, -4))
    q = cv2.warpAffine(g, A, (synth.W, synth.H), flags=cv2.INTER_LINEAR)
    pa = PairAlign("affine", A)
    for f in (1, 2, 4, 8):
        gs = cv2.resize(g, (synth.W // f, synth.H // f), interpolation=cv2.INTER_AREA)
        qs = cv2.resize(q, (synth.W // f, synth.H // f), interpolation=cv2.INTER_AREA)
        w = oracle_warp(qs, pa, (synth.H, synth.W), (synth.H, synth.W), gs.shape)
        ok = np.zeros_like(gs, bool)
        ok[gs.shape[0] // 4: 3 * gs.shape[0] // 4, gs.shape[1] // 4: 3 * gs.shape[1] // 4] = True
        r = np.corrcoef(w[ok], gs[ok])[0, 1]
        assert r > 0.9, (f, r)


def test_plausible_affine_rejects_degenerate_maps():
    from zebrasight.positional.align import plausible_affine

    ident = np.array([[1.0, 0, 0], [0, 1.0, 0]])
    assert plausible_affine(ident, ident)[0]
    assert plausible_affine(np.array([[1.1, 0.1, 5], [-0.1, 0.95, 2]]), ident)[0]
    assert plausible_affine(np.array([[-1.0, 0, 0], [0, 1.0, 0]]), ident) == (False, "reflection")
    assert plausible_affine(np.array([[3.0, 0, 0], [0, 0.2, 0]]), ident) == (False, "anisotropy")
    assert plausible_affine(np.array([[3.0, 0, 0], [0, 3.0, 0]]), ident) == (False, "scale")
