"""C2 gabor_phase: exact stripe POSITION, IrisCode style.

At the centre of every canonical cell (phase_grid_uv), for each valid scale and orientation, the complex
Gabor response gives 2 bits: sign(Re), sign(Im). A bit is valid when the cell centre is on BOTH masks and
both responses exceed the noise floor (phase_noise_floor x that image's median |response| over all scales
and orientations, so scales with little real energy drop out).
score = 1 - fractional Hamming distance over valid bits (1 = identical phase, 0.5 = unrelated, 0 = every
bit flipped, e.g. a half-period shift at that scale). A small residual shift of the query sampling grid
(+/- phase_shift_search_bl of body length) is searched and the best kept, as iris matching searches
rotations. Outputs per-scale scores (at the best shift) and per-cell agreement (C8).
"""

from __future__ import annotations

import numpy as np

from zebrasight.positional.observers import gabor_bank as gb
from zebrasight.positional.observers.base import View, bilinear, cell_pixels, cfg, inside

name = "gabor_phase"
needs_alignment = True


def _floor(R: np.ndarray, mask: np.ndarray, k: float) -> np.ndarray:
    """(S,) noise floor: k x the median |R| over mask pixels, ALL scales and orientations together.
    One floor for the whole bank, so a scale carrying little real energy (only noise) drops out instead
    of adding chance-level bits (a per-scale floor would always pass half of any scale's samples)."""
    mag = np.abs(R)
    f = k * np.median(mag[:, :, mask]) if mask.any() else np.inf
    return np.full(len(R), f)


def score(g: View, q: View, device: str | None = None, shift_search: bool = True) -> dict:
    c = cfg()
    scales = gb.wavelengths(g.bl_px, c.gabor_scales_cpbl, c.min_wavelength_px)
    use = [(f, lam) for f, lam in scales if lam is not None]
    nu, nv = c.phase_grid_uv
    nan = {"score": float("nan"), "per_scale": {f: float("nan") for f, _ in scales},
           "cells": np.full((nv, nu), np.nan), "n_valid": 0}
    if not use:
        return nan
    lams = [lam for _, lam in use]
    Rg = gb.responses(g.y, g.mask, lams, c.gabor_orientations, c.gabor_bandwidth_oct, device)
    Rq = gb.responses(q.y, q.mask, lams, c.gabor_orientations, c.gabor_bandwidth_oct, device)
    S, O = Rg.shape[:2]
    pts = cell_pixels(g, (nu, nv))
    fg, fq = _floor(Rg, g.mask, c.phase_noise_floor), _floor(Rq, q.mask, c.phase_noise_floor)

    def sample(R, p):
        flat = R.reshape(S * O, *R.shape[2:])
        return (bilinear(flat.real, p) + 1j * bilinear(flat.imag, p)).reshape(S, O, -1)

    a = sample(Rg, pts)
    ok_g = inside(g.mask, pts)
    span = c.phase_shift_search_bl * g.bl_px if shift_search else 0.0
    steps = np.linspace(-span, span, int(c.phase_shift_steps)) if span > 0 else np.array([0.0])
    best = None
    for dx in steps:
        for dy in steps:
            pq = pts + np.array([dx, dy])
            b = sample(Rq, pq)
            valid = (ok_g & inside(q.mask, pq))[None, None, :] \
                & (np.abs(a) >= fg[:, None, None]) & (np.abs(b) >= fq[:, None, None])
            agree = (np.sign(a.real) == np.sign(b.real)).astype(float) + (np.sign(a.imag) == np.sign(b.imag))
            n = valid.sum()
            if n == 0:
                continue
            sc = float((agree * valid).sum() / (2 * n))
            if best is None or sc > best[0]:
                best = (sc, agree, valid)
    if best is None:
        return nan
    sc, agree, valid = best
    n_cells = int(valid.any(axis=(0, 1)).sum())
    if n_cells < c.min_valid_cells:
        return {**nan, "n_valid": n_cells}
    per_scale = {f: float("nan") for f, _ in scales}
    for i, (f, _) in enumerate(use):
        v = valid[i]
        if v.any():
            per_scale[f] = float((agree[i] * v).sum() / (2 * v.sum()))
    nv_cell = valid.sum(axis=(0, 1))
    cells = np.where(nv_cell > 0, (agree * valid).sum(axis=(0, 1)) / np.maximum(2 * nv_cell, 1), np.nan)
    return {"score": sc, "per_scale": per_scale, "cells": cells.reshape(nv, nu), "n_valid": n_cells}
