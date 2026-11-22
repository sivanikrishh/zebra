"""C3 gabor_magnitude: WHERE the stripe energy is (position of energy, not of phase).

|Gabor response| per (scale, orientation) sampled at the canonical phase-grid cell centres that lie on
both masks; log(1 + magnitude) with each channel's mean removed; score = Pearson r over all channels and
cells. A half-period shift barely changes it (magnitude is phase-free); moving stripe GROUPS does.
"""

from __future__ import annotations

import numpy as np

from zebrasight.positional.observers import gabor_bank as gb
from zebrasight.positional.observers.base import View, bilinear, cell_pixels, cfg, inside

name = "gabor_magnitude"
needs_alignment = True


def score(g: View, q: View, device: str | None = None) -> dict:
    c = cfg()
    lams = [lam for _, lam in gb.wavelengths(g.bl_px, c.gabor_scales_cpbl, c.min_wavelength_px) if lam is not None]
    if not lams:
        return {"score": float("nan")}
    pts = cell_pixels(g, c.phase_grid_uv)
    ok = inside(g.mask, pts) & inside(q.mask, pts)
    if ok.sum() < c.min_valid_cells:
        return {"score": float("nan")}
    out = []
    for v in (g, q):
        R = np.abs(gb.responses(v.y, v.mask, lams, c.gabor_orientations, c.gabor_bandwidth_oct, device))
        m = np.log1p(bilinear(R.reshape(-1, *R.shape[2:]), pts[ok]))
        out.append(m - m.mean(axis=1, keepdims=True))
    a, b = out[0].ravel(), out[1].ravel()
    if a.std() < 1e-12 or b.std() < 1e-12:
        return {"score": float("nan")}
    return {"score": float(np.corrcoef(a, b)[0, 1])}
