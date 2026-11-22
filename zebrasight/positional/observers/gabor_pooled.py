"""C4 gabor_pooled: control WITHOUT position. Mean |Gabor response| per (scale, orientation) over the whole
region, each image in its own geometry (no alignment); score = cosine similarity of the two energy vectors.
"""

from __future__ import annotations

import numpy as np

from zebrasight.positional.observers import gabor_bank as gb
from zebrasight.positional.observers.base import View, cfg

name = "gabor_pooled"
needs_alignment = False


def energy(v: View, device: str | None = None) -> np.ndarray:
    """(S * O,) mean magnitude per scale/orientation; NaN rows for scales finer than the grid."""
    c = cfg()
    scales = gb.wavelengths(v.bl_px, c.gabor_scales_cpbl, c.min_wavelength_px)
    lams = [lam for _, lam in scales if lam is not None]
    R = np.abs(gb.responses(v.y, v.mask, lams, c.gabor_orientations, c.gabor_bandwidth_oct, device))
    e = np.full((len(scales), c.gabor_orientations), np.nan)
    k = 0
    for i, (_, lam) in enumerate(scales):
        if lam is not None:
            e[i] = R[k][:, v.mask].mean(axis=1) if v.mask.any() else np.nan
            k += 1
    return e.ravel()


def score(g: View, q: View, device: str | None = None) -> dict:
    a, b = energy(g, device), energy(q, device)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 2:
        return {"score": float("nan")}
    a, b = a[ok], b[ok]
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return {"score": float(a @ b / d) if d > 0 else float("nan")}
