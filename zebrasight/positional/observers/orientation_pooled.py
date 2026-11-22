"""C6 orientation_pooled: control WITHOUT position. Coherence-weighted histogram of stripe orientation over
the whole region (orient_hist_bins over 180 deg, rho = the first orient_rho_periods value), each image in
its own geometry; score = cosine similarity of the normalised histograms.
"""

from __future__ import annotations

import numpy as np

from zebrasight.positional.observers.base import View, cfg
from zebrasight.positional.observers.orientation_field import tensor_field

name = "orientation_pooled"
needs_alignment = False


def histogram(v: View) -> np.ndarray:
    c = cfg()
    phi, coh = tensor_field(v, c.orient_rho_periods[0])
    theta = (phi[v.mask] / 2.0) % np.pi
    h, _ = np.histogram(theta, bins=int(c.orient_hist_bins), range=(0, np.pi), weights=coh[v.mask])
    s = h.sum()
    return h / s if s > 0 else h


def score(g: View, q: View, device: str | None = None) -> dict:
    if not g.mask.any() or not q.mask.any():
        return {"score": float("nan")}
    a, b = histogram(g), histogram(q)
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return {"score": float(a @ b / d) if d > 0 else float("nan")}
