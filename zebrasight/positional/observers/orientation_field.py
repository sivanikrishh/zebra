"""C5 orientation_field: stripe FLOW (direction of the stripes at each place), not their phase.

Structure tensor J = G_rho * (grad y grad y^T) inside the mask (normalised convolution), with gradient
sigma = orient_grad_sigma_period x the stripe period and integration rho = 1 and 2 stripe periods (all in
body-length units, converted to px). Doubled-angle orientation phi = atan2(2 Jxy, Jxx - Jyy) and coherence
c = ((l1 - l2) / (l1 + l2))^2, sampled on the orient_grid_uv canonical cells on both masks.
score = sum w cos(phi_g - phi_q) / sum w, w = sqrt(c_g c_q) (a 30 deg rotation of the stripes gives
cos 60 deg = 0.5); the mean over the rho values. A half-period shift leaves it unchanged.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

from zebrasight.positional.observers.base import View, bilinear, cell_pixels, cfg, inside

name = "orientation_field"
needs_alignment = True


def tensor_field(v: View, rho_periods: float):
    """-> (phi doubled angle, coherence) maps on the View's grid."""
    c = cfg()
    period = v.bl_px / c.stripe_period_cpbl
    m = v.mask.astype(float)
    mean = v.y[v.mask].mean() if v.mask.any() else 0.0
    y = np.where(v.mask, v.y, mean).astype(float)
    sg = max(0.5, c.orient_grad_sigma_period * period)
    gx = ndi.gaussian_filter(y, sg, order=(0, 1))
    gy = ndi.gaussian_filter(y, sg, order=(1, 0))
    sr = max(0.5, rho_periods * period)
    norm = ndi.gaussian_filter(m, sr) + 1e-9
    jxx = ndi.gaussian_filter(m * gx * gx, sr) / norm
    jyy = ndi.gaussian_filter(m * gy * gy, sr) / norm
    jxy = ndi.gaussian_filter(m * gx * gy, sr) / norm
    phi = np.arctan2(2 * jxy, jxx - jyy)
    tr = jxx + jyy
    coh = np.where(tr > 1e-12, ((jxx - jyy) ** 2 + 4 * jxy**2) / np.maximum(tr, 1e-12) ** 2, 0.0)
    return phi, coh


def score(g: View, q: View, device: str | None = None) -> dict:
    c = cfg()
    nu, nv = c.orient_grid_uv
    pts = cell_pixels(g, (nu, nv))
    ok = inside(g.mask, pts) & inside(q.mask, pts)
    if ok.sum() < c.min_valid_cells:
        return {"score": float("nan"), "cells": np.full((nv, nu), np.nan)}
    scores, cell_num, cell_den = [], np.zeros(len(pts)), np.zeros(len(pts))
    for rho in c.orient_rho_periods:
        (pg, cg), (pq, cq) = tensor_field(g, rho), tensor_field(q, rho)
        # sample cos/sin of the doubled angle (continuous), not the angle itself
        sg = bilinear(np.stack([np.cos(pg), np.sin(pg), cg]), pts[ok])
        sq = bilinear(np.stack([np.cos(pq), np.sin(pq), cq]), pts[ok])
        cos_d = (sg[0] * sq[0] + sg[1] * sq[1]) / np.maximum(np.hypot(sg[0], sg[1]) * np.hypot(sq[0], sq[1]), 1e-12)
        w = np.sqrt(np.clip(sg[2], 0, None) * np.clip(sq[2], 0, None))
        if w.sum() <= 0:
            continue
        scores.append(float((w * cos_d).sum() / w.sum()))
        cell_num[ok] += w * cos_d
        cell_den[ok] += w
    if not scores:
        return {"score": float("nan"), "cells": np.full((nv, nu), np.nan)}
    cells = np.where(cell_den > 0, cell_num / np.maximum(cell_den, 1e-12), np.nan).reshape(nv, nu)
    return {"score": float(np.mean(scores)), "cells": cells}
