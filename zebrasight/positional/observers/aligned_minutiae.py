"""C7 aligned_minutiae: stripe junctions (forks) and endings at the same places.

Minutiae come from the existing skeleton code (zebrasight.features.common.prepare: lighting-corrected Otsu
binary, spur pruning, junction clusters and skeleton endpoints) on the gallery percept and on the query
percept already warped onto the gallery grid (oracle alignment). Points are matched one-to-one per type
(Hungarian assignment) within minutiae_tol_period x the stripe period.
score = 2 x matches / (n_gallery + n_query)  (F1 of matches; 1 = all coincide, 0 = none).
Per-cell outputs (C8): matched and total minutiae per orient_grid_uv canonical cell.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from zebrasight.config import load_config
from zebrasight.features.common import prepare
from zebrasight.positional.frame import apply
from zebrasight.positional.observers.base import View, cfg

name = "aligned_minutiae"
needs_alignment = True


def minutiae(v: View) -> dict[str, np.ndarray] | None:
    """{"J": (n, 2) (x, y) px, "E": ...} or None when the percept is too small / degenerate."""
    p = {"y": v.y, "mask": v.mask, "deg_per_px": v.deg_per_px, "region": v.region if v.region != "body" else "flank",
         "cond": v.cond}
    ctx = prepare(p, v.distance_m)
    if not ctx.valid:
        return None
    rc = lambda pts: np.array([[c, r] for r, c in pts], float).reshape(-1, 2)  # (row, col) -> (x, y)
    return {"J": rc(ctx.junctions), "E": rc(ctx.endpoints)}


def match_points(a: np.ndarray, b: np.ndarray, tol: float) -> tuple[np.ndarray, np.ndarray]:
    """One-to-one matches within tol: -> (indices into a, indices into b)."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros(0, int), np.zeros(0, int)
    d = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1)
    cost = np.where(d <= tol, d, 1e6)
    i, j = linear_sum_assignment(cost)
    keep = d[i, j] <= tol
    return i[keep], j[keep]


def score_points(mg: dict, mq: dict, tol: float) -> tuple[float, dict]:
    matched, total, where = 0, 0, {"matched": [], "all": []}
    for t in ("J", "E"):
        i, _ = match_points(mg[t], mq[t], tol)
        matched += len(i)
        total += len(mg[t]) + len(mq[t])
        where["matched"].append(mg[t][i])
        where["all"].extend([mg[t], mq[t]])
    sc = 2.0 * matched / total if total else float("nan")
    return sc, where


def score(g: View, q: View, device: str | None = None) -> dict:
    c = cfg()
    nu, nv = c.orient_grid_uv
    mg, mq = minutiae(g), minutiae(q.like(q.y, q.mask & g.mask) if q.y.shape == g.y.shape else q)
    if mg is None or mq is None:
        return {"score": float("nan"), "cells": np.full((nv, nu), np.nan)}
    tol = c.minutiae_tol_period * g.bl_px / c.stripe_period_cpbl
    sc, where = score_points(mg, mq, tol)

    def cell_counts(pts):
        h = np.zeros((nv, nu))
        if len(pts):
            uv = apply(g.px2can, pts)
            U, V = (float(x) for x in load_config("positional").frame.grid_uv)
            iu = np.clip((uv[:, 0] / U * nu).astype(int), 0, nu - 1)
            iv = np.clip((uv[:, 1] / V * nv).astype(int), 0, nv - 1)
            np.add.at(h, (iv, iu), 1)
        return h

    m = cell_counts(np.vstack(where["matched"]) if where["matched"] else np.zeros((0, 2)))
    a = cell_counts(np.vstack(where["all"]) if where["all"] else np.zeros((0, 2)))
    cells = np.where(a > 0, 2 * m / np.maximum(a, 1), np.nan)
    return {"score": sc, "cells": cells, "n_minutiae": int(a.sum())}
