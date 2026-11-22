"""Shared interface for the positional observers (Part C).

Every observer works on a View: one luminance image on a pixel grid, its region mask, and the grid's
geometry (body length in px, the px -> canonical-frame map). Positional observers compare a gallery View
with the QUERY ALREADY WARPED onto the gallery grid (oracle alignment, B3), so both share the gallery's
geometry. Pooled controls (needs_alignment = False) compare each image in its own geometry.

score(gallery, query) -> dict with "score" (similarity; higher = more alike; NaN = not computable, which
the metrics count as a miss) and optional "per_scale" and "cells" (per-canonical-cell agreement, C8).

Scores are comparable only WITHIN one distance and light (never across: blur inflates similarity).
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from zebrasight.config import load_config
from zebrasight.positional.frame import apply, canonical_points, invert


def cfg():
    return load_config("positional").observers


@dataclass
class View:
    y: np.ndarray            # (H, W) float32 luminance (L cone; linear)
    mask: np.ndarray         # (H, W) bool region mask
    bl_px: float             # body length in this grid's px
    px2can: np.ndarray       # 2x3 this grid's px -> canonical (u, v)
    deg_per_px: float = float("nan")
    distance_m: float = float("nan")
    region: str = "flank"
    cond: str = "c0"

    def like(self, y: np.ndarray, mask: np.ndarray) -> "View":
        """A View on the SAME grid (used for a query warped onto this gallery grid)."""
        return replace(self, y=y.astype(np.float32), mask=mask.astype(bool))


def cell_pixels(view: View, grid_cells_uv) -> np.ndarray:
    """(N, 2) px positions (x, y) of canonical cell centres on this View's grid."""
    return apply(invert(view.px2can), canonical_points(grid_cells_uv))


def inside(mask: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """(N,) bool: point lies on the grid and on the mask (nearest pixel)."""
    x, y = np.round(pts[:, 0]).astype(int), np.round(pts[:, 1]).astype(int)
    ok = (x >= 0) & (y >= 0) & (x < mask.shape[1]) & (y < mask.shape[0])
    out = np.zeros(len(pts), bool)
    out[ok] = mask[y[ok], x[ok]]
    return out


def bilinear(img: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """img (C, H, W) or (H, W) sampled at (N, 2) px -> (C, N) or (N,), zero outside."""
    import cv2

    a = img if img.ndim == 3 else img[None]
    mx = pts[:, 0].astype(np.float32)[None]
    my = pts[:, 1].astype(np.float32)[None]
    out = np.stack([cv2.remap(np.ascontiguousarray(c, np.float32), mx, my, cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)[0] for c in a])
    return out if img.ndim == 3 else out[0]


def score_matrix(observer, gallery: list[View], query: list[View]) -> np.ndarray:
    """S (n_query x n_gallery) like zebrasight.metrics.core (query rows); NaN -> minimum similarity."""
    S = np.array([[observer.score(g, q)["score"] for g in gallery] for q in query], float)
    return np.where(np.isfinite(S), S, -np.inf)
