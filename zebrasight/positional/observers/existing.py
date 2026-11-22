"""Existing observers made usable with the positional pipeline.

- registered_layout_oracle: the registered-layout observer (amendment A4) with the ORACLE alignment instead
  of RANSAC on the percept: Pearson r of the local black-fraction map (features.common rho_corr: lighting-
  corrected Otsu binary, box mean over one acuity cycle) over the overlap of both masks.
- hotspotter_self: the HotSpotter observer recomputed on the two percepts (its own alignment):
  similarity = log(1 + RANSAC inliers). It has no oracle variant: matching IS its alignment.
- Pooled F1-F10 (zebrasight.features) and MegaDescriptor (zebrasight.metrics.megadesc) are unchanged:
  pooled statistics have no position; MegaDescriptor sees the whole crop and does its own pooling.
"""

from __future__ import annotations

import numpy as np

from zebrasight.features.common import prepare
from zebrasight.positional.align import self_align
from zebrasight.positional.observers.base import View, cfg


def _rho(v: View):
    p = {"y": v.y, "mask": v.mask, "deg_per_px": v.deg_per_px, "region": v.region if v.region != "body" else "flank",
         "cond": v.cond}
    ctx = prepare(p, v.distance_m)
    return ctx.rho_corr if ctx.valid else None


class registered_layout_oracle:  # noqa: N801 (module-like observer object)
    name = "registered_layout_oracle"
    needs_alignment = True

    @staticmethod
    def score(g: View, q: View, device: str | None = None) -> dict:
        rg, rq = _rho(g), _rho(q)
        if rg is None or rq is None:
            return {"score": float("nan")}
        ov = g.mask & q.mask
        if ov.sum() < cfg().min_valid_cells:
            return {"score": float("nan")}
        a, b = rg[ov], rq[ov]
        if a.std() < 1e-9 or b.std() < 1e-9:
            return {"score": float("nan")}
        return {"score": float(np.corrcoef(a, b)[0, 1])}


class hotspotter_self:  # noqa: N801
    name = "hotspotter_self"
    needs_alignment = False

    @staticmethod
    def score(g: View, q: View, device: str | None = None) -> dict:
        _, _, n = self_align(q.y, q.mask, g.y, g.mask)
        return {"score": float(np.log1p(n)), "inliers": n}
