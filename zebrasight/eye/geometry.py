"""Geometry stage: coat centimetres -> visual-field degrees.

Physics: a source pixel of size ``cm_per_px`` at distance d subtends
    deg_per_px = 2 * atan(cm_per_px / 100 / (2 d))   (degrees)
Each pixel gets visual-field coordinates: the crop centre sits at
(``cond.field_az_deg``, ``cond.field_el_deg``) and pixel offsets add ``deg_per_px`` steps
(elevation positive up). These maps let the streak stage vary acuity across the image.

``source_limited`` is set when the source Nyquist frequency 1 / (2 deg_per_px) is below the
best zebra acuity: the camera, not the eye, then limits what can be resolved.

Phase 1 TODO: resample image and mask to the working grid (eye.yaml working_grid,
0.01 deg/px) with an anti-aliasing filter; currently the source sampling is kept.
"""

from __future__ import annotations

import math

import torch

from zebrasight.registry import Stage, register_stage
from zebrasight.types import ViewState


def deg_per_px(cm_per_px: float, distance_m: float) -> float:
    return math.degrees(2.0 * math.atan(cm_per_px / 100.0 / (2.0 * distance_m)))


def field_maps(b: int, h: int, w: int, dpp: float, az0: float, el0: float, device: torch.device):
    """(B,1,H,W) azimuth and elevation maps in degrees, centred on (az0, el0)."""
    x = (torch.arange(w, device=device, dtype=torch.float32) - (w - 1) / 2) * dpp
    y = (torch.arange(h, device=device, dtype=torch.float32) - (h - 1) / 2) * dpp
    az = (az0 + x).view(1, 1, 1, w).expand(b, 1, h, w)
    el = (el0 - y).view(1, 1, h, 1).expand(b, 1, h, w)
    return az.contiguous(), el.contiguous()


@register_stage("geometry")
class Geometry(Stage):
    def forward(self, state: ViewState) -> ViewState:
        c = state.cond
        dpp = deg_per_px(state.cm_per_px, c.distance_m)
        b, _, h, w = state.mask.shape
        az, el = field_maps(b, h, w, dpp, c.field_az_deg, c.field_el_deg, state.device)
        best_acuity = max(self.param("acuity.photopic"), *self.eye_cfg.acuity.photopic.sweep)
        source_nyquist = 1.0 / (2.0 * dpp)
        state = state.evolve(deg_per_px=dpp, az_map=az, el_map=el,
                             source_limited=source_nyquist < best_acuity)
        self.record(state, deg_per_px=dpp, source_nyquist_cpd=source_nyquist, best_acuity_cpd=best_acuity,
                    source_limited=state.source_limited, resampled=False,
                    working_grid_deg_per_px=self.param("working_grid.deg_per_px"))
        return state
