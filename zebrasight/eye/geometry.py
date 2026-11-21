"""Geometry stage: coat centimetres -> visual-field degrees on the working grid.

Physics: a source pixel of size ``cm_per_px`` at distance d subtends
    deg_per_px_src = degrees(atan(cm_per_px / 100 / d))
- finer than the working grid (eye.yaml ``working_grid``, 0.01 deg/px): area-average
  (anti-aliased) downsample of rgb and mask to the working grid;
- coarser (close range): keep the source grid and set ``source_limited`` = True, because
  the camera, not the eye, then limits what can be resolved.
Each pixel then gets visual-field coordinates: the crop centre sits at
(``cond.field_az_deg``, ``cond.field_el_deg``) plus pixel offsets x deg_per_px (elevation
positive up). These maps let the streak stage vary acuity across the image.

Hook (off in phase 1, ``geometry.foreshortening``): a flank viewed at
``cond.azimuth_view_deg`` away from side-on is compressed horizontally by cos(azimuth).
"""

from __future__ import annotations

import math

import torch

from zebrasight.eye.ops import area_resize, resize_mask
from zebrasight.registry import Stage, register_stage
from zebrasight.types import ViewState


def deg_per_px(cm_per_px: float, distance_m: float) -> float:
    return math.degrees(math.atan(cm_per_px / 100.0 / distance_m))


def field_maps(b: int, h: int, w: int, dpp: float, az0: float, el0: float, device: torch.device):
    """(B,1,H,W) azimuth and elevation maps in degrees, centred on (az0, el0)."""
    x = (torch.arange(w, device=device, dtype=torch.float32) - (w - 1) / 2) * dpp
    y = (torch.arange(h, device=device, dtype=torch.float32) - (h - 1) / 2) * dpp
    az = (az0 + x).view(1, 1, 1, w).expand(b, 1, h, w)
    el = (el0 - y).view(1, 1, h, 1).expand(b, 1, h, w)
    return az.contiguous(), el.contiguous()


def foreshorten(rgb: torch.Tensor, mask: torch.Tensor, azimuth_view_deg: float):
    """Compress width by cos(azimuth_view). Hook only: not applied in phase 1."""
    h, w = rgb.shape[-2:]
    new_w = max(1, round(w * math.cos(math.radians(azimuth_view_deg))))
    return area_resize(rgb, (h, new_w)), resize_mask(mask, (h, new_w))


@register_stage("geometry")
class Geometry(Stage):
    def forward(self, state: ViewState) -> ViewState:
        c = state.cond
        rgb, mask = state.rgb, state.mask
        grid = float(self.param("working_grid.deg_per_px"))
        do_foreshorten = bool(self.param("geometry.foreshortening"))
        if do_foreshorten and c.azimuth_view_deg:
            rgb, mask = foreshorten(rgb, mask, c.azimuth_view_deg)

        dpp_src = deg_per_px(state.cm_per_px, c.distance_m)
        h, w = mask.shape[-2:]
        if dpp_src < grid:
            new_h = max(1, round(h * dpp_src / grid))
            new_w = max(1, round(w * dpp_src / grid))
            rgb = area_resize(rgb, (new_h, new_w)) if rgb is not None else None
            mask = resize_mask(mask, (new_h, new_w))
            dpp = dpp_src * h / new_h  # exact angular pixel size after rounding the size
            source_limited, resampled = False, True
        else:
            dpp, source_limited, resampled = dpp_src, True, False

        b, _, h2, w2 = mask.shape
        az, el = field_maps(b, h2, w2, dpp, c.field_az_deg, c.field_el_deg, mask.device)
        state = state.evolve(rgb=rgb, mask=mask, deg_per_px=dpp, az_map=az, el_map=el,
                             source_limited=source_limited)
        self.record(state, deg_per_px_src=dpp_src, deg_per_px=dpp, working_grid_deg_per_px=grid,
                    resampled=resampled, size_in=(h, w), size_out=(h2, w2),
                    source_limited=source_limited, foreshortening=do_foreshorten)
        return state
