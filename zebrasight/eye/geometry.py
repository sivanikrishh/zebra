"""Geometry stage: coat centimetres -> visual-field degrees on the working grid.

Physics: a source pixel of size ``cm_per_px`` at distance d subtends
    deg_per_px_src = degrees(atan(cm_per_px / 100 / d))
- finer than the working grid (eye.yaml ``working_grid``, 0.01 deg/px): Gaussian pre-filtered
  (anti-aliased) downsample of rgb and mask to the working grid;
- coarser (close range): keep the source grid and set ``source_limited`` = True, because
  the camera, not the eye, then limits what can be resolved.
Each pixel then gets visual-field coordinates: the crop centre sits at
(``cond.field_az_deg``, ``cond.field_el_deg``) plus pixel offsets x deg_per_px (elevation
positive up). These maps let the streak stage vary acuity across the image.

Working-size cap (``working_grid.max_side``, 512): if the working image's longer side is larger
(close range), it is coarsened to fit and ``display_limited`` = True; the log records the
effective resolvable cpd, 1 / (2.5 deg_per_px). This bounds time and memory per percept.

Hook (off in phase 1, ``geometry.foreshortening``): a flank viewed at
``cond.azimuth_view_deg`` away from side-on is compressed horizontally by cos(azimuth).
"""

from __future__ import annotations

import math

import torch

from zebrasight.eye.ops import gaussian_resize, resize_mask
from zebrasight.eye.streak import LIGHT_TO_REGIME
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
    return gaussian_resize(rgb, (h, new_w)), resize_mask(mask, (h, new_w))


@register_stage("geometry")
class Geometry(Stage):
    def forward(self, state: ViewState) -> ViewState:
        c = state.cond
        rgb, mask = state.rgb, state.mask
        grid = float(self.param("working_grid.deg_per_px"))
        do_foreshorten = bool(self.param("geometry.foreshortening"))
        if do_foreshorten and c.azimuth_view_deg:
            rgb, mask = foreshorten(rgb, mask, c.azimuth_view_deg)

        # regime-adaptive working grid: no finer than samples_per_cycle px per acuity cycle
        # (day: 0.01 deg/px unchanged; dusk ~1.6x and night ~12x coarser, same filtering margin)
        acuity = float(self.param(f"acuity.{LIGHT_TO_REGIME[c.light]}"))
        grid = max(grid, 1.0 / (float(self.param("working_grid.samples_per_cycle")) * acuity))
        dpp_src = deg_per_px(state.cm_per_px, c.distance_m)
        h, w = mask.shape[-2:]
        source_limited = dpp_src >= float(self.param("working_grid.deg_per_px"))
        target = max(dpp_src, grid)  # never upsample a coarse source
        # working-size cap (speed, memory): coarsen the target grid if the longer side would exceed it
        max_side = int(self.param("working_grid.max_side"))
        display_limited = max(h, w) * dpp_src / target > max_side
        if display_limited:
            target = max(h, w) * dpp_src / max_side
        k = dpp_src / target  # <= 1
        size = (max(1, round(h * k)), max(1, round(w * k)))
        resampled = size != (h, w)
        if resampled:  # ONE resampling step: two Gaussian pre-filters would compound near-limit losses
            rgb = gaussian_resize(rgb, size) if rgb is not None else None
            mask = resize_mask(mask, size)
        dpp = dpp_src * h / size[0]  # exact angular pixel size after rounding the size

        b, _, h2, w2 = mask.shape
        az, el = field_maps(b, h2, w2, dpp, c.field_az_deg, c.field_el_deg, mask.device)
        state = state.evolve(rgb=rgb, mask=mask, deg_per_px=dpp, az_map=az, el_map=el,
                             source_limited=source_limited, display_limited=display_limited)
        self.record(state, deg_per_px_src=dpp_src, deg_per_px=dpp, working_grid_deg_per_px=grid,
                    resampled=resampled, size_in=(h, w), size_out=(h2, w2),
                    source_limited=source_limited, foreshortening=do_foreshorten,
                    working_max_side=max_side, display_limited=display_limited,
                    effective_cpd=(1.0 / (2.5 * dpp)) if display_limited else None)
        return state
