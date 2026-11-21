"""Scene stage (no-op stub in phase 1).

Planned physics, all in linear RGB (3 channels):
- composite the coat crop onto a background (grass, sky) using the region mask;
- illumination: sun/sky ratio from the light regime, cast shadows on the body;
- body orientation (``cond.azimuth_view_deg``): foreshortening of the flank by
  cos(azimuth) along the horizontal axis, resampling image and mask together.
"""

from __future__ import annotations

from zebrasight.registry import Stage, register_stage
from zebrasight.types import ViewState


@register_stage("scene")
class Scene(Stage):
    def forward(self, state: ViewState) -> ViewState:
        self.record(state, stub=True, azimuth_view_deg=state.cond.azimuth_view_deg)
        return state
