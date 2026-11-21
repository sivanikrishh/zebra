"""Motion-blur stage: temporal integration of a moving target.

Planned physics: a target moving at ``cond.speed_mps`` across the line of sight at distance d
sweeps an angle  theta = degrees(speed * T / d)  during the integration time T (day 30 ms,
sweep 15-60; night 100 ms; grade C, no horse flicker-fusion value found). Blur = box kernel
of length theta / deg_per_px pixels along the motion direction.

STUB: pass-through.
"""

from __future__ import annotations

from zebrasight.registry import Stage, register_stage
from zebrasight.types import ViewState


@register_stage("motion")
class Motion(Stage):
    def forward(self, state: ViewState) -> ViewState:
        key = "temporal.integration_ms_night" if state.cond.light == "night" else "temporal.integration_ms_day"
        self.record(state, stub=True, speed_mps=state.cond.speed_mps, integration_ms=self.param(key))
        return state
