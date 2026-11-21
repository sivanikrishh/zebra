"""Atmosphere stage (no-op stub in phase 1).

Planned physics (Koschmieder model), per channel in linear RGB:
    I(d) = I0 * t(d) + A * (1 - t(d)),   t(d) = exp(-beta * d)
with ``beta = cond.beta_haze`` (1/m) and coloured airlight A (3-vector, sky colour).
Contrast falls by t(d); coloured airlight shifts the cone ratio, which is why colour must be
kept until the eye. Later: wavelength-dependent beta, heat shimmer at long range.
"""

from __future__ import annotations

from zebrasight.registry import Stage, register_stage
from zebrasight.types import ViewState


@register_stage("atmosphere")
class Atmosphere(Stage):
    def forward(self, state: ViewState) -> ViewState:
        self.record(state, stub=True, beta_haze=state.cond.beta_haze, distance_m=state.cond.distance_m)
        return state
