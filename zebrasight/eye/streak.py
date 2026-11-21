"""Visual-streak stage: per-pixel resolution limit rho (cycles/deg).

Planned physics: horse acuity is limited by ganglion-cell density in a horizontal visual
streak at elevation 0 (Timney & Keil 1992; peak 4000 cells/mm2, Guo & Sugita 2000).
    rho(az, el) = A_regime * max(floor, g(el) * f(az))
    g(el) = exp(-el^2 / (2 sigma_el^2))         sigma_el = 5 deg (sweep 3-10, grade C)
    f(az) = 1 - (1 - falloff90) * |az| / 90     falloff90 = 0.5 (sweep 0.3-0.7, grade C)
    floor = 0.16 (periphery 2.7 / streak 16.5 cpd, Harman et al. 1999)
A_regime is the photopic / mesopic / scotopic acuity chosen from ``cond.light``.

STUB: rho is uniform at A_regime (no streak falloff yet).
"""

from __future__ import annotations

import torch

from zebrasight.registry import Stage, register_stage
from zebrasight.types import ViewState

LIGHT_TO_REGIME = {"day": "photopic", "dawn": "mesopic", "dusk": "mesopic", "night": "scotopic"}


@register_stage("streak")
class Streak(Stage):
    def forward(self, state: ViewState) -> ViewState:
        if state.az_map is None:
            raise RuntimeError("streak needs az/el maps; run the geometry stage first")
        regime = LIGHT_TO_REGIME[state.cond.light]
        acuity = float(self.param(f"acuity.{regime}"))
        rho = torch.full_like(state.az_map, acuity)
        state = state.evolve(rho=rho)
        self.record(state, stub=True, regime=regime, acuity_cpd=acuity)
        return state
