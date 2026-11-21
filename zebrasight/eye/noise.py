"""Noise stage: photon and neural noise.

Planned physics: Poisson photon noise whose mean photon count scales with
``cond.luminance_cd_m2`` x integration time x receptor aperture, then additive Gaussian
neural noise (grade C). The large-field Weber fraction 0.42-0.45 (Geisbauer et al. 2004)
bounds the noise sweep from above. Seeded from ``cond.seed`` with a torch.Generator so runs
are reproducible.

STUB: pass-through.
"""

from __future__ import annotations

from zebrasight.registry import Stage, register_stage
from zebrasight.types import ViewState


@register_stage("noise")
class Noise(Stage):
    def forward(self, state: ViewState) -> ViewState:
        self.record(state, stub=True, model=self.param("noise.model"), seed=state.cond.seed)
        return state
