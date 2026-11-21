"""Contrast-sensitivity stage: spatial filtering of the cone/rod signals.

Planned physics (Melin et al. 2016 method): log-parabola CSF
    log10 S(f) = log10 c - a * (log10 f - log10 b)^2
with peak frequency b, peak sensitivity c and slope a. Constraint S(acuity) = 1 fixes a
given b and c. b = 0.10 x acuity (sweep 0.08-0.15) and c = 50 (sweep 20-100) are grade C
because Melin did not report them. The filter is applied per pixel using the local rho
map from the streak stage. The S (chromatic) channel is low-passed at
``colour.chroma_acuity_ratio`` x luminance acuity (0.5, sweep 0.3-1.0, grade C).

STUB: pass-through.
"""

from __future__ import annotations

from zebrasight.registry import Stage, register_stage
from zebrasight.types import ViewState


@register_stage("csf")
class CSF(Stage):
    def forward(self, state: ViewState) -> ViewState:
        self.record(state, stub=True, b_peak_fraction=self.param("csf.b_peak_fraction"),
                    c_peak_sensitivity=self.param("csf.c_peak_sensitivity"),
                    chroma_acuity_ratio=self.param("colour.chroma_acuity_ratio"))
        return state
