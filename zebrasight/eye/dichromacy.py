"""Dichromacy stage: linear RGB -> zebra cone signals [L, S], or rod at night.

Planned physics:
- cone fundamentals from Govardovskii et al. 2000 A1 templates at S 430 nm and
  L 545 nm (sweep 539), integrated against approximate camera RGB spectral sensitivities
  (photos are uncalibrated sRGB, grade C) to get a 3 -> 2 linear map;
- below the colour threshold (0.02 cd/m2, Roth et al. 2008) only rods respond
  (rhodopsin ~500 nm): output a single rod channel and no S channel;
- check: the L/S opponent signal must be ~0 near the 480 nm neutral point.

STUB: placeholder signals (L = channel mean, S = blue channel, rod = channel mean) so
shapes and the day/night switch are testable. The log marks them ``placeholder=True``.
"""

from __future__ import annotations

from zebrasight.registry import Stage, register_stage
from zebrasight.types import ViewState


@register_stage("dichromacy")
class Dichromacy(Stage):
    def forward(self, state: ViewState) -> ViewState:
        rgb = state.rgb
        if rgb is None:
            raise RuntimeError("dichromacy needs linear RGB; an earlier stage dropped it")
        threshold = self.param("colour.threshold_cd_m2")
        rod_only = state.cond.luminance_cd_m2 < threshold
        if rod_only:
            state = state.evolve(rgb=None, rod=rgb.mean(dim=1, keepdim=True), cones=None)
        else:
            cones = rgb.new_empty(rgb.shape[0], 2, *rgb.shape[-2:])
            cones[:, 0:1] = rgb.mean(dim=1, keepdim=True)
            cones[:, 1:2] = rgb[:, 2:3]
            state = state.evolve(rgb=None, cones=cones, rod=None)
        self.record(state, placeholder=True, rod_only=rod_only, colour_threshold_cd_m2=threshold,
                    luminance_cd_m2=state.cond.luminance_cd_m2,
                    S_peak_nm=self.param("cones.S_peak"), L_peak_nm=self.param("cones.L_peak"))
        return state
