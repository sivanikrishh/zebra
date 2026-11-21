"""Visual-streak stage: per-pixel acuity map and spatially varying blur.

Physics: horse acuity is limited by ganglion-cell density in a horizontal visual streak at
elevation 0 (Timney & Keil 1992; peak 4000 cells/mm2, Guo & Sugita 2000):
    acuity_map(az, el) = A * max(floor, g(el) * f(az))
    g(el) = exp(-el^2 / (2 sigma_el^2))                     sigma_el 5 deg (sweep 3-10, C)
    f(az) = max(0, 1 - (1 - falloff90) * |az| / 90)         falloff90 0.5 (sweep 0.3-0.7, C)
    floor = 0.16 (periphery 2.7 / streak 16.5 cpd, Harman et al. 1999)
A is the regime acuity (photopic / mesopic / scotopic) chosen from ``cond.light``.

With csf.mode = "peli" (default) this stage only sets ``acuity_map``; the csf stage applies
it per pixel and frequency band. With csf.mode = "global" it also blurs, as follows.

Blur (global mode): a Gaussian stack (a full-resolution Gaussian pyramid, built by FFT so any blur width
is cheap). Level 0 is the input; level k >= 1 is a Gaussian whose transfer is 0.5 at
A / 2^k. Each pixel blends the two levels around l = log2(A / acuity_map), so on the streak
centre (acuity_map = A) nothing changes and the csf stage alone sets the cut-off at A.
This stage therefore removes only the loss relative to the peak; together with the csf
stage the effective cut-off is acuity_map for luminance. The chroma (S) channel gets the
same relative blend, and the csf stage low-passes it at 0.5 x A, so its effective
cut-off is 0.5 x acuity_map (``colour.chroma_acuity_ratio``). Blurring S to 0.5 x the
map here as well would filter it twice.
"""

from __future__ import annotations

import math

import torch

from zebrasight.eye.ops import gaussian_blur_fft
from zebrasight.registry import Stage, register_stage
from zebrasight.types import ViewState

LIGHT_TO_REGIME = {"day": "photopic", "dawn": "mesopic", "dusk": "mesopic", "night": "scotopic"}


def regime_acuity(stage: Stage, light: str) -> tuple[str, float]:
    regime = LIGHT_TO_REGIME[light]
    return regime, float(stage.param(f"acuity.{regime}"))


def acuity_map(az: torch.Tensor, el: torch.Tensor, peak: float, sigma_el: float, falloff90: float,
               floor: float) -> torch.Tensor:
    g = torch.exp(-(el**2) / (2.0 * sigma_el**2))
    f = (1.0 - (1.0 - falloff90) * az.abs().clamp(max=180.0) / 90.0).clamp_min(0.0)
    return peak * torch.clamp(g * f, min=floor)


def blend_stack(x: torch.Tensor, level: torch.Tensor, peak: float, dpp: float, n_levels: int) -> torch.Tensor:
    """Per-pixel blend of a Gaussian stack. ``level`` (B,1,H,W) in [0, n_levels]."""
    out = x
    for k in range(1, n_levels + 1):
        w = (level - (k - 1)).clamp(0.0, 1.0)  # 0 below level k-1, 1 at level k and above
        if float(w.max()) <= 1e-6:
            break
        f_half = peak / 2**k  # cpd at which this level's transfer is 0.5
        sigma_deg = math.sqrt(math.log(2.0) / 2.0) / (math.pi * f_half)
        out = torch.lerp(out, gaussian_blur_fft(x, sigma_deg / dpp), w)
    return out


@register_stage("streak")
class Streak(Stage):
    def forward(self, state: ViewState) -> ViewState:
        if state.az_map is None:
            raise RuntimeError("streak needs az/el maps; run the geometry stage first")
        regime, peak = regime_acuity(self, state.cond.light)
        sigma_el = float(self.param("streak.sigma_el_deg"))
        falloff90 = float(self.param("streak.az_falloff_at_90"))
        floor = float(self.param("streak.periphery_floor"))
        amap = acuity_map(state.az_map, state.el_map, peak, sigma_el, falloff90, floor)
        n_levels = math.ceil(math.log2(1.0 / floor))
        # In peli mode the csf stage applies acuity_map per pixel and band; blurring here too
        # would filter twice. The Gaussian stack is only used with the global CSF.
        blur = str(self.param("csf.mode")) == "global"
        if blur:
            level = torch.log2(peak / amap).clamp(0.0, float(n_levels))
            if state.rod is not None:
                state = state.evolve(rod=blend_stack(state.rod, level, peak, state.deg_per_px, n_levels))
            else:
                state = state.evolve(cones=blend_stack(state.cones, level, peak, state.deg_per_px, n_levels))
        state = state.evolve(acuity_map=amap)
        self.record(state, regime=regime, acuity_cpd=peak, sigma_el_deg=sigma_el, az_falloff_at_90=falloff90,
                    periphery_floor=floor, n_levels=n_levels, stack_blur=blur,
                    acuity_map_min=float(amap.min()), acuity_map_max=float(amap.max()))
        return state
