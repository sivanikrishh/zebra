"""Atmosphere stage: haze and dust along the viewing path (Koschmieder), in LINEAR RGB.

Per channel c (sRGB R, G, B with effective wavelengths 611 / 549 / 464 nm):
    I_c(d) = I0_c * t_c(d) + A_c * (1 - t_c(d)),     t_c(d) = exp(-beta_c * d)
    beta_c = beta_550 * (lambda_c / 550 nm)^(-alpha)          (Angstrom law)
beta_550 = ``cond.beta_haze`` (1/m); alpha and the airlight colour A come from the aerosol type
(``cond.aerosol``: haze alpha 1.3 bluish-white, dust alpha 0.2 yellow-brown; configs/env.yaml).

Contrast against the horizon sky (the airlight), C = (I - A) / A, falls exactly as
C(d) = C0 exp(-beta d): Koschmieder visual range V = 3.912 / beta (2% threshold), WMO
meteorological optical range MOR = 2.996 / beta (5%). Both are logged.

Airlight brightness: images are relative (a sunlit white ~ 1), and the scene and its airlight
are both lit by the same sky, so A is relative too and the haze/scene ratio does not change with
the light regime. The ABSOLUTE airlight luminance (relative A x regime luminance, cd/m2) is
logged and is 100x dimmer at dusk; the photon-noise stage already works in absolute units.

aerosol = "clear" (c0 and every condition without haze) leaves the image untouched, so the clear
reference is unchanged (``env.atmosphere.apply_to_clear``).
"""

from __future__ import annotations

import math

import torch

from zebrasight.registry import Stage, register_stage
from zebrasight.types import ViewState

SRGB_Y = (0.2126, 0.7152, 0.0722)


def channel_betas(beta_550: float, alpha: float, wavelengths_nm, ref_nm: float = 550.0) -> list[float]:
    return [beta_550 * (wl / ref_nm) ** (-alpha) for wl in wavelengths_nm]


def airlight_rgb(colour, luminance: float) -> list[float]:
    """Scale a relative linear-RGB colour so that its sRGB luminance Y equals ``luminance``."""
    y = sum(w * c for w, c in zip(SRGB_Y, colour))
    return [c * luminance / y for c in colour]


def koschmieder(rgb: torch.Tensor, d: float, betas, airlight) -> torch.Tensor:
    t = torch.tensor([math.exp(-b * d) for b in betas], dtype=rgb.dtype, device=rgb.device).view(1, 3, 1, 1)
    a = torch.tensor(airlight, dtype=rgb.dtype, device=rgb.device).view(1, 3, 1, 1)
    return rgb * t + a * (1.0 - t)


def visual_range_m(beta: float, threshold: float) -> float:
    return math.log(1.0 / threshold) / beta if beta > 0 else math.inf


@register_stage("atmosphere")
class Atmosphere(Stage):
    def forward(self, state: ViewState) -> ViewState:
        c = state.cond
        if c.aerosol == "clear" and not bool(self.param("env.atmosphere.apply_to_clear")):
            self.record(state, applied=False, aerosol="clear", beta_550=c.beta_haze)
            return state
        if c.aerosol not in ("haze", "dust", "clear"):
            raise ValueError(f"unknown aerosol {c.aerosol!r}")
        kind = "haze" if c.aerosol == "clear" else c.aerosol
        alpha = float(self.param(f"env.atmosphere.aerosols.{kind}.alpha"))
        colour = list(self.param(f"env.atmosphere.aerosols.{kind}.airlight_rgb"))
        lum = float(self.param(f"env.atmosphere.aerosols.{kind}.airlight_luminance"))
        wls = list(self.param("env.atmosphere.wavelengths_nm"))
        betas = channel_betas(c.beta_haze, alpha, wls, float(self.param("env.atmosphere.reference_nm")))
        a = airlight_rgb(colour, lum)
        rgb = koschmieder(state.rgb, c.distance_m, betas, a)
        state = state.evolve(rgb=rgb)
        self.record(state, applied=True, aerosol=c.aerosol, beta_550=c.beta_haze, alpha=alpha,
                    beta_rgb=betas, transmittance_rgb=[math.exp(-b * c.distance_m) for b in betas],
                    airlight_rgb_rel=a, airlight_luminance_rel=lum,
                    airlight_luminance_cd_m2=lum * c.luminance_cd_m2,
                    koschmieder_range_m=visual_range_m(c.beta_haze, float(self.param("env.atmosphere.koschmieder_threshold"))),
                    mor_m=visual_range_m(c.beta_haze, float(self.param("env.atmosphere.mor_threshold"))))
        return state
