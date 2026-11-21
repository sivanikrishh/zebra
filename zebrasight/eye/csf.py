"""Contrast-sensitivity stage: global CSF on luminance, low-pass on chroma.

Luminance channel (L by day, rod at night), FFT of the reflect-padded image:
    S(f) = c * 10^(-((log10 f - log10 b) / w)^2)        log-parabola (Melin et al. 2016 form)
    b = b_peak_fraction * A,  c = c_peak_sensitivity,  A = regime acuity
    w = (log10 A - log10 b) / sqrt(log10 c)              so that S(A) = 1
Transfer H(f) = 1 for f <= b (flat top: shape and body outline are kept), S(f) / c above.
Visibility threshold: above b, components whose input contrast amplitude m(f) times S(f)
is below 1 are set to zero, with m(f) = 2 |F(f)| / |F(0)| (Michelson contrast of the
sinusoid). Above A, S < 1 and m <= 1, so everything beyond the acuity is removed.

Chroma channel (S cone): low-pass only, at the chromatic acuity
``colour.chroma_acuity_ratio`` x A, with a raised-cosine edge from 0.8x to 1x the cut-off
to limit ringing. No contrast threshold is applied (grade C: no horse chromatic CSF).

The spatial variation of acuity (visual streak) is handled by the streak stage; this
stage uses the regime's peak acuity A.
"""

from __future__ import annotations

import math

import torch

from zebrasight.eye.ops import fft_filter
from zebrasight.eye.streak import regime_acuity
from zebrasight.registry import Stage, register_stage
from zebrasight.types import ViewState


def csf_params(acuity: float, b_fraction: float, c_peak: float) -> tuple[float, float, float]:
    b = b_fraction * acuity
    w = (math.log10(acuity) - math.log10(b)) / math.sqrt(math.log10(c_peak))
    return b, c_peak, w


def csf_sensitivity(f_cpd: torch.Tensor, b: float, c: float, w: float) -> torch.Tensor:
    lf = torch.log10(f_cpd.clamp_min(1e-9))
    return c * torch.pow(10.0, -(((lf - math.log10(b)) / w) ** 2))


def luminance_csf(x: torch.Tensor, dpp: float, b: float, c: float, w: float, threshold: bool = True) -> torch.Tensor:
    def transfer(f_px, X):
        f = f_px / dpp  # cycles/deg
        s = csf_sensitivity(f, b, c, w)
        above = f > b
        h = torch.where(above, s / c, torch.ones_like(s))
        if threshold:
            dc = X[..., :1, :1].abs().clamp_min(1e-12)
            contrast = 2.0 * X.abs() / dc
            visible = ~above | (contrast * s >= 1.0)
        else:  # no per-component threshold: only zero beyond the acuity, where S < 1
            visible = f <= f.new_tensor(b) * 10 ** (w * math.sqrt(math.log10(c)))
        h = h * visible
        h[..., 0, 0] = 1.0
        return X * h

    return fft_filter(x, transfer)


def chroma_lowpass(x: torch.Tensor, dpp: float, cutoff_cpd: float) -> torch.Tensor:
    lo = 0.8 * cutoff_cpd

    def transfer(f_px, X):
        f = f_px / dpp
        t = ((f - lo) / (cutoff_cpd - lo)).clamp(0.0, 1.0)
        return X * (0.5 * (1.0 + torch.cos(math.pi * t)))

    return fft_filter(x, transfer)


@register_stage("csf")
class CSF(Stage):
    def forward(self, state: ViewState) -> ViewState:
        regime, acuity = regime_acuity(self, state.cond.light)
        b, c, w = csf_params(acuity, float(self.param("csf.b_peak_fraction")),
                             float(self.param("csf.c_peak_sensitivity")))
        dpp = state.deg_per_px
        ratio = float(self.param("colour.chroma_acuity_ratio"))
        threshold = bool(self.param("csf.contrast_threshold"))
        if state.rod is not None:
            state = state.evolve(rod=luminance_csf(state.rod, dpp, b, c, w, threshold))
        else:
            lum = luminance_csf(state.cones[:, 0:1], dpp, b, c, w, threshold)
            chroma = chroma_lowpass(state.cones[:, 1:2], dpp, ratio * acuity)
            state = state.evolve(cones=torch.cat([lum, chroma], dim=1))
        self.record(state, regime=regime, acuity_cpd=acuity, b_cpd=b, c_peak=c, width_log10=w,
                    contrast_threshold=threshold,
                    chroma_cutoff_cpd=None if state.rod is not None else ratio * acuity,
                    nyquist_cpd=0.5 / dpp)
        return state
