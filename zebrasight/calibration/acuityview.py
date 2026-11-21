"""AcuityView MTF (Caves & Johnsen 2018, Methods Ecol Evol 9:793) re-implemented for comparison.

From the AcuityView R source (function AcuityView):
    widthInDegrees  = 57.2958 * 2 * atan(realWidth / distance / 2)
    pixelsPerDegree = widthInPixels / widthInDegrees
    freq  = round(sqrt(x^2 + y^2)) / widthInPixels * pixelsPerDegree      # cycles/degree,
                                                                           # radius rounded in cycles/image
    eyeResolution = R_x R_y / sqrt((R_y cos)^2 + (R_x sin)^2)              # = R when R_x = R_y
    MTF = exp(-3.56 * (eyeResolution * freq)^2),  MTF(0) = 1
eyeResolution is the minimum resolvable angle in degrees, R = 1 / acuity (cpd). So
MTF(f) = exp(-3.56 (f / A)^2): 0.028 at the acuity, nonzero above it.
The R code multiplies the fftshift-ed spectrum and takes Mod() of the inverse without
shifting back; here the MTF is applied to the unshifted spectrum and the real part is kept,
which is the intended operation. AcuityView requires square images (it uses one width).
"""

from __future__ import annotations

import math

import torch


def acuityview_mtf_cpd(f_cpd: torch.Tensor, acuity_cpd: float) -> torch.Tensor:
    return torch.exp(-3.56 * (f_cpd / acuity_cpd) ** 2)


def acuityview_filter(img: torch.Tensor, dpp: float, acuity_cpd: float) -> torch.Tensor:
    """img (B,C,N,N) linear; dpp degrees per pixel. AcuityView's rounded radial frequency grid."""
    n = img.shape[-1]
    if img.shape[-2] != n:
        raise ValueError("AcuityView works on square images")
    k = torch.fft.fftfreq(n, d=1.0 / n, device=img.device)  # integer cycles/image
    radius = torch.round(torch.sqrt(k[:, None] ** 2 + k[None, :] ** 2))
    f_cpd = radius / n / dpp
    mtf = acuityview_mtf_cpd(f_cpd, acuity_cpd)
    mtf[0, 0] = 1.0
    return torch.fft.ifft2(torch.fft.fft2(img) * mtf).real


def compare_curves(acuity: float, b_frac: float, c_peak: float, ratios=(0.05, 0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5)):
    from zebrasight.eye.csf import csf_params, csf_transfer

    b, c, w = csf_params(acuity, b_frac, c_peak)
    f = torch.tensor([r * acuity for r in ratios], dtype=torch.float64)
    ours = csf_transfer(f, acuity, b, c, w, taper=0.02)
    theirs = acuityview_mtf_cpd(f, acuity)
    return [(r, float(o), float(t)) for r, o, t in zip(ratios, ours, theirs)]


def ours_acuity_only(img: torch.Tensor, dpp: float, acuity: float, b_frac: float, c_peak: float, taper: float):
    from zebrasight.eye.csf import csf_params, csf_transfer
    from zebrasight.eye.ops import fft_filter

    b, c, w = csf_params(acuity, b_frac, c_peak)
    return fft_filter(img, lambda f_px, X: X * csf_transfer(f_px / dpp, acuity, b, c, w, taper))


__all__ = ["acuityview_filter", "acuityview_mtf_cpd", "compare_curves", "ours_acuity_only", "math"]
