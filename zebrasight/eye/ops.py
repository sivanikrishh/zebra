"""Tensor helpers shared by the eye stages: FFT filtering, resampling, Otsu, box mean.

All inputs are (B, C, H, W) float32 on any device.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def _pad_amount(n: int) -> int:
    return max(0, min(n - 1, max(8, n // 4)))


def reflect_pad(x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
    """Reflect-pad H and W so FFT filtering does not wrap edges; returns (padded, (ph, pw))."""
    ph, pw = _pad_amount(x.shape[-2]), _pad_amount(x.shape[-1])
    return F.pad(x, (pw, pw, ph, ph), mode="reflect"), (ph, pw)


def unpad(x: torch.Tensor, pads: tuple[int, int]) -> torch.Tensor:
    ph, pw = pads
    return x[..., ph : x.shape[-2] - ph, pw : x.shape[-1] - pw]


def radial_freq(h: int, w: int, device: torch.device) -> torch.Tensor:
    """(h, w//2+1) radial spatial frequency in cycles/px for an rfft2 of an (h, w) image."""
    fy = torch.fft.fftfreq(h, device=device)
    fx = torch.fft.rfftfreq(w, device=device)
    return torch.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2)


def fft_filter(x: torch.Tensor, transfer) -> torch.Tensor:
    """Apply a radial transfer function ``transfer(f_cyc_per_px, X) -> X'`` in the Fourier domain."""
    xp, pads = reflect_pad(x)
    h, w = xp.shape[-2:]
    X = torch.fft.rfft2(xp)
    X = transfer(radial_freq(h, w, x.device), X)
    return unpad(torch.fft.irfft2(X, s=(h, w)), pads)


def gaussian_blur_fft(x: torch.Tensor, sigma_px: float) -> torch.Tensor:
    """Gaussian blur of any width via FFT. sigma_px <= 0 returns x unchanged."""
    if sigma_px <= 0:
        return x
    k = 2.0 * (math.pi * sigma_px) ** 2
    return fft_filter(x, lambda f, X: X * torch.exp(-k * f**2))


def fft_conv(x: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """Same-size convolution of every channel with a centred 2-D kernel (odd size)."""
    kh, kw = kernel.shape
    ph, pw = kh // 2, kw // 2
    xp = F.pad(x, (pw, pw, ph, ph), mode="replicate")
    h, w = xp.shape[-2:]
    kpad = torch.zeros(h, w, device=x.device, dtype=x.dtype)
    kpad[:kh, :kw] = kernel
    kpad = torch.roll(kpad, shifts=(-ph, -pw), dims=(0, 1))
    y = torch.fft.irfft2(torch.fft.rfft2(xp) * torch.fft.rfft2(kpad), s=(h, w))
    return y[..., ph : ph + x.shape[-2], pw : pw + x.shape[-1]]


def area_resize(x: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    """Anti-aliased downsample by area averaging."""
    if tuple(x.shape[-2:]) == tuple(size):
        return x
    return F.interpolate(x, size=size, mode="area")


def is_binary(mask: torch.Tensor) -> bool:
    return bool(((mask == 0) | (mask == 1)).all())


def resize_mask(mask: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    """Binary masks: area average then threshold at 0.5 (keeps area, stays binary).
    Label masks (face/flank/rump codes): nearest-exact, so labels are never mixed."""
    if tuple(mask.shape[-2:]) == tuple(size):
        return mask
    if is_binary(mask):
        return (F.interpolate(mask, size=size, mode="area") >= 0.5).to(mask.dtype)
    return F.interpolate(mask, size=size, mode="nearest-exact")


def otsu_threshold(x: torch.Tensor, weight: torch.Tensor | None = None, bins: int = 256) -> torch.Tensor:
    """Per-image Otsu threshold, (B,) tensor. ``weight`` (B,1,H,W) selects pixels (e.g. mask)."""
    b = x.shape[0]
    flat = x.reshape(b, -1)
    wts = torch.ones_like(flat) if weight is None else (weight.reshape(b, -1) > 0).to(flat.dtype)
    wts = torch.where(wts.sum(1, keepdim=True) > 0, wts, torch.ones_like(wts))
    big = torch.finfo(flat.dtype).max
    lo = torch.where(wts > 0, flat, torch.full_like(flat, big)).min(1).values
    hi = torch.where(wts > 0, flat, torch.full_like(flat, -big)).max(1).values
    span = (hi - lo).clamp_min(1e-12)
    idx = ((flat - lo[:, None]) / span[:, None] * (bins - 1)).round().clamp(0, bins - 1).long()
    hist = torch.zeros(b, bins, device=x.device, dtype=flat.dtype).scatter_add_(1, idx, wts)
    p = hist / hist.sum(1, keepdim=True)
    centres = torch.arange(bins, device=x.device, dtype=flat.dtype)
    w0 = p.cumsum(1)
    m0 = (p * centres).cumsum(1)
    mt = m0[:, -1:]
    between = (mt * w0 - m0) ** 2 / (w0 * (1 - w0)).clamp_min(1e-12)
    k = between[:, :-1].argmax(1).to(flat.dtype)
    return lo + (k + 0.5) / (bins - 1) * span


def box_mean(x: torch.Tensor, size_px: int) -> torch.Tensor:
    """Box average with an odd window, edges handled by replicate padding."""
    size_px = max(1, int(size_px) | 1)
    if size_px == 1:
        return x
    r = size_px // 2
    xp = F.pad(x, (r, r, r, r), mode="replicate")
    return F.avg_pool2d(xp, size_px, stride=1)
