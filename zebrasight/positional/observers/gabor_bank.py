"""C1 complex Gabor filter bank in coat units (cycles per body length), on GPU or CPU (torch).

Filters are applied in the frequency domain (exactly a complex Gabor: a Gaussian centred on the wave
vector f0 (cos t, sin t), one side only, so the response is analytic). Bandwidth b octaves sets the
spatial sigma: sigma = (lambda / pi) sqrt(ln 2 / 2) (2^b + 1) / (2^b - 1). Orientation t is the direction
of the wave vector (across the stripes). The image is mean-subtracted and divided by its standard
deviation inside the mask (contrast-invariant magnitudes), zero outside, zero-padded by 3 sigma.

Precision: the FFTs run in float32 (cuFFT half precision needs power-of-two sizes and loses accuracy on
the large low-frequency kernels); half=True returns float16 real/imag parts for storage and transfer.
"""

from __future__ import annotations

import math

import numpy as np
import torch


def sigma_px(wavelength_px: float, bw_oct: float) -> float:
    return wavelength_px / math.pi * math.sqrt(math.log(2) / 2) * (2**bw_oct + 1) / (2**bw_oct - 1)


def wavelengths(bl_px: float, scales_cpbl, min_wavelength_px: float) -> list[tuple[float, float | None]]:
    """[(cpbl, wavelength px or None if finer than the grid allows)]."""
    out = []
    for f in scales_cpbl:
        lam = bl_px / float(f)
        out.append((float(f), lam if lam >= min_wavelength_px else None))
    return out


def device_of(device: str | None) -> torch.device:
    if device in (None, "auto"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def normalise(y: np.ndarray, mask: np.ndarray) -> np.ndarray:
    v = y[mask]
    if v.size < 2 or v.std() < 1e-9:
        return np.zeros_like(y, np.float32)
    return np.where(mask, (y - v.mean()) / v.std(), 0.0).astype(np.float32)


def responses(y: np.ndarray, mask: np.ndarray, wavelengths_px: list[float], n_orient: int, bw_oct: float,
              device: str | None = None, half: bool = False):
    """-> complex array (S, O, H, W) (numpy complex64; or float16 (S, O, 2, H, W) if half)."""
    dev = device_of(device)
    x = torch.from_numpy(normalise(y, mask)).to(dev)
    H, W = x.shape
    if not wavelengths_px:
        return np.zeros((0, n_orient, H, W), np.complex64)
    pad = int(math.ceil(3 * sigma_px(max(wavelengths_px), bw_oct)))
    pad = min(pad, 2 * max(H, W))
    Hp, Wp = H + 2 * pad, W + 2 * pad
    X = torch.fft.fft2(torch.nn.functional.pad(x, (pad, pad, pad, pad)))
    fy = torch.fft.fftfreq(Hp, device=dev)[:, None]
    fx = torch.fft.fftfreq(Wp, device=dev)[None, :]
    out = []
    for lam in wavelengths_px:
        f0, s = 1.0 / lam, sigma_px(lam, bw_oct)
        row = []
        for k in range(n_orient):
            t = math.pi * k / n_orient
            G = torch.exp(-2 * math.pi**2 * s**2 * ((fx - f0 * math.cos(t)) ** 2 + (fy - f0 * math.sin(t)) ** 2))
            r = torch.fft.ifft2(X * G)[pad:pad + H, pad:pad + W]
            row.append(r)
        out.append(torch.stack(row))
    R = torch.stack(out)
    if half:
        return torch.stack([R.real, R.imag], dim=2).half().cpu().numpy()
    return R.cpu().numpy().astype(np.complex64)
