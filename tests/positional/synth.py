"""Synthetic zebra-like test images: wavy dark/light stripes with a known period, angle and phase on an
elliptical body mask whose geometry (and so the canonical frame) is known."""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

from zebrasight.positional.frame import fit_frame
from zebrasight.positional.observers.base import View

H, W = 240, 480
CX, CY, A_AX, B_AX = 240.0, 120.0, 220.0, 100.0   # ellipse: body length 440 px -> stripe period 440/24 px
BL = 2 * A_AX
PERIOD = BL / 24.0


def ellipse(h=H, w=W, cx=CX, cy=CY, a=A_AX, b=B_AX, angle_deg=0.0) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w].astype(float)
    t = np.radians(angle_deg)
    u = (xx - cx) * np.cos(t) + (yy - cy) * np.sin(t)
    v = -(xx - cx) * np.sin(t) + (yy - cy) * np.cos(t)
    return (u / a) ** 2 + (v / b) ** 2 <= 1.0


def stripes(h=H, w=W, period=PERIOD, angle_deg=0.0, phase=0.0, wobble=0.15, wobble_len=None, sharp=3.0,
            seed: int | None = None) -> np.ndarray:
    """Stripes whose wave vector points at angle_deg (0 = vertical stripes); wobble = lateral waviness as a
    fraction of the period; seed adds a random smooth warp (a different 'individual')."""
    yy, xx = np.mgrid[0:h, 0:w].astype(float)
    t = np.radians(angle_deg)
    s = xx * np.cos(t) + yy * np.sin(t)          # across the stripes
    along = -xx * np.sin(t) + yy * np.cos(t)
    wl = wobble_len or 4 * period
    s = s + wobble * period * np.sin(2 * np.pi * along / wl)
    if seed is not None:
        rng = np.random.default_rng(seed)
        s = s + 1.5 * period * ndi.gaussian_filter(rng.standard_normal((h, w)), 2 * period, mode="reflect") \
            / max(1e-9, ndi.gaussian_filter(rng.standard_normal((h, w)), 2 * period, mode="reflect").std())
    return (0.5 + 0.5 * np.tanh(sharp * np.cos(2 * np.pi * s / period + phase))).astype(np.float32)


def view(y: np.ndarray, mask: np.ndarray, head_left: bool = True) -> View:
    fr = fit_frame(mask, (0.0, 0.0), head_left)
    return View(y=y.astype(np.float32), mask=mask, bl_px=fr.length_px, px2can=fr.A, deg_per_px=0.01,
                distance_m=10.0, region="flank", cond="c0")
