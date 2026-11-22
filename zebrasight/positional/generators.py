"""Part D generators: controlled edits of a percept INSIDE its region mask (outside is left unchanged).

All sizes are in coat units (fraction of body length, cycles per body length, fraction of the stripe
period) and converted with the View's bl_px. Random generators take an explicit seed (deterministic).

D1 band_filter       keep only / remove one frequency band (cycles per body length).
D2 patch_shuffle     permute square tiles lying fully inside the mask (tile = fraction of body length).
D3 phase_scramble    random Fourier phase, same amplitude spectrum (power spectrum kept exactly).
D4 silhouette        mask filled with uniform coat, or with phase-scrambled stripes.
D5 occlusions        grey patches (size/stride as fraction of body length) filled with the local mean.
D6 morph_positional  smooth random displacement, RMS amplitude as a fraction of the stripe period.
   morph_style       stripe width (grey morphology, symmetric about stripe centres) and contrast.
D7 transplant        a body part from a donor pasted into the recipient through the canonical frames
                     (or a stored pair alignment), luminance-matched, feathered seam; donor = recipient
                     is the "own region re-pasted" control.
"""

from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage as ndi

from zebrasight.positional.frame import apply, canonical_points, compose, invert, part_mask
from zebrasight.positional.observers.base import View, cfg


def _filled(y, mask):
    m = float(y[mask].mean()) if mask.any() else 0.0
    return np.where(mask, y, m).astype(np.float64), m


def _freq_radius(shape):
    fy = np.fft.fftfreq(shape[0])[:, None]
    fx = np.fft.fftfreq(shape[1])[None, :]
    return np.hypot(fx, fy)


def band_filter(y, mask, bl_px: float, band_cpbl, keep: bool = True):
    """D1: keep (or remove) radial frequencies in [lo, hi) cycles per body length; the mean is kept."""
    x, m = _filled(y, mask)
    F = np.fft.fft2(x - m)
    r = _freq_radius(x.shape) * bl_px              # cycles per body length
    inband = (r >= band_cpbl[0]) & (r < band_cpbl[1])
    out = np.real(np.fft.ifft2(F * (inband if keep else ~inband))) + m
    return np.where(mask, out, y).astype(np.float32)


def patch_shuffle(y, mask, bl_px: float, tile_bl: float, seed: int):
    """D2: -> (image, tiles [(r, c)], permutation). Only tiles fully inside the mask move."""
    s = max(2, int(round(tile_bl * bl_px)))
    rows, cols = np.nonzero(mask)
    if rows.size == 0:
        return y.copy(), [], np.zeros(0, int)
    r0, c0 = rows.min(), cols.min()
    tiles = [(r, c) for r in range(r0, rows.max() - s + 2, s) for c in range(c0, cols.max() - s + 2, s)
             if mask[r:r + s, c:c + s].all() and r + s <= y.shape[0] and c + s <= y.shape[1]]
    perm = np.random.default_rng(seed).permutation(len(tiles))
    out = y.copy()
    for (r, c), k in zip(tiles, perm):
        rr, cc = tiles[k]
        out[r:r + s, c:c + s] = y[rr:rr + s, cc:cc + s]
    return out, tiles, perm


def phase_scramble(y, mask, seed: int, return_field: bool = False):
    """D3: amplitude spectrum of the (mean-filled) region kept, phase randomised (Hermitian-symmetric random
    phase from the FFT of white noise, so the result is real)."""
    x, m = _filled(y, mask)
    F = np.fft.fft2(x - m)
    noise = np.fft.fft2(np.random.default_rng(seed).standard_normal(x.shape))
    field = np.real(np.fft.ifft2(np.abs(F) * np.exp(1j * (np.angle(F) + np.angle(noise)))))
    out = np.where(mask, field + m, y).astype(np.float32)
    return (out, field) if return_field else out


def silhouette(y, mask, mode: str = "uniform", seed: int = 0):
    """D4: 'uniform' = mask filled with the region's mean (no stripes); 'scrambled' = phase-scrambled coat."""
    if mode == "uniform":
        return np.where(mask, float(y[mask].mean()) if mask.any() else 0.0, y).astype(np.float32)
    return phase_scramble(y, mask, seed)


def occlusions(y, mask, bl_px: float, size_bl: float, stride_bl: float):
    """D5: yields (image, (r, c, size)) for each patch position overlapping the mask; the patch is filled
    with the mean luminance of the masked pixels it covers."""
    s, st = max(1, int(round(size_bl * bl_px))), max(1, int(round(stride_bl * bl_px)))
    H, W = y.shape
    for r in range(0, max(1, H - s + 1), st):
        for c in range(0, max(1, W - s + 1), st):
            sub = mask[r:r + s, c:c + s]
            if not sub.any():
                continue
            out = y.copy()
            fill = float(y[r:r + s, c:c + s][sub].mean())
            out[r:r + s, c:c + s] = np.where(sub, fill, out[r:r + s, c:c + s])
            yield out.astype(np.float32), (r, c, s)


def displacement_field(shape, mask, bl_px: float, amp_period: float, smooth_period: float, seed: int):
    """(2, H, W) smooth random field (dx, dy) whose RMS length over the mask = amp_period x stripe period."""
    period = bl_px / cfg().stripe_period_cpbl
    rng = np.random.default_rng(seed)
    d = np.stack([ndi.gaussian_filter(rng.standard_normal(shape), smooth_period * period, mode="reflect")
                  for _ in range(2)])
    rms = np.sqrt((d[0][mask] ** 2 + d[1][mask] ** 2).mean()) if mask.any() else 1.0
    return d * (amp_period * period / max(rms, 1e-12))


def morph_positional(y, mask, bl_px: float, amp_period: float, smooth_period: float, seed: int, return_field=False):
    """D6 positional tweak: the coat inside the mask is moved by a smooth random displacement."""
    d = displacement_field(y.shape, mask, bl_px, amp_period, smooth_period, seed)
    yy, xx = np.mgrid[0:y.shape[0], 0:y.shape[1]].astype(np.float32)
    w = cv2.remap(y.astype(np.float32), xx - d[0].astype(np.float32), yy - d[1].astype(np.float32),
                  cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    out = np.where(mask, w, y).astype(np.float32)
    return (out, d) if return_field else out


def morph_style(y, mask, bl_px: float, width_change: float = 0.0, contrast: float = 1.0):
    """D6 style tweak: widen (width_change > 0) or thin the DARK stripes by grey erosion / dilation with a
    disc of radius |width_change| x stripe width / 2 (symmetric, so stripe centres stay put), then scale
    contrast around the region mean."""
    period = bl_px / cfg().stripe_period_cpbl
    r = int(round(abs(width_change) * (period / 2) / 2))
    out = y.astype(np.float32)
    if r > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
        out = cv2.erode(out, k) if width_change > 0 else cv2.dilate(out, k)
    m = float(y[mask].mean()) if mask.any() else 0.0
    out = m + contrast * (out - m)
    return np.where(mask, out, y).astype(np.float32)


def transplant(recipient: View, donor: View, part: str, feather_bl: float | None = None, g2q=None):
    """D7: the donor's `part` (frame.parts rule) pasted into the recipient.

    Mapping recipient px -> donor px: through both canonical frames (default), or `g2q` (a callable mapping
    recipient px -> donor px, e.g. a stored pair alignment through oracle_warp's chain). Luminance-matched
    (donor mean/std set to the recipient's inside the part), feathered with a Gaussian of feather_bl x body
    length. donor = recipient returns the recipient (control)."""
    feather_bl = cfg_gen().transplant_feather_bl if feather_bl is None else feather_bl
    H, W = recipient.y.shape
    yy, xx = np.mgrid[0:H, 0:W]
    pts = np.c_[xx.ravel(), yy.ravel()].astype(float)
    src = g2q(pts) if g2q is not None else apply(compose(invert(donor.px2can), recipient.px2can), pts)
    mx, my = src[:, 0].reshape(H, W).astype(np.float32), src[:, 1].reshape(H, W).astype(np.float32)
    d_y = cv2.remap(donor.y.astype(np.float32), mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    d_m = cv2.remap(donor.mask.astype(np.uint8), mx, my, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT) > 0
    uv = apply(recipient.px2can, pts)
    pm = part_mask(part)
    iu = np.clip(uv[:, 0].astype(int), 0, pm.shape[1] - 1)
    iv = np.clip(uv[:, 1].astype(int), 0, pm.shape[0] - 1)
    inframe = (uv[:, 0] >= 0) & (uv[:, 0] < pm.shape[1]) & (uv[:, 1] >= 0) & (uv[:, 1] < pm.shape[0])
    region = (pm[iv, iu] & inframe).reshape(H, W) & recipient.mask & d_m
    if not region.any():
        return recipient.y.copy(), region
    a, b = recipient.y[region], d_y[region]
    matched = (d_y - b.mean()) / max(b.std(), 1e-9) * a.std() + a.mean() if b.std() > 1e-9 else d_y - b.mean() + a.mean()
    alpha = ndi.gaussian_filter(region.astype(float), max(0.5, feather_bl * recipient.bl_px)) * region
    out = alpha * matched + (1 - alpha) * recipient.y
    return out.astype(np.float32), region


def cfg_gen():
    from zebrasight.config import load_config

    return load_config("positional").generators


__all__ = ["band_filter", "patch_shuffle", "phase_scramble", "silhouette", "occlusions", "displacement_field",
           "morph_positional", "morph_style", "transplant", "canonical_points"]
