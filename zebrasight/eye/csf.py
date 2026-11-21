"""Contrast-sensitivity stage. Two modes (``csf.mode``):

CSF shape (both modes; Melin et al. 2016 form, log reading of the peak shift):
    S(f; A) = c * 10^(-((log10 f - log10 b) / w)^2),  b = b_frac * A,  c = c_peak
    w = log10(1 / b_frac) / sqrt(log10 c)                so that S(A; A) = 1
Flat top: S_eff = c for f <= b (body outline and shading are kept).

"peli" (default): Peli (1990) local band-limited contrast.
- Filter bank: raised-cosine bands in log2 frequency, spacing s octaves
  (``csf.band_spacing_oct``, 0.5), centres f_k = A * 2^(k s) so A is a band centre:
      G_k(f) = cos^2(pi/2 * (log2 f - log2 f_k) / s)  for |log2 f - log2 f_k| < s
  The G_k sum to 1; frequencies below the lowest centre go to a residual low-pass.
- Local contrast (Peli): c_k(x) = a_k(x) / l_k(x), l_k = everything below band k
  (residual + bands j < k), floored at 1% of the mean.
- Visibility, per pixel and band:
      v_k(x) = env[ IFFT(X * T_k(f) * S_eff(f; A)) ](x) / l_k(x) * S_eff(f_k; A(x)) / S_eff(f_k; A)
  Three refinements over thresholding a_k with S(f_k):
  (1) T_k is a flat-top analysis band (gain 1 within s/2 of f_k), so a grating is not
      split between two bands, each seeing only part of its contrast;
  (2) each frequency is weighted by its own S(f), not by S at the band centre (with
      half-octave bands the centre can be 1/4 octave off, which near the acuity changes
      S by ~2x);
  (3) env = monogenic-signal amplitude sqrt(b^2 + (R_y b)^2 + (R_x b)^2) (Riesz transform),
      the local contrast amplitude; |a_k| itself oscillates with the stripes and would
      switch the threshold off at every zero crossing.
  The last factor carries the per-pixel acuity (streak); it is 1 where A(x) = A.
  Without (1)-(2) a full-contrast grating at 0.9 x acuity was erased (tests/eye/test_fixes.py).
- Soft threshold:
      q_k(x) = clamp((log10 v_k + delta) / (2 delta), 0, 1)      delta = ``csf.soft_threshold_log10``
  i.e. 0 below v = 10^-delta, 1 above 10^+delta, linear in log v in between.
- Local acuity ramp: r_k(x) = clamp(1 + (log2 A(x) - log2 f_k) / s, 0, 1) removes bands above
  the local acuity A(x) = acuity_map (visual streak). On the streak centre A(x) = A.
- Reconstruction: y = residual + sum_k q_k r_k * band_k filtered by H(f), where
  H = 1 for f <= b, S(f; A) / c above, with a raised-cosine cut-off from A to
  A * (1 + ``csf.cutoff_taper``) so nothing beyond the acuity survives.
  H uses the regime peak A; the per-pixel acuity acts through q_k and r_k.
- Chroma (S cone): the same bands with the ramp at ``colour.chroma_acuity_ratio`` x A(x), plus a
  raised-cosine low-pass from 0.8 to 1.0 x (ratio x A). No contrast threshold (no horse data).

"global" (comparison only): FFT of the whole image, H as above, and components above b
whose global contrast 2|F(f)|/|F(0)| x S(f) < 1 are zeroed. The per-coefficient contrast
depends on how spread the spectrum is (wavy stripes, crop size), which is why peli is the
default. In this mode the streak stage applies the spatial acuity variation as a blur.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from zebrasight.eye.ops import fft_filter, radial_freq, reflect_pad, unpad
from zebrasight.eye.streak import regime_acuity
from zebrasight.registry import Stage, register_stage
from zebrasight.types import ViewState

MODES = ("peli", "global")


def csf_params(acuity: float, b_fraction: float, c_peak: float) -> tuple[float, float, float]:
    b = b_fraction * acuity
    w = (math.log10(acuity) - math.log10(b)) / math.sqrt(math.log10(c_peak))
    return b, c_peak, w


def csf_sensitivity(f_cpd: torch.Tensor, b, c: float, w: float) -> torch.Tensor:
    """S(f); ``b`` may be a float or a tensor broadcastable with f (per-pixel peak)."""
    lf = torch.log10(f_cpd.clamp_min(1e-9))
    lb = math.log10(b) if isinstance(b, (int, float)) else torch.log10(b)
    return c * torch.pow(10.0, -(((lf - lb) / w) ** 2))


def csf_transfer(f: torch.Tensor, acuity: float, b: float, c: float, w: float, taper: float) -> torch.Tensor:
    """H(f): 1 up to b, S/c above, raised-cosine to 0 between A and A(1 + taper)."""
    h = torch.where(f > b, csf_sensitivity(f, b, c, w) / c, torch.ones_like(f))
    t = ((f - acuity) / (acuity * taper)).clamp(0.0, 1.0)
    return h * 0.5 * (1.0 + torch.cos(math.pi * t))


def cosine_lowpass(f: torch.Tensor, cutoff: float, start: float = 0.8) -> torch.Tensor:
    t = ((f - start * cutoff) / ((1 - start) * cutoff)).clamp(0.0, 1.0)
    return 0.5 * (1.0 + torch.cos(math.pi * t))


def band_centres(f_lo: float, f_hi: float, anchor: float, s: float) -> list[float]:
    k0 = math.floor(math.log2(f_lo / anchor) / s)
    k1 = math.ceil(math.log2(f_hi / anchor) / s)
    return [anchor * 2 ** (k * s) for k in range(k0, k1 + 1)]


def log_cos_band(f: torch.Tensor, centre: float, s: float) -> torch.Tensor:
    d = (torch.log2(f.clamp_min(1e-12)) - math.log2(centre)) / s
    g = torch.cos(0.5 * math.pi * d) ** 2
    return torch.where((d.abs() < 1.0) & (f > 0), g, torch.zeros_like(g))


def flat_top_band(f: torch.Tensor, centre: float, s: float) -> torch.Tensor:
    """Analysis band: gain 1 within s/2 octaves of the centre, raised-cosine to 0 at s.
    Every frequency gets full gain in its nearest band, so a grating's contrast is not
    split between two bands when visibility is judged."""
    d = ((torch.log2(f.clamp_min(1e-12)) - math.log2(centre)) / s).abs()
    t = ((d - 0.5) / 0.5).clamp(0.0, 1.0)
    g = 0.5 * (1.0 + torch.cos(math.pi * t))
    return torch.where((d < 1.0) & (f > 0), g, torch.zeros_like(g))


def riesz_kernels(hp: int, wp: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    """Frequency responses -i fy/|f| and -i fx/|f| of the Riesz transform (rfft layout)."""
    fy = torch.fft.fftfreq(hp, device=device)[:, None].expand(hp, wp // 2 + 1)
    fx = torch.fft.rfftfreq(wp, device=device)[None, :].expand(hp, wp // 2 + 1)
    r = torch.sqrt(fy**2 + fx**2).clamp_min(1e-12)
    return -1j * fy / r, -1j * fx / r


def envelope(X_band: torch.Tensor, ry: torch.Tensor, rx: torch.Tensor, size) -> torch.Tensor:
    """Local amplitude of a band-pass signal: monogenic signal |(b, R_y b, R_x b)|.
    Constant for a sinusoid of any orientation, unlike |b| which oscillates with the carrier."""
    b = torch.fft.irfft2(X_band, s=size)
    by = torch.fft.irfft2(X_band * ry, s=size)
    bx = torch.fft.irfft2(X_band * rx, s=size)
    return torch.sqrt(b**2 + by**2 + bx**2)


def peli_csf(lum: torch.Tensor, chroma: torch.Tensor | None, dpp: float, acuity: float,
             acuity_map: torch.Tensor | None, b_frac: float, c: float, s: float, delta: float, taper: float,
             chroma_ratio: float) -> tuple[torch.Tensor, torch.Tensor | None, dict]:
    b, c, w = csf_params(acuity, b_frac, c)
    xp, pads = reflect_pad(lum)
    hp, wp = xp.shape[-2:]
    amap = (torch.full_like(lum, acuity) if acuity_map is None else acuity_map)
    amap = F.pad(amap, (pads[1], pads[1], pads[0], pads[0]), mode="replicate")
    log2_amap = torch.log2(amap)
    b_map = b_frac * amap

    f = radial_freq(hp, wp, lum.device) / dpp  # cpd
    f_lo = min(1.0 / hp, 1.0 / wp) / dpp
    f_hi = float(f.max())
    centres = band_centres(f_lo, f_hi, acuity, s)
    g0 = log_cos_band(f, centres[0], s)
    residual = torch.where(f < centres[0], 1.0 - g0, torch.zeros_like(f))
    H = csf_transfer(f, acuity, b, c, w, taper)

    X = torch.fft.rfft2(xp)
    size = (hp, wp)
    low = torch.fft.irfft2(X * residual, s=size)  # unfiltered, for contrast denominators
    out = torch.fft.irfft2(X * (residual * H), s=size)
    floor = 0.01 * xp.mean(dim=(-2, -1), keepdim=True).abs().clamp_min(1e-12)

    if chroma is not None:
        cp, _ = reflect_pad(chroma)
        Xc = torch.fft.rfft2(cp)
        Hc = cosine_lowpass(f, chroma_ratio * acuity)
        out_c = torch.fft.irfft2(Xc * (residual * Hc), s=size)
        log2_cmap = log2_amap + math.log2(chroma_ratio)

    s_glob = torch.where(f > b, csf_sensitivity(f, b, c, w), torch.full_like(f, c))  # flat-top S(f; A)
    ry, rx = riesz_kernels(hp, wp, lum.device)

    def s_eff_at(fk: float, peak_b):
        if isinstance(peak_b, float):
            return c if fk <= peak_b else float(csf_sensitivity(torch.tensor(fk), peak_b, c, w))
        return torch.where(fk <= peak_b, torch.full_like(peak_b, c),
                           csf_sensitivity(torch.full_like(peak_b, fk), peak_b, c, w))

    kept = []
    for fk in centres:
        g = log_cos_band(f, fk, s)
        if float(g.max()) == 0.0:
            continue
        band = torch.fft.irfft2(X * g, s=size)
        # visibility: S-weighted flat-top analysis band, local amplitude / local mean,
        # rescaled per pixel from the peak acuity A to the local acuity A(x)
        amp = envelope(X * (flat_top_band(f, fk, s) * s_glob), ry, rx, size)
        vis = amp / torch.maximum(low, floor) * (s_eff_at(fk, b_map) / s_eff_at(fk, b))
        q = ((torch.log10(vis.clamp_min(1e-12)) + delta) / (2 * delta)).clamp(0.0, 1.0)
        r = (1.0 + (log2_amap - math.log2(fk)) / s).clamp(0.0, 1.0)
        out = out + q * r * torch.fft.irfft2(X * (g * H), s=size)
        low = low + band
        kept.append(float((q * r).mean()))
        if chroma is not None:
            rc = (1.0 + (log2_cmap - math.log2(fk)) / s).clamp(0.0, 1.0)
            out_c = out_c + rc * torch.fft.irfft2(Xc * (g * Hc), s=size)

    info = {"n_bands": len(centres), "band_centres_cpd": [round(x, 4) for x in centres],
            "mean_kept_per_band": [round(x, 4) for x in kept]}
    return unpad(out, pads), (unpad(out_c, pads) if chroma is not None else None), info


def global_csf(x: torch.Tensor, dpp: float, acuity: float, b: float, c: float, w: float, taper: float) -> torch.Tensor:
    def transfer(f_px, X):
        f = f_px / dpp
        s = csf_sensitivity(f, b, c, w)
        h = csf_transfer(f, acuity, b, c, w, taper)
        contrast = 2.0 * X.abs() / X[..., :1, :1].abs().clamp_min(1e-12)
        h = h * ((f <= b) | (contrast * s >= 1.0))
        h[..., 0, 0] = 1.0
        return X * h

    return fft_filter(x, transfer)


def chroma_lowpass(x: torch.Tensor, dpp: float, cutoff_cpd: float) -> torch.Tensor:
    return fft_filter(x, lambda f_px, X: X * cosine_lowpass(f_px / dpp, cutoff_cpd))


@register_stage("csf")
class CSF(Stage):
    def forward(self, state: ViewState) -> ViewState:
        mode = str(self.param("csf.mode"))
        if mode not in MODES:
            raise ValueError(f"csf.mode must be one of {MODES}, got {mode!r}")
        regime, acuity = regime_acuity(self, state.cond.light)
        b_frac = float(self.param("csf.b_peak_fraction"))
        c_peak = float(self.param("csf.c_peak_sensitivity"))
        taper = float(self.param("csf.cutoff_taper"))
        ratio = float(self.param("colour.chroma_acuity_ratio"))
        b, c, w = csf_params(acuity, b_frac, c_peak)
        dpp = state.deg_per_px
        lum = state.rod if state.rod is not None else state.cones[:, 0:1]
        chroma = None if state.rod is not None else state.cones[:, 1:2]
        info = {}
        if mode == "peli":
            s = float(self.param("csf.band_spacing_oct"))
            delta = float(self.param("csf.soft_threshold_log10"))
            lum, chroma, info = peli_csf(lum, chroma, dpp, acuity, state.acuity_map, b_frac, c_peak, s, delta,
                                         taper, ratio)
            info.update(band_spacing_oct=s, soft_threshold_log10=delta, per_pixel_acuity=state.acuity_map is not None)
        else:
            lum = global_csf(lum, dpp, acuity, b, c, w, taper)
            chroma = chroma_lowpass(chroma, dpp, ratio * acuity) if chroma is not None else None
        if state.rod is not None:
            state = state.evolve(rod=lum)
        else:
            state = state.evolve(cones=torch.cat([lum, chroma], dim=1))
        self.record(state, mode=mode, regime=regime, acuity_cpd=acuity, b_cpd=b, c_peak=c, width_log10=w,
                    cutoff_taper=taper, chroma_cutoff_cpd=None if state.rod is not None else ratio * acuity,
                    nyquist_cpd=0.5 / dpp, **info)
        return state
