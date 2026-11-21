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


def _to_canvas(x: torch.Tensor, hc: int, wc: int) -> torch.Tensor:
    """Extend an already reflect-padded image to the canvas by PERIODIC tiling. The single-image
    FFT treats the padded image as periodic, so tiling keeps each item's surroundings as close
    as possible to what it would see alone (replicate padding added low-frequency structure)."""
    h, w = x.shape[-2:]
    if (h, w) == (hc, wc):
        return x
    return x.repeat(1, 1, math.ceil(hc / h), math.ceil(wc / w))[..., :hc, :wc]


def size_buckets(sizes: list[tuple[int, int]], ratio: float = 1.5, small: int = 64) -> list[list[int]]:
    """Group item indices whose padded sizes are within ``ratio`` of the bucket's largest
    (everything up to ``small`` px shares one bucket). Batching mostly pays off for the many
    small far-distance images; similar sizes keep the shared canvas close to each item's own."""
    order = sorted(range(len(sizes)), key=lambda i: -max(sizes[i]))
    buckets: list[list[int]] = []
    for i in order:
        h, w = sizes[i]
        if buckets:
            bh = max(sizes[j][0] for j in buckets[-1])
            bw = max(sizes[j][1] for j in buckets[-1])
            if max(bh, bw) <= small or (h * ratio >= bh and w * ratio >= bw):
                buckets[-1].append(i)
                continue
        buckets.append([i])
    return buckets


def peli_csf(lum: torch.Tensor, chroma: torch.Tensor | None, dpp: float, acuity: float,
             acuity_map: torch.Tensor | None, b_frac: float, c: float, s: float, delta: float, taper: float,
             chroma_ratio: float) -> tuple[torch.Tensor, torch.Tensor | None, dict]:
    """Single image: a batch of one (identical to the batched computation)."""
    lums, chromas, info = peli_csf_batch([lum], None if chroma is None else [chroma], [dpp], acuity, [acuity_map],
                                         b_frac, c, s, delta, taper, chroma_ratio)
    return lums[0], (chromas[0] if chromas is not None else None), info


def peli_csf_batch(lums: list[torch.Tensor], chromas: list[torch.Tensor] | None, dpps: list[float], acuity: float,
                   acuity_maps: list[torch.Tensor | None], b_frac: float, c: float, s: float, delta: float,
                   taper: float, chroma_ratio: float) -> tuple[list[torch.Tensor], list[torch.Tensor] | None, dict]:
    """Peli CSF on several images of ONE condition (same regime acuity A), e.g. the 16 distances.

    Images may differ in size and deg/px. Each is reflect-padded as in the single-image case,
    then replicate-padded (bottom/right) to a shared canvas; all frequency filters are built per
    item from its own deg/px, so every item sees its own cpd grid. One FFT pass serves the
    batch, removing per-call GPU launch overhead. With one item the canvas is the item itself,
    so the result is the single-image computation exactly. With several items, the extra canvas
    padding lies outside each item's reflect pad and changes results only near the borders
    (checked in tests/eye/test_batch.py).
    """
    b, c, w = csf_params(acuity, b_frac, c)
    dev = lums[0].device
    padded = [reflect_pad(x) for x in lums]
    hc = max(xp.shape[-2] for xp, _ in padded)
    wc = max(xp.shape[-1] for xp, _ in padded)
    xp = torch.cat([_to_canvas(x, hc, wc) for x, _ in padded])  # (B,1,hc,wc)
    floor = torch.stack([0.01 * x.mean().abs().clamp_min(1e-12) for x, _ in padded]).view(-1, 1, 1, 1)
    amaps = []
    for x, (_, pads), am in zip(lums, padded, acuity_maps):
        am = torch.full_like(x, acuity) if am is None else am
        am = F.pad(am, (pads[1], pads[1], pads[0], pads[0]), mode="replicate")
        amaps.append(_to_canvas(am, hc, wc))
    amap = torch.cat(amaps)
    log2_amap = torch.log2(amap)
    b_map = b_frac * amap

    dpp_t = torch.tensor(dpps, dtype=xp.dtype, device=dev).view(-1, 1, 1, 1)
    f = radial_freq(hc, wc, dev)[None, None] / dpp_t  # (B,1,hc,wc//2+1), cpd per item
    # each item keeps the band set it would have alone (lowest band from its OWN padded size),
    # so batching changes nothing but the canvas padding; bands below an item's lowest band
    # stay in its residual (always kept, never thresholded)
    item_lo = [1.0 / max(x.shape[-2:]) / d for (x, _), d in zip(padded, dpps)]
    f_hi = float(f.max())
    if f_hi <= 0.0:  # every item is a single pixel: DC only, which the CSF passes unchanged
        return list(lums), (list(chromas) if chromas is not None else None), {"n_bands": 0, "batch": len(lums)}
    centres = band_centres(min(item_lo), f_hi, acuity, s)
    k0 = [next(k for k, ck in enumerate(centres) if ck >= band_centres(lo, f_hi, acuity, s)[0] - 1e-9)
          for lo in item_lo]
    residual = torch.cat([
        torch.where(f[i : i + 1] < centres[k0[i]], 1.0 - log_cos_band(f[i : i + 1], centres[k0[i]], s),
                    torch.zeros_like(f[i : i + 1]))
        for i in range(len(lums))])
    k0_t = torch.tensor(k0, device=dev).view(-1, 1, 1, 1)
    H = csf_transfer(f, acuity, b, c, w, taper)

    X = torch.fft.rfft2(xp)
    size = (hc, wc)
    low = torch.fft.irfft2(X * residual, s=size)  # unfiltered, for contrast denominators
    out = torch.fft.irfft2(X * (residual * H), s=size)

    if chromas is not None:
        cp = torch.cat([_to_canvas(reflect_pad(x)[0], hc, wc) for x in chromas])
        Xc = torch.fft.rfft2(cp)
        Hc = cosine_lowpass(f, chroma_ratio * acuity)
        out_c = torch.fft.irfft2(Xc * (residual * Hc), s=size)
        log2_cmap = log2_amap + math.log2(chroma_ratio)

    s_glob = torch.where(f > b, csf_sensitivity(f, b, c, w), torch.full_like(f, c))  # flat-top S(f; A)
    ry, rx = riesz_kernels(hc, wc, dev)

    def s_eff_at(fk: float, peak_b):
        if isinstance(peak_b, float):
            return c if fk <= peak_b else float(csf_sensitivity(torch.tensor(fk), peak_b, c, w))
        return torch.where(fk <= peak_b, torch.full_like(peak_b, c),
                           csf_sensitivity(torch.full_like(peak_b, fk), peak_b, c, w))

    kept = []
    for k, fk in enumerate(centres):
        if fk * 2 ** (-s) > acuity * (1.0 + taper):
            break  # band lies entirely above the cut-off: H = 0 there, so it can add nothing (exact)
        active = (k0_t <= k).to(xp.dtype)  # (B,1,1,1): band exists for this item
        if float(active.max()) == 0.0:
            continue
        g = log_cos_band(f, fk, s) * active
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
        if chromas is not None:
            rc = (1.0 + (log2_cmap - math.log2(fk)) / s).clamp(0.0, 1.0)
            out_c = out_c + rc * torch.fft.irfft2(Xc * (g * Hc), s=size)

    def crop(t: torch.Tensor, i: int) -> torch.Tensor:
        ph, pw = padded[i][1]
        h, w_ = lums[i].shape[-2:]
        return t[i : i + 1, :, ph : ph + h, pw : pw + w_]

    info = {"n_bands": len(centres), "band_centres_cpd": [round(x, 4) for x in centres],
            "mean_kept_per_band": [round(x, 4) for x in kept], "batch": len(lums), "canvas": (hc, wc)}
    lum_out = [crop(out, i) for i in range(len(lums))]
    chroma_out = [crop(out_c, i) for i in range(len(lums))] if chromas is not None else None
    return lum_out, chroma_out, info


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
        return self.forward_batch([state])[0]

    def forward_batch(self, states: list[ViewState]) -> list[ViewState]:
        """CSF for several states of ONE condition (same light regime), e.g. the 16 distances.
        In peli mode they share one batched FFT pass (peli_csf_batch); global mode loops."""
        mode = str(self.param("csf.mode"))
        if mode not in MODES:
            raise ValueError(f"csf.mode must be one of {MODES}, got {mode!r}")
        lights = {st.cond.light for st in states}
        if len({regime_acuity(self, li)[0] for li in lights}) != 1:
            raise ValueError(f"forward_batch needs one light regime, got {lights}")
        regime, acuity = regime_acuity(self, states[0].cond.light)
        b_frac = float(self.param("csf.b_peak_fraction"))
        c_peak = float(self.param("csf.c_peak_sensitivity"))
        taper = float(self.param("csf.cutoff_taper"))
        ratio = float(self.param("colour.chroma_acuity_ratio"))
        b, c, w = csf_params(acuity, b_frac, c_peak)
        night = states[0].rod is not None
        lums = [st.rod if night else st.cones[:, 0:1] for st in states]
        chromas = None if night else [st.cones[:, 1:2] for st in states]
        infos: list = [{} for _ in states]
        common: dict = {}
        if mode == "peli":
            s = float(self.param("csf.band_spacing_oct"))
            delta = float(self.param("csf.soft_threshold_log10"))
            from zebrasight.eye.ops import _pad_amount

            # flatten to single images: (state index, sample index within that state's batch)
            items = [(i, j) for i, x in enumerate(lums) for j in range(x.shape[0])]
            one = lambda t, j: None if t is None else t[j : j + 1]  # noqa: E731
            sizes = [(lums[i].shape[-2] + 2 * _pad_amount(lums[i].shape[-2]),
                      lums[i].shape[-1] + 2 * _pad_amount(lums[i].shape[-1])) for i, _ in items]
            res_l: list = [None] * len(items)
            res_c: list = [None] * len(items)
            buckets = size_buckets(sizes)
            for bucket in buckets:
                sel = [items[k] for k in bucket]
                bl, bc, binfo = peli_csf_batch(
                    [one(lums[i], j) for i, j in sel], None if chromas is None else [one(chromas[i], j) for i, j in sel],
                    [states[i].deg_per_px for i, _ in sel], acuity, [one(states[i].acuity_map, j) for i, j in sel],
                    b_frac, c_peak, s, delta, taper, ratio)
                for n, k in enumerate(bucket):
                    res_l[k] = bl[n]
                    res_c[k] = bc[n] if bc is not None else None
                    infos[items[k][0]] = binfo
            lums = [torch.cat([res_l[k] for k, (i, _) in enumerate(items) if i == s_]) for s_ in range(len(states))]
            if chromas is not None:
                chromas = [torch.cat([res_c[k] for k, (i, _) in enumerate(items) if i == s_])
                           for s_ in range(len(states))]
            common = dict(band_spacing_oct=s, soft_threshold_log10=delta, n_buckets=len(buckets),
                          per_pixel_acuity=states[0].acuity_map is not None)
        else:
            lums = [global_csf(x, st.deg_per_px, acuity, b, c, w, taper) for x, st in zip(lums, states)]
            if chromas is not None:
                chromas = [chroma_lowpass(x, st.deg_per_px, ratio * acuity) for x, st in zip(chromas, states)]
        out = []
        for i, st in enumerate(states):
            st = st.evolve(rod=lums[i]) if night else st.evolve(cones=torch.cat([lums[i], chromas[i]], dim=1))
            self.record(st, mode=mode, regime=regime, acuity_cpd=acuity, b_cpd=b, c_peak=c, width_log10=w,
                        cutoff_taper=taper, chroma_cutoff_cpd=None if night else ratio * acuity,
                        nyquist_cpd=0.5 / st.deg_per_px, **infos[i], **common)
            out.append(st)
        return out
