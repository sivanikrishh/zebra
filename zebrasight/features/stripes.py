"""F1-F10 stripe features and the spatial families L (layout) and M (minutiae map).

All lengths in coat cm, densities per 100 cm^2, frequencies in cycles/cm. Every function returns
NaN for every one of its outputs when the percept is invalid (too small / degenerate): NaN later
counts as "not recognisable", never as dropped.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import ndimage as ndi
from skimage.filters import threshold_multiotsu
from skimage.measure import label, regionprops
from skimage.transform import resize

from zebrasight.features.common import EIGHT, Ctx, config, prepare
from zebrasight.features.registry import register_feature

NAN = float("nan")


def _ctx(percept, distance_m, ctx):
    return ctx if ctx is not None else prepare(percept, distance_m)


def _nan(*names):
    return {n: NAN for n in names}


def _cv(a):
    a = np.asarray(a, float)
    return float(a.std() / a.mean()) if a.size >= 2 and a.mean() > 0 else NAN


@register_feature("F1")
def f1_width(percept, distance_m, ctx: Ctx | None = None):
    """Stripe width: (2 x distance transform - 1) sampled on the skeleton, in cm. The EDT at a run's
    centre pixel is measured to the nearest background pixel CENTRE, so 2*dt = n + 1 for an n-px
    run; the -1 removes that one-pixel bias (it matters at the few-pixel widths of far percepts)."""
    names = ("F1_width_cm_median", "F1_width_cv")
    c = _ctx(percept, distance_m, ctx)
    if not c.valid or not c.skel.any():
        return _nan(*names)
    w = np.maximum(2.0 * c.dt[c.skel] - 1.0, 1.0) * c.cm_per_px
    return dict(zip(names, (float(np.median(w)), _cv(w))))


@register_feature("F2")
def f2_black_fraction(percept, distance_m, ctx: Ctx | None = None):
    """Mean rho in the mask; black:white run-length ratio across the stripes."""
    names = ("F2_rho_mean", "F2_run_ratio")
    c = _ctx(percept, distance_m, ctx)
    if not c.valid:
        return _nan(*names)
    rho = float(np.asarray(percept["rho"], float)[c.mask].mean())
    rb, rw = c.runs_black_px, c.runs_white_px
    ratio = float(np.median(rb) / np.median(rw)) if rb.size and rw.size else NAN
    return {"F2_rho_mean": rho, "F2_run_ratio": ratio}


@register_feature("F3")
def f3_curvature(percept, distance_m, ctx: Ctx | None = None):
    """Curvature = 1/2 |d(phi)/ds| along the stripe tangent, phi = doubled orientation angle; on the skeleton, 1/cm."""
    names = ("F3_curv_mean", "F3_curv_median")
    c = _ctx(percept, distance_m, ctx)
    if not c.valid or not c.skel.any():
        return _nan(*names)
    zy, zx = np.gradient(c.z)
    # d(phi) = Im(conj(z) dz) for a unit field z = exp(i phi)
    dphi_y = np.imag(np.conj(c.z) * zy)
    dphi_x = np.imag(np.conj(c.z) * zx)
    tang = c.theta_n + math.pi / 2
    dphi_ds = dphi_x * np.cos(tang) + dphi_y * np.sin(tang)
    k = 0.5 * np.abs(dphi_ds[c.skel]) / c.cm_per_px
    return dict(zip(names, (float(k.mean()), float(np.median(k)))))


@register_feature("F4")
def f4_orientation(percept, distance_m, ctx: Ctx | None = None):
    """Dominant stripe angle (as cos 2a, sin 2a: angles are circular), orientation entropy, coherence."""
    names = ("F4_cos2a", "F4_sin2a", "F4_entropy", "F4_coherence")
    c = _ctx(percept, distance_m, ctx)
    if not c.valid:
        return _nan(*names)
    a = c.dominant_normal + math.pi / 2  # stripe direction
    th = (c.theta_n[c.mask] + math.pi / 2) % math.pi
    w = c.energy[c.mask]
    h, _ = np.histogram(th, bins=config()["orientation_bins"], range=(0, math.pi), weights=w)
    p = h / h.sum() if h.sum() > 0 else h
    ent = float(-(p[p > 0] * np.log2(p[p > 0])).sum())
    coh = float((c.coherence[c.mask] * w).sum() / max(w.sum(), 1e-12))
    return {"F4_cos2a": math.cos(2 * a), "F4_sin2a": math.sin(2 * a), "F4_entropy": ent, "F4_coherence": coh}


@register_feature("F5")
def f5_junctions(percept, distance_m, ctx: Ctx | None = None):
    c = _ctx(percept, distance_m, ctx)
    if not c.valid:
        return _nan("F5_junctions_per100cm2")
    return {"F5_junctions_per100cm2": 100.0 * len(c.junctions) / c.area_cm2}


@register_feature("F6")
def f6_endings(percept, distance_m, ctx: Ctx | None = None):
    c = _ctx(percept, distance_m, ctx)
    if not c.valid:
        return _nan("F6_endings_per100cm2")
    return {"F6_endings_per100cm2": 100.0 * len(c.endpoints) / c.area_cm2}


@register_feature("F7")
def f7_spacing(percept, distance_m, ctx: Ctx | None = None):
    """Transect period and spacing CV (cm); FFT peak (cycles/cm) and radial frequency entropy (Hann window)."""
    names = ("F7_period_cm", "F7_spacing_cv", "F7_fft_peak_cpcm", "F7_fft_entropy")
    c = _ctx(percept, distance_m, ctx)
    if not c.valid:
        return _nan(*names)
    out = _nan(*names)
    rb, rw = c.runs_black_px, c.runs_white_px
    if rb.size and rw.size:
        out["F7_period_cm"] = float((np.median(rb) + np.median(rw)) * c.cm_per_px)
    gaps, prev = [], None
    for ce in c.black_centres_px:
        if ce is None:
            prev = None
            continue
        if prev is not None:
            gaps.append(ce - prev)
        prev = ce
    out["F7_spacing_cv"] = _cv(gaps)
    r0, r1, c0, c1 = c.bbox
    sub = (c.yn[r0:r1, c0:c1] - 1.0) * c.mask[r0:r1, c0:c1]
    win = np.outer(np.hanning(sub.shape[0]), np.hanning(sub.shape[1]))
    P = np.abs(np.fft.rfft2(sub * win)) ** 2
    fy = np.fft.fftfreq(sub.shape[0])[:, None]
    fx = np.fft.rfftfreq(sub.shape[1])[None, :]
    fr = np.sqrt(fy**2 + fx**2)
    fmin = 1.0 / max(sub.shape)
    band = (fr >= 1.5 * fmin) & (fr <= 0.5)
    if band.any() and P[band].sum() > 0:
        out["F7_fft_peak_cpcm"] = float(fr[band][np.argmax(P[band])] / c.cm_per_px)
        edges = np.geomspace(1.5 * fmin, 0.5, config()["fft_entropy_bins"] + 1)
        h, _ = np.histogram(fr[band], bins=edges, weights=P[band])
        p = h / h.sum()
        out["F7_fft_entropy"] = float(-(p[p > 0] * np.log2(p[p > 0])).sum())
    return out


@register_feature("F8")
def f8_shadow(percept, distance_m, ctx: Ctx | None = None):
    """3-class multi-Otsu; elongated middle-class components inside white regions (not touching black)."""
    names = ("F8_shadow_area_frac", "F8_shadow_count_per100cm2")
    c = _ctx(percept, distance_m, ctx)
    if not c.valid:
        return _nan(*names)
    cfg = config()
    try:
        t1, t2 = threshold_multiotsu(c.yn[c.mask], classes=cfg["multiotsu_classes"])
    except ValueError:
        return _nan(*names)
    dark = (c.yn < t1) & c.mask
    mid = (c.yn >= t1) & (c.yn < t2) & c.mask
    near_dark = ndi.binary_dilation(dark, structure=EIGHT.astype(bool))
    lab = label(mid, connectivity=2)
    area = count = 0
    for rp in regionprops(lab):
        if rp.area < cfg["shadow_min_px"]:
            continue
        coords = tuple(rp.coords.T)
        if near_dark[coords].any():  # blurred black/white boundaries, not a shadow stripe
            continue
        if rp.axis_minor_length > 0 and rp.axis_major_length / rp.axis_minor_length >= cfg["shadow_min_elongation"]:
            area += rp.area
            count += 1
    return {"F8_shadow_area_frac": area / c.mask.sum(), "F8_shadow_count_per100cm2": 100.0 * count / c.area_cm2}


@register_feature("F9")
def f9_contrast(percept, distance_m, ctx: Ctx | None = None):
    """Michelson contrast black vs white classes; boundary gradient / contrast (sharpness, 1/cm)."""
    names = ("F9_michelson", "F9_sharpness_per_cm")
    c = _ctx(percept, distance_m, ctx)
    if not c.valid:
        return _nan(*names)
    yb, yw = c.yn[c.black].mean(), c.yn[c.white].mean()
    mich = float((yw - yb) / (yw + yb)) if yw + yb > 0 else NAN
    boundary = c.black & ndi.binary_dilation(c.white, structure=EIGHT.astype(bool))
    gm = np.hypot(ndi.sobel(c.yn, 0), ndi.sobel(c.yn, 1)) / 8.0
    sharp = float(gm[boundary].mean() / max(yw - yb, 1e-9) / c.cm_per_px) if boundary.any() else NAN
    return {"F9_michelson": mich, "F9_sharpness_per_cm": sharp}


@register_feature("F10")
def f10_topology(percept, distance_m, ctx: Ctx | None = None):
    """GUDHI cubical persistence on yn over the mask box (outside mask = median, 1.0)."""
    import gudhi

    names = ("F10_h0_count", "F10_h1_count", "F10_total_pers", "F10_pers_entropy")
    c = _ctx(percept, distance_m, ctx)
    if not c.valid:
        return _nan(*names)
    r0, r1, c0, c1 = c.bbox
    img = np.where(c.mask[r0:r1, c0:c1], c.yn[r0:r1, c0:c1], 1.0)
    cc = gudhi.CubicalComplex(top_dimensional_cells=img)
    pairs = cc.persistence()
    thr = config()["persistence_threshold"]
    life = {0: [], 1: []}
    for dim, (b, d) in pairs:
        if dim in life and np.isfinite(d) and d - b > thr:
            life[dim].append(d - b)
    allp = np.array(life[0] + life[1])
    ent = NAN
    if allp.size:
        p = allp / allp.sum()
        ent = float(-(p * np.log2(p)).sum())
    return {"F10_h0_count": float(len(life[0])), "F10_h1_count": float(len(life[1])),
            "F10_total_pers": float(allp.sum()) if allp.size else 0.0, "F10_pers_entropy": ent}


@register_feature("L")
def layout(percept, distance_m, ctx: Ctx | None = None):
    """rho in the region box resampled to 16 x 24 (outside the mask = 0.5); PCA is fitted later on gallery only."""
    gh, gw = config()["layout_grid"]
    names = [f"L_{i:03d}" for i in range(gh * gw)]
    c = _ctx(percept, distance_m, ctx)
    if not c.valid:
        return _nan(*names)
    r0, r1, c0, c1 = c.bbox
    rho = np.where(c.mask[r0:r1, c0:c1], np.asarray(percept["rho"], float)[r0:r1, c0:c1], 0.5)
    g = resize(rho, (gh, gw), order=1, anti_aliasing=True, mode="edge")
    return dict(zip(names, g.ravel().astype(float)))


@register_feature("M")
def minutiae(percept, distance_m, ctx: Ctx | None = None):
    """Junctions (J) and endings (E) on a 4 x 6 grid over the region box, per 100 cm^2 of cell."""
    gh, gw = config()["minutiae_grid"]
    names = [f"M_{k}_{i}{j}" for k in "JE" for i in range(gh) for j in range(gw)]
    c = _ctx(percept, distance_m, ctx)
    if not c.valid:
        return _nan(*names)
    r0, r1, c0, c1 = c.bbox
    cell_cm2 = (r1 - r0) * (c1 - c0) / (gh * gw) * c.cm_per_px**2
    out = {}
    for k, pts in (("J", c.junctions), ("E", c.endpoints)):
        h = np.zeros((gh, gw))
        for (r, col) in pts:
            i = min(gh - 1, int((r - r0) / max(r1 - r0, 1) * gh))
            j = min(gw - 1, int((col - c0) / max(c1 - c0, 1) * gw))
            h[i, j] += 1
        for i in range(gh):
            for j in range(gw):
                out[f"M_{k}_{i}{j}"] = 100.0 * h[i, j] / cell_cm2
    return out
