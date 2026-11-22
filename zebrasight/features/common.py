"""Shared per-percept preprocessing for the stripe features (computed once, used by every feature).

Units: lengths in coat centimetres, cm_per_px = 100 * distance_m * tan(radians(deg_per_px)).
Luminance is Weber-normalised per percept, yn = y / median(y[mask]), so the light level is never
an identity cue. Binary stripes: Otsu on yn inside the mask (black = below threshold), the same
rule A used for rho. A percept is invalid (all features NaN) when its mask is too small or the
binary split is degenerate (configs/features.yaml).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
from scipy import ndimage as ndi
from skimage.filters import threshold_otsu
from skimage.morphology import skeletonize

from zebrasight.config import REPO_ROOT, load_config

EIGHT = np.ones((3, 3), int)


@lru_cache(maxsize=1)
def config() -> dict:
    from omegaconf import OmegaConf

    return OmegaConf.to_container(OmegaConf.load(REPO_ROOT / "configs" / "features.yaml"))


@lru_cache(maxsize=None)
def regime_acuity(cond: str) -> float:
    grid, eye = load_config("grid"), load_config("eye")
    light = str(grid.conditions[cond].light)
    regime = {"day": "photopic", "dawn": "mesopic", "dusk": "mesopic", "night": "scotopic"}[light]
    return float(eye.acuity[regime].value)


@dataclass
class Ctx:
    valid: bool
    reason: str = ""
    cm_per_px: float = math.nan
    cycle_px: float = math.nan
    mask: np.ndarray | None = None
    yn: np.ndarray | None = None
    black: np.ndarray | None = None
    white: np.ndarray | None = None
    bbox: tuple = ()
    area_cm2: float = math.nan
    skel: np.ndarray | None = None
    junctions: list = field(default_factory=list)  # (row, col) centroids of junction clusters
    endpoints: list = field(default_factory=list)
    dt: np.ndarray | None = None  # distance transform of black stripes (px)
    theta_n: np.ndarray | None = None  # per-pixel gradient (stripe-normal) angle, radians
    coherence: np.ndarray | None = None
    energy: np.ndarray | None = None
    z: np.ndarray | None = None  # unit doubled-angle field exp(i 2 theta_n)
    dominant_normal: float = math.nan
    runs_black_px: np.ndarray | None = None
    runs_white_px: np.ndarray | None = None
    black_centres_px: list = field(default_factory=list)
    rho_corr: np.ndarray | None = None  # black fraction of the lighting-corrected binary (A's rho rule)
    black_raw: np.ndarray | None = None  # single-Otsu binary without lighting correction (sensitivity)
    runs_black_raw_px: np.ndarray | None = None
    runs_white_raw_px: np.ndarray | None = None
    prune_px: int = 0
    illum_sigma_px: float = math.nan


def neighbours(skel: np.ndarray) -> np.ndarray:
    return ndi.convolve(skel.astype(int), EIGHT, mode="constant") - skel.astype(int)


def prune(skel: np.ndarray, length: int) -> np.ndarray:
    """Remove spurs shorter than ``length`` px: strip endpoints ``length`` times, then regrow the
    surviving branches back along the original skeleton (classic pruning)."""
    if length < 1 or not skel.any():
        return skel
    s = skel.copy()
    for _ in range(length):
        ends = s & (neighbours(s) == 1)
        if not ends.any():
            break
        s &= ~ends
    grown = s.copy()
    for _ in range(length):
        ends = grown & (neighbours(grown) <= 1)
        add = ndi.binary_dilation(ends, structure=EIGHT.astype(bool)) & skel & ~grown
        if not add.any():
            break
        grown |= add
    return grown


def illumination(y: np.ndarray, mask: np.ndarray, sigma_px: float) -> np.ndarray:
    """Large-scale illumination inside the mask by normalised convolution: G*(y m) / G*m, so pixels
    near the mask edge are not darkened by the zeros outside it."""
    m = mask.astype(float)
    num = ndi.gaussian_filter(y * m, sigma_px, mode="constant")
    den = ndi.gaussian_filter(m, sigma_px, mode="constant")
    return np.where(den > 1e-6, num / np.maximum(den, 1e-6), 1.0)


def transect_runs(ctx: Ctx, n_lines: int, black: np.ndarray | None = None) -> tuple:
    """Black/white run lengths (px) along parallel lines across the stripes (along the dominant
    normal). Runs touching a segment end (truncated by the mask) are dropped. With ``black`` given,
    returns (blacks, whites) for that binary instead of filling ctx."""
    use_black = ctx.black if black is None else black
    r0, r1, c0, c1 = ctx.bbox
    cy, cx = (r0 + r1 - 1) / 2, (c0 + c1 - 1) / 2
    nvec = np.array([math.sin(ctx.dominant_normal), math.cos(ctx.dominant_normal)])  # (dy, dx)
    tvec = np.array([nvec[1], -nvec[0]])
    half = 0.5 * math.hypot(r1 - r0, c1 - c0)
    offsets = np.linspace(-0.4, 0.4, n_lines) * 2 * half
    t = np.arange(-half, half, 0.5)
    blacks, whites, centres = [], [], []
    for off in offsets:
        ys = cy + off * tvec[0] + t * nvec[0]
        xs = cx + off * tvec[1] + t * nvec[1]
        inside = (ys >= 0) & (ys < ctx.mask.shape[0] - 0.5) & (xs >= 0) & (xs < ctx.mask.shape[1] - 0.5)
        yi, xi = np.round(ys[inside]).astype(int), np.round(xs[inside]).astype(int)
        if yi.size < 4:
            continue
        m = ctx.mask[yi, xi]
        b = use_black[yi, xi]
        # split into contiguous in-mask segments
        edges = np.flatnonzero(np.diff(np.r_[0, m.astype(int), 0]))
        for s, e in zip(edges[::2], edges[1::2]):
            seg = b[s:e]
            if seg.size < 3:
                continue
            cuts = np.flatnonzero(np.diff(seg.astype(int))) + 1
            bounds = np.r_[0, cuts, seg.size]
            for k in range(1, len(bounds) - 2):  # drop first and last (truncated) runs
                L = (bounds[k + 1] - bounds[k]) * 0.5  # 0.5 px sampling step
                if seg[bounds[k]]:
                    blacks.append(L)
                    centres.append((s + (bounds[k] + bounds[k + 1]) / 2) * 0.5)
                else:
                    whites.append(L)
            centres.append(None)  # segment separator
    if black is not None:
        return np.array(blacks), np.array(whites)
    ctx.runs_black_px = np.array(blacks)
    ctx.runs_white_px = np.array(whites)
    ctx.black_centres_px = centres
    return ctx.runs_black_px, ctx.runs_white_px


def prepare(percept: dict, distance_m: float, cfg: dict | None = None) -> Ctx:
    cfg = cfg or config()
    mask = np.asarray(percept["mask"], bool)
    y = np.asarray(percept["y"], np.float64)
    dpp = float(percept["deg_per_px"])
    cm_per_px = 100.0 * distance_m * math.tan(math.radians(dpp))
    rows, cols = np.nonzero(mask.any(axis=1))[0], np.nonzero(mask.any(axis=0))[0]
    if rows.size == 0:
        return Ctx(False, "empty mask", cm_per_px)
    bbox = (rows[0], rows[-1] + 1, cols[0], cols[-1] + 1)
    short = min(bbox[1] - bbox[0], bbox[3] - bbox[2])
    if short < cfg["min_short_side_px"] or mask.sum() < cfg["min_mask_px"]:
        return Ctx(False, f"too small ({short} px short side)", cm_per_px)
    med = float(np.median(y[mask]))
    if not np.isfinite(med) or med <= 0:
        return Ctx(False, "non-positive luminance", cm_per_px)
    # D1 (configs/features.yaml): divide out large-scale lighting before Otsu
    region = str(percept.get("region") or "flank")
    sigma_px = max(cfg["illumination_sigma_min_px"], cfg["illumination_sigma_cm"].get(region, cfg["illumination_sigma_cm"]["flank"]) / cm_per_px)
    yc = np.where(mask, y / np.maximum(illumination(y, mask, sigma_px), 1e-9), 0.0)
    medc = float(np.median(yc[mask]))
    if not np.isfinite(medc) or medc <= 0:
        return Ctx(False, "non-positive luminance", cm_per_px)
    yn = yc / medc
    try:
        t = threshold_otsu(yn[mask])
    except ValueError:
        return Ctx(False, "constant region", cm_per_px)
    black = (yn < t) & mask
    frac = black.sum() / mask.sum()
    if not cfg["min_black_frac"] <= frac <= cfg["max_black_frac"]:
        return Ctx(False, f"degenerate binary ({frac:.2f} black)", cm_per_px)
    white = mask & ~black
    cond = str(percept.get("computed_as") or percept.get("cond") or "c0")
    cycle_px = (1.0 / regime_acuity(cond)) / dpp
    ctx = Ctx(True, "", cm_per_px, cycle_px, mask, yn, black, white, bbox,
              area_cm2=float(mask.sum()) * cm_per_px**2)
    ctx.illum_sigma_px = sigma_px
    # raw single-Otsu binary (no lighting correction), kept for F2_rawotsu / L_rawotsu
    yr = y / med
    try:
        ctx.black_raw = (yr < threshold_otsu(yr[mask])) & mask
    except ValueError:
        ctx.black_raw = black
    # corrected black fraction, same rule as A's rho: box mean over one acuity cycle
    win = max(1, int(round(cycle_px)) | 1)
    ctx.rho_corr = ndi.uniform_filter(black.astype(float), size=win, mode="nearest")
    # D2: prune spurs shorter than max(1 acuity cycle, 3 px, 0.5 x median stripe width)
    raw_skel = skeletonize(black)
    dt = ndi.distance_transform_edt(black)
    med_width_px = float(np.median(np.maximum(2 * dt[raw_skel] - 1, 1))) if raw_skel.any() else 0.0
    ctx.prune_px = max(round(cfg["spur_cycles"] * cycle_px), cfg["spur_min_px"],
                       round(cfg["spur_width_frac"] * med_width_px), 1)
    skel = prune(raw_skel, ctx.prune_px)
    ctx.skel = skel
    nb = neighbours(skel)
    jpx = skel & (nb >= cfg["junction_min_neighbours"])
    lab, n = ndi.label(jpx, structure=EIGHT)
    ctx.junctions = [tuple(c) for c in ndi.center_of_mass(jpx, lab, range(1, n + 1))] if n else []
    ctx.endpoints = [tuple(p) for p in np.argwhere(skel & (nb == 1))]
    ctx.dt = dt
    # structure tensor on yn (integration scale = one acuity cycle)
    gy, gx = ndi.sobel(yn, axis=0) / 8.0, ndi.sobel(yn, axis=1) / 8.0
    sig = max(0.7, cfg["structure_sigma_cycles"] * cycle_px)
    jxx, jyy, jxy = (ndi.gaussian_filter(a, sig) for a in (gx * gx, gy * gy, gx * gy))
    ctx.energy = jxx + jyy
    ctx.theta_n = 0.5 * np.arctan2(2 * jxy, jxx - jyy)  # gradient direction = stripe normal (x right, y down)
    ctx.coherence = np.sqrt((jxx - jyy) ** 2 + 4 * jxy**2) / (ctx.energy + 1e-12)
    ctx.z = np.exp(2j * ctx.theta_n)
    w = ctx.energy[mask]
    zz = (w * ctx.z[mask]).sum()
    ctx.dominant_normal = 0.5 * math.atan2(zz.imag, zz.real)
    transect_runs(ctx, cfg["transect_count"])
    ctx.runs_black_raw_px, ctx.runs_white_raw_px = transect_runs(ctx, cfg["transect_count"], ctx.black_raw)
    return ctx
