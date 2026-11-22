"""B2 pairwise alignment on sharp ORIGINAL crops, and B3 oracle / self alignment of percepts.

B2 (once per gallery-query pair and region, on the original photo crops):
  SIFT inside the mask (HotSpotter observer's descriptor), Lowe ratio test, RANSAC affine (query -> gallery);
  >= min_inliers_tps inliers: thin-plate spline refinement; < min_inliers_affine: canonical-frame
  fallback (query crop -> photo -> canonical -> gallery photo -> gallery crop).
  Stored as a BACKWARD map gallery-crop px -> query-crop px (what resampling needs) plus diagnostics:
  kind, inliers, residual_px (median over inliers), overlap (warped query mask on the gallery mask, as a
  fraction of the smaller mask), aligned_ok.
B3 oracle alignment: the stored map is applied to percepts at ANY distance, rescaled through each
  percept's grid (percept px -> crop px), so every distance is aligned exactly as the sharp photos were.
  self_align() recomputes SIFT + RANSAC on the percepts themselves (degrades with blur) for comparison.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import cv2
import numpy as np

from zebrasight.config import load_config
from zebrasight.metrics.hotspotter import describe, grey_u8
from zebrasight.positional.frame import Frame, apply, compose, crop_to_photo, invert, percept_to_crop


def _cfg():
    return load_config("positional").align


# ---- thin-plate spline (numpy) -------------------------------------------------------------------------

def _U(r2: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(r2 > 0, 0.5 * r2 * np.log(r2), 0.0)   # r^2 log r = 0.5 r^2 log r^2


def fit_tps(src: np.ndarray, dst: np.ndarray, lam: float = 1.0) -> dict:
    """TPS mapping src -> dst (n x 2 each), smoothing lam (px^2). Coordinates are normalised for conditioning."""
    src, dst = np.asarray(src, float), np.asarray(dst, float)
    mu, sc = src.mean(axis=0), src.std() + 1e-9
    s = (src - mu) / sc
    n = len(s)
    K = _U(((s[:, None, :] - s[None, :, :]) ** 2).sum(-1)) + (lam / sc**2) * np.eye(n)
    P = np.c_[np.ones(n), s]
    L = np.zeros((n + 3, n + 3))
    L[:n, :n], L[:n, n:], L[n:, :n] = K, P, P.T
    rhs = np.zeros((n + 3, 2))
    rhs[:n] = dst
    coef = np.linalg.solve(L, rhs)
    return {"ctrl": s.tolist(), "coef": coef.tolist(), "mu": mu.tolist(), "sc": float(sc)}


def apply_tps(t: dict, pts: np.ndarray) -> np.ndarray:
    ctrl, coef = np.asarray(t["ctrl"]), np.asarray(t["coef"])
    p = (np.asarray(pts, float).reshape(-1, 2) - np.asarray(t["mu"])) / t["sc"]
    n = len(ctrl)
    Kp = _U(((p[:, None, :] - ctrl[None, :, :]) ** 2).sum(-1))
    return Kp @ coef[:n] + np.c_[np.ones(len(p)), p] @ coef[n:]


# ---- stored pair transform -----------------------------------------------------------------------------

@dataclass
class PairAlign:
    kind: str                          # "tps" | "affine" | "canonical" | "none"
    M_g2q: np.ndarray                  # 2x3 affine gallery-crop px -> query-crop px (always set)
    tps: dict | None = None            # TPS gallery-crop px -> query-crop px (kind == "tps")
    inliers: int = 0
    residual_px: float = float("nan")
    overlap: float = float("nan")
    aligned_ok: bool = False
    rejected: str = ""                 # why a SIFT affine was rejected for the fallback ("" = not rejected)
    extra: dict = field(default_factory=dict)

    def g2q(self, pts: np.ndarray) -> np.ndarray:
        return apply_tps(self.tps, pts) if self.kind == "tps" else apply(self.M_g2q, pts)

    def to_row(self) -> dict:
        return {"kind": self.kind, "M_g2q": json.dumps(np.asarray(self.M_g2q).ravel().tolist()),
                "tps": json.dumps(self.tps) if self.tps is not None else "", "inliers": self.inliers,
                "residual_px": self.residual_px, "overlap": self.overlap, "aligned_ok": self.aligned_ok, "rejected": self.rejected}

    @classmethod
    def from_row(cls, r) -> "PairAlign":
        return cls(r["kind"], np.asarray(json.loads(r["M_g2q"])).reshape(2, 3),
                   json.loads(r["tps"]) if r["tps"] else None, int(r["inliers"]), float(r["residual_px"]),
                   float(r["overlap"]), bool(r["aligned_ok"]), str(r.get("rejected", "") or ""))


def _work(img_rgb: np.ndarray, mask: np.ndarray, max_side: int):
    s = min(1.0, max_side / max(mask.shape))
    if s < 1.0:
        size = (max(1, round(mask.shape[1] * s)), max(1, round(mask.shape[0] * s)))
        img_rgb = cv2.resize(img_rgb.astype(np.float32), size, interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST) > 0
    return img_rgb, mask, s


def crop_features(rgb, mask, cfg=None):
    """SIFT keypoints (in the crop's ORIGINAL px) and descriptors, computed once per crop and reused."""
    cfg = cfg or _cfg()
    img, m, s = _work(rgb, mask, cfg.max_side_px)
    pts, des = describe(grey_u8(img), m)
    return pts / s, des, s


def sift_matches(q_rgb, q_mask, g_rgb, g_mask, cfg=None, q_feat=None, g_feat=None):
    """-> (q_pts, g_pts) inlier correspondences in each crop's ORIGINAL px, and the RANSAC affine q->g (orig px).
    q_feat / g_feat: precomputed crop_features (then the images are not needed)."""
    cfg = cfg or _cfg()
    pa, da, sq = q_feat if q_feat is not None else crop_features(q_rgb, q_mask, cfg)
    pb, db, sg = g_feat if g_feat is not None else crop_features(g_rgb, g_mask, cfg)
    empty = (np.zeros((0, 2)), np.zeros((0, 2)), None)
    if da is None or db is None or len(da) < 2 or len(db) < 2:
        return empty
    m = cv2.BFMatcher(cv2.NORM_L2).knnMatch(da, db, k=2)
    good = [x[0] for x in m if len(x) == 2 and x[0].distance < cfg.ratio * x[1].distance]
    if len(good) < 3:
        return empty
    src = pa[[g.queryIdx for g in good]]
    dst = pb[[g.trainIdx for g in good]]
    M, inl = cv2.estimateAffine2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=cfg.ransac_px / min(sq, sg))
    if M is None or inl is None:
        return empty
    k = inl.ravel().astype(bool)
    return src[k], dst[k], M


def crop_record(rgb, mask, bbox=None, frame: Frame | None = None, cfg=None) -> dict:
    """Everything pair alignment needs from one crop, computed once: SIFT features (original px), the mask
    at working size (for the overlap check) and its scale, the crop's photo box and frame."""
    cfg = cfg or _cfg()
    feat = crop_features(rgb, mask, cfg)
    _, m_small, s = _work(rgb, mask, cfg.max_side_px)
    return {"feat": feat, "mask_small": m_small, "s": s, "bbox": bbox, "frame": frame}


def _overlap(pa: PairAlign, q: dict, g: dict) -> float:
    """Warped query mask on the gallery mask, both at working size: fraction of the smaller mask."""
    gm, qm = g["mask_small"], q["mask_small"]
    h, w = gm.shape
    yy, xx = np.mgrid[0:h, 0:w]
    g_orig = (np.c_[xx.ravel(), yy.ravel()] + 0.5) / g["s"] - 0.5
    q_small = (pa.g2q(g_orig) + 0.5) * q["s"] - 0.5
    wq = _remap(qm.astype(np.float32), q_small, (h, w), nearest=True) > 0.5
    small = min(wq.sum(), gm.sum())   # warped query area (a scaling map changes it), not the original
    return float((wq & gm).sum() / small) if small else 0.0


def _canonical_M(q: dict, g: dict):
    if q["frame"] is None or g["frame"] is None:
        return None
    # gallery crop -> photo -> canonical -> query photo -> query crop
    return compose(invert(crop_to_photo(q["bbox"])), q["frame"].Ainv, g["frame"].A, crop_to_photo(g["bbox"]))


def plausible_affine(M_g2q: np.ndarray, M_ref: np.ndarray | None, cfg=None) -> tuple[bool, str]:
    """Reject degenerate RANSAC affines (few inliers on a line give a squashed or mirrored map).
    Checks the linear part: no reflection, singular-value ratio <= max_anisotropy, and (if the canonical
    map is known) area scale within max_scale_ratio of the canonical map's area scale."""
    cfg = cfg or _cfg()
    A = np.asarray(M_g2q, float)[:, :2]
    if not np.all(np.isfinite(A)) or np.linalg.det(A) <= 0:
        return False, "reflection"
    s = np.linalg.svd(A, compute_uv=False)
    if s[0] / max(s[1], 1e-12) > cfg.max_anisotropy:
        return False, "anisotropy"
    if M_ref is not None:
        r = np.sqrt(abs(np.linalg.det(A)) / max(abs(np.linalg.det(np.asarray(M_ref, float)[:, :2])), 1e-12))
        if not (1 / cfg.max_scale_ratio <= r <= cfg.max_scale_ratio):
            return False, "scale"
    return True, ""


def align_records(q: dict, g: dict, cfg=None) -> PairAlign:
    cfg = cfg or _cfg()
    qp, gp, M_q2g = sift_matches(None, None, None, None, cfg, q["feat"], g["feat"])
    n = len(qp)
    M_can = _canonical_M(q, g)
    ok, reject = False, ""
    if n >= cfg.min_inliers_affine:
        ok, reject = plausible_affine(invert(M_q2g), M_can, cfg)
    if n >= cfg.min_inliers_affine and ok:
        M_g2q = invert(M_q2g)
        if n >= cfg.min_inliers_tps:
            t = fit_tps(gp, qp, cfg.tps_lambda)
            pa = PairAlign("tps", M_g2q, t, n)
        else:
            pa = PairAlign("affine", M_g2q, None, n)
        # residual in WORKING px (the scale RANSAC's threshold is set in), not original crop px
        pa.residual_px = float(np.median(np.linalg.norm(pa.g2q(gp) - qp, axis=1)) * q["s"])
    elif M_can is not None:
        pa = PairAlign("canonical", M_can, None, n)
    else:
        pa = PairAlign("none", np.array([[1.0, 0, 0], [0, 1.0, 0]]), None, n)
    pa.rejected = reject
    pa.overlap = _overlap(pa, q, g)
    pa.aligned_ok = bool(pa.kind in ("tps", "affine") and pa.overlap >= cfg.min_overlap
                         and pa.residual_px <= cfg.max_residual_px)
    return pa


def align_pair(q_rgb, q_mask, g_rgb, g_mask, q_frame: Frame | None = None, q_bbox=None,
               g_frame: Frame | None = None, g_bbox=None, cfg=None) -> PairAlign:
    """Convenience wrapper for two crops (the batch run uses crop_record + align_records)."""
    cfg = cfg or _cfg()
    return align_records(crop_record(q_rgb, q_mask, q_bbox, q_frame, cfg), crop_record(g_rgb, g_mask, g_bbox, g_frame, cfg), cfg)


# ---- B3: warping ---------------------------------------------------------------------------------------

def _remap(img: np.ndarray, map_xy: np.ndarray, shape, nearest: bool) -> np.ndarray:
    mx = map_xy[:, 0].reshape(shape).astype(np.float32)
    my = map_xy[:, 1].reshape(shape).astype(np.float32)
    return cv2.remap(np.ascontiguousarray(img, dtype=np.float32), mx, my,
                     cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)


def warp_backward(q_img: np.ndarray, pa: PairAlign, g_shape, nearest: bool = False) -> np.ndarray:
    """Query crop image resampled on the gallery crop grid (original crops, B2 diagnostics)."""
    h, w = g_shape
    yy, xx = np.mgrid[0:h, 0:w]
    return _remap(q_img, pa.g2q(np.c_[xx.ravel(), yy.ravel()]), (h, w), nearest)


def oracle_warp(q_img: np.ndarray, pa: PairAlign, q_crop_hw, g_crop_hw, g_percept_hw, nearest: bool = False) -> np.ndarray:
    """B3: query PERCEPT resampled on the gallery PERCEPT grid using the pair's stored (sharp-photo) alignment.

    gallery percept px -> gallery crop px -> (stored map) -> query crop px -> query percept px."""
    h, w = g_percept_hw
    yy, xx = np.mgrid[0:h, 0:w]
    g_crop = apply(percept_to_crop(g_crop_hw, g_percept_hw), np.c_[xx.ravel(), yy.ravel()])
    q_crop = pa.g2q(g_crop)
    q_pix = apply(invert(percept_to_crop(q_crop_hw, q_img.shape[:2])), q_crop)
    return _remap(q_img, q_pix, (h, w), nearest)


def self_align(q_y: np.ndarray, q_mask: np.ndarray, g_y: np.ndarray, g_mask: np.ndarray, cfg=None):
    """Alignment recomputed on the percepts themselves: -> (warped y, warped mask, inliers)."""
    cfg = cfg or _cfg()

    def disp(y, m):
        v = y[m] if m.any() else y.ravel()
        lo, hi = np.percentile(v, [1, 99]) if v.size else (0.0, 1.0)
        g = np.clip((y - lo) / max(hi - lo, 1e-9), 0, 1)
        return np.repeat(g[..., None], 3, axis=-1)

    qp, gp, M = sift_matches(disp(q_y, q_mask), q_mask, disp(g_y, g_mask), g_mask, cfg)
    n = len(qp)
    M_q2g = M if n >= cfg.min_inliers_affine else np.array([[g_y.shape[1] / q_y.shape[1], 0, 0],
                                                             [0, g_y.shape[0] / q_y.shape[0], 0]])
    size = (g_y.shape[1], g_y.shape[0])
    wy = cv2.warpAffine(q_y.astype(np.float32), M_q2g, size, flags=cv2.INTER_LINEAR)
    wm = cv2.warpAffine(q_mask.astype(np.uint8), M_q2g, size, flags=cv2.INTER_NEAREST) > 0
    return wy, wm, n
