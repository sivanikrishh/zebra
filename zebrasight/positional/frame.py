"""B1 canonical body frame.

A photo's full-body mask (Mask R-CNN, v2 label PNG > 0) defines the frame:
- body axis = first principal axis (PCA) of the mask pixel coordinates;
- u runs along the axis from the HEAD end (u = 0) to the tail end (u = U), using the facing direction
  (left-side views face the viewer's left; head_left), checked against the OWLv2 head box when present;
- v runs across, from the back (v = 0) to the hooves (v = V);
- both are normalised by the mask's extent: body length -> U units, body height -> V units.

All maps are 2x3 affine matrices on (x, y) pixel coordinates (pixel centres at integer + 0.5 offset
conventions are folded into the crop/percept maps below). A crop at photo offset (x0, y0) and a percept
resampled from that crop share the frame through composed affines, so any region at any distance maps
into the same 100 x 60 grid.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from zebrasight.config import load_config


def _cfg():
    return load_config("positional")


@dataclass
class Frame:
    A: np.ndarray       # 2x3: photo px (x, y) -> canonical (u, v)
    Ainv: np.ndarray    # 2x3: canonical (u, v) -> photo px (x, y)
    length_px: float    # body length along the axis, photo px
    height_px: float    # body height across the axis, photo px
    angle_deg: float    # axis angle (image x towards image y), in (-90, 90]
    head_left: bool

    def to_dict(self) -> dict:
        return {"A": self.A.ravel().tolist(), "Ainv": self.Ainv.ravel().tolist(), "length_px": self.length_px,
                "height_px": self.height_px, "angle_deg": self.angle_deg, "head_left": self.head_left}

    @classmethod
    def from_dict(cls, d: dict) -> "Frame":
        return cls(np.asarray(d["A"], float).reshape(2, 3), np.asarray(d["Ainv"], float).reshape(2, 3),
                   float(d["length_px"]), float(d["height_px"]), float(d["angle_deg"]), bool(d["head_left"]))


def to3(M: np.ndarray) -> np.ndarray:
    return np.vstack([M, [0.0, 0.0, 1.0]])


def compose(*Ms: np.ndarray) -> np.ndarray:
    """compose(A, B, C) = A o B o C (C applied first)."""
    out = np.eye(3)
    for M in Ms:
        out = out @ to3(M)
    return out[:2]


def invert(M: np.ndarray) -> np.ndarray:
    return np.linalg.inv(to3(M))[:2]


def apply(M: np.ndarray, pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, float).reshape(-1, 2)
    return pts @ M[:, :2].T + M[:, 2]


def fit_frame(body: np.ndarray, origin_xy=(0.0, 0.0), head_left: bool = True, grid_uv=None) -> Frame:
    """body: (H, W) bool mask whose pixel (row r, col c) sits at photo (origin_x + c, origin_y + r)."""
    U, V = grid_uv if grid_uv is not None else _cfg().frame.grid_uv
    rows, cols = np.nonzero(body)
    if rows.size < 3:
        raise ValueError("body mask too small for a frame")
    p = np.c_[cols + origin_xy[0], rows + origin_xy[1]].astype(float)
    c = p.mean(axis=0)
    w, vec = np.linalg.eigh(np.cov((p - c).T))
    e1 = vec[:, np.argmax(w)]
    if e1[0] < 0 or (e1[0] == 0 and e1[1] < 0):
        e1 = -e1                                   # axis points to image right (within +/- 90 deg)
    e2 = np.array([-e1[1], e1[0]])                 # perpendicular, pointing image-down
    a, b = (p - c) @ e1, (p - c) @ e2
    a0, a1, b0, b1 = a.min(), a.max() + 1.0, b.min(), b.max() + 1.0
    L, Hh = a1 - a0, b1 - b0
    su, sv = U / L, V / Hh
    if head_left:   # head at small a: u = su (a - a0)
        ru, tu = su * e1, -su * (e1 @ c + a0)
    else:           # head at large a: u = su (a1 - a)
        ru, tu = -su * e1, su * (e1 @ c + a1)
    rv, tv = sv * e2, -sv * (e2 @ c + b0)
    A = np.array([[ru[0], ru[1], tu], [rv[0], rv[1], tv]])
    return Frame(A, invert(A), float(L), float(Hh), float(np.degrees(np.arctan2(e1[1], e1[0]))), bool(head_left))


def head_check(frame: Frame, head_box_xyxy) -> bool | None:
    """True if the head box centre maps to the front half of the frame (u < U/2); None without a box."""
    if head_box_xyxy is None or not np.all(np.isfinite(head_box_xyxy)):
        return None
    x0, y0, x1, y1 = head_box_xyxy
    u = apply(frame.A, [[(x0 + x1) / 2, (y0 + y1) / 2]])[0, 0]
    return bool(u < _cfg().frame.grid_uv[0] / 2)


def crop_to_photo(bbox_xyxy) -> np.ndarray:
    """Crop pixel (x, y) -> photo pixel."""
    return np.array([[1.0, 0.0, float(bbox_xyxy[0])], [0.0, 1.0, float(bbox_xyxy[1])]])


def percept_to_crop(crop_hw, percept_hw) -> np.ndarray:
    """Percept pixel (x, y) -> crop pixel: the geometry stage resamples the whole crop to the percept grid
    (pixel-centre aligned): x_c + 0.5 = (x_p + 0.5) * w_c / w_p."""
    sy, sx = crop_hw[0] / percept_hw[0], crop_hw[1] / percept_hw[1]
    return np.array([[sx, 0.0, 0.5 * sx - 0.5], [0.0, sy, 0.5 * sy - 0.5]])


def percept_to_canonical(frame: Frame, bbox_xyxy, crop_hw, percept_hw) -> np.ndarray:
    return compose(frame.A, crop_to_photo(bbox_xyxy), percept_to_crop(crop_hw, percept_hw))


def warp_to_canonical(img: np.ndarray, M_img2can: np.ndarray, grid_uv=None, order: str = "linear") -> np.ndarray:
    """Sample an image (H, W) on the canonical grid (V, U) at cell centres (u + 0.5, v + 0.5)."""
    U, V = grid_uv if grid_uv is not None else _cfg().frame.grid_uv
    inv = invert(M_img2can)
    uu, vv = np.meshgrid(np.arange(U) + 0.5, np.arange(V) + 0.5)
    xy = apply(inv, np.c_[uu.ravel(), vv.ravel()])
    mx, my = xy[:, 0].reshape(V, U).astype(np.float32), xy[:, 1].reshape(V, U).astype(np.float32)
    interp = cv2.INTER_NEAREST if order == "nearest" else cv2.INTER_LINEAR
    return cv2.remap(np.ascontiguousarray(img, dtype=np.float32), mx, my, interp,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)


def canonical_points(grid_cells_uv, grid_uv=None) -> np.ndarray:
    """Centres (u, v) of a coarser cell grid (nu x nv cells) over the canonical frame, row-major (v, u)."""
    U, V = grid_uv if grid_uv is not None else _cfg().frame.grid_uv
    nu, nv = grid_cells_uv
    uu, vv = np.meshgrid((np.arange(nu) + 0.5) * U / nu, (np.arange(nv) + 0.5) * V / nv)
    return np.c_[uu.ravel(), vv.ravel()]


def part_mask(name: str, grid_uv=None) -> np.ndarray:
    """(V, U) bool mask of a fixed body part rule (configs/positional.yaml frame.parts)."""
    c = _cfg().frame
    U, V = grid_uv if grid_uv is not None else c.grid_uv
    u0, u1, v0, v1 = c.parts[name]
    m = np.zeros((V, U), bool)
    m[int(v0 * V / c.grid_uv[1]):int(v1 * V / c.grid_uv[1]), int(u0 * U / c.grid_uv[0]):int(u1 * U / c.grid_uv[0])] = True
    return m
