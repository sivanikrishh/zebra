"""Final ViewState -> Percept: output resolution and the rho-field (feature F2).

- y = L cone (rod at night), s = S cone (None at night).
- Area-downsample y, s and mask so the longer side is at most ``percept.max_side`` (256);
  deg_per_px is updated to the new pixel size.
- rho = local BLACK FRACTION: pixels of y below the per-image Otsu threshold (computed over
  the mask, or the whole image if the mask is empty), box-averaged over a window of one
  acuity cycle, 1 / A degrees (A = regime acuity).
"""

from __future__ import annotations

from omegaconf import DictConfig

from zebrasight.eye.ops import area_resize, box_mean, otsu_threshold, resize_mask
from zebrasight.eye.streak import LIGHT_TO_REGIME
from zebrasight.types import Percept, ViewState


def black_fraction(y, mask, deg_per_px: float, acuity_cpd: float):
    t = otsu_threshold(y, mask)
    black = (y < t.view(-1, 1, 1, 1)).to(y.dtype)
    window = round((1.0 / acuity_cpd) / deg_per_px)
    return box_mean(black, window), t, window


def make_percept(state: ViewState, eye_cfg: DictConfig) -> Percept:
    if state.rod is not None:
        y, s = state.rod, None
    elif state.cones is not None:
        y, s = state.cones[:, 0:1], state.cones[:, 1:2]
    else:
        raise RuntimeError("eye stage produced neither cones nor rod")

    h, w = y.shape[-2:]
    max_side = int(eye_cfg.percept.max_side.value)
    dpp, mask = state.deg_per_px, state.mask
    if max(h, w) > max_side:
        k = max_side / max(h, w)
        size = (max(1, round(h * k)), max(1, round(w * k)))
        y = area_resize(y, size)
        s = area_resize(s, size) if s is not None else None
        mask = resize_mask(mask, size)
        dpp = dpp * h / size[0]

    regime = LIGHT_TO_REGIME[state.cond.light]
    acuity = float(eye_cfg.acuity[regime].value)
    rho, thr, window = black_fraction(y, mask, dpp, acuity)
    log = state.log + [{"stage": "percept", "max_side": max_side, "size_in": (h, w), "size_out": tuple(y.shape[-2:]),
                        "deg_per_px": dpp, "rho_window_px": window, "rho_acuity_cpd": acuity,
                        "otsu_threshold": thr.tolist()}]
    return Percept(y=y, s=s, rho=rho, mask=mask, deg_per_px=dpp, meta=state.meta, log=log,
                   source_limited=state.source_limited)
