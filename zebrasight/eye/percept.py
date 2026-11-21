"""Final ViewState -> Percept: distance-adaptive storage resolution and the rho-field (F2).

- y = L cone (rod at night), s = S cone (None at night).
- Storage grid: deg_per_px_store = max(working_grid, 1 / (samples_per_cycle x A)), with
  A the regime's peak acuity and samples_per_cycle = 2.5, so the finest resolvable stripe
  period always gets 2.5 pixels. Never upsampled: a coarser (source-limited) grid is kept.
  If the longer side would exceed ``percept.max_side`` (1024) the grid is coarsened to fit
  and ``display_limited`` = True; the log records the effective resolvable cpd.
- Resampling: Gaussian pre-filter + bilinear (``ops.gaussian_resize``), mask by
  ``ops.resize_mask``.
- rho = local BLACK FRACTION: pixels of y below the per-image Otsu threshold (computed over
  the mask, or the whole image if the mask is empty), box-averaged over a window of one
  acuity cycle, 1 / A degrees.
"""

from __future__ import annotations

from omegaconf import DictConfig

from zebrasight.eye.ops import box_mean, gaussian_resize, otsu_threshold, resize_mask
from zebrasight.eye.streak import LIGHT_TO_REGIME
from zebrasight.types import Percept, ViewState


def black_fraction(y, mask, deg_per_px: float, acuity_cpd: float):
    t = otsu_threshold(y, mask)
    black = (y < t.view(-1, 1, 1, 1)).to(y.dtype)
    window = round((1.0 / acuity_cpd) / deg_per_px)
    return box_mean(black, window), t, window


def storage_size(h: int, w: int, dpp: float, acuity: float, grid: float, samples_per_cycle: float,
                 max_side: int) -> tuple[tuple[int, int], bool]:
    """Output (h, w) and whether the max_side cap limits resolution."""
    target = max(grid, 1.0 / (samples_per_cycle * acuity))
    k = min(1.0, dpp / target)  # <= 1: only downsample
    size = (max(1, round(h * k)), max(1, round(w * k)))
    limited = max(size) > max_side
    if limited:
        c = max_side / max(size)
        size = (max(1, round(size[0] * c)), max(1, round(size[1] * c)))
    return size, limited


def make_percept(state: ViewState, eye_cfg: DictConfig) -> Percept:
    if state.rod is not None:
        y, s = state.rod, None
    elif state.cones is not None:
        y, s = state.cones[:, 0:1], state.cones[:, 1:2]
    else:
        raise RuntimeError("eye stage produced neither cones nor rod")

    regime = LIGHT_TO_REGIME[state.cond.light]
    acuity = float(eye_cfg.acuity[regime].value)
    spc = float(eye_cfg.percept.samples_per_cycle.value)
    max_side = int(eye_cfg.percept.max_side.value)
    grid = float(eye_cfg.working_grid.deg_per_px.value)

    h, w = y.shape[-2:]
    size, display_limited = storage_size(h, w, state.deg_per_px, acuity, grid, spc, max_side)
    mask = state.mask
    if size != (h, w):
        y = gaussian_resize(y, size)
        s = gaussian_resize(s, size) if s is not None else None
        mask = resize_mask(mask, size)
    dpp = state.deg_per_px * h / size[0]
    resolvable = min(acuity, 1.0 / (spc * dpp))

    rho, thr, window = black_fraction(y, mask, dpp, acuity)
    log = state.log + [{"stage": "percept", "max_side": max_side, "samples_per_cycle": spc,
                        "size_in": (h, w), "size_out": tuple(size), "deg_per_px": dpp,
                        "nyquist_cpd": 0.5 / dpp, "effective_cpd": resolvable, "acuity_cpd": acuity,
                        "display_limited": display_limited, "rho_window_px": window,
                        "otsu_threshold": thr.tolist()}]
    return Percept(y=y, s=s, rho=rho, mask=mask, deg_per_px=dpp, meta=state.meta, log=log,
                   source_limited=state.source_limited, display_limited=display_limited)
