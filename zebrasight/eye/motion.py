"""Motion-blur stage: temporal integration of a moving target.

A target moving at ``cond.speed_mps`` across the line of sight at distance d sweeps
    theta = degrees(speed * t_int / d)
during the integration time t_int (day 30 ms, sweep 15-60; night 100 ms; grade C, no horse
flicker-fusion value found; dawn/dusk use the day value). The blur is a line kernel of
length theta / deg_per_px pixels along ``cond.motion_dir_deg`` (0 = horizontal,
counter-clockwise, image y up), anti-aliased by sub-pixel sampling, normalised to sum 1.
Applied to all receptor channels (blur happens at the receptors, before neural filtering;
eye_order puts this stage before streak and csf).
"""

from __future__ import annotations

import math

import torch

from zebrasight.eye.ops import fft_conv
from zebrasight.registry import Stage, register_stage
from zebrasight.types import ViewState


def line_kernel(length_px: float, angle_deg: float, device=None) -> torch.Tensor:
    """Odd-sized 2-D kernel: a line segment of the given length, bilinear-splatted."""
    half = length_px / 2.0
    size = 2 * math.ceil(half) + 1
    c = size // 2
    n = max(2, int(math.ceil(length_px * 8)))
    t = torch.linspace(-half, half, n, device=device)
    a = math.radians(angle_deg)
    x = c + t * math.cos(a)
    y = c - t * math.sin(a)  # image rows grow downwards
    k = torch.zeros(size, size, device=device)
    x0, y0 = x.floor().long(), y.floor().long()
    fx, fy = x - x0, y - y0
    for dx, dy, wt in ((0, 0, (1 - fx) * (1 - fy)), (1, 0, fx * (1 - fy)), (0, 1, (1 - fx) * fy), (1, 1, fx * fy)):
        xi, yi = (x0 + dx).clamp(0, size - 1), (y0 + dy).clamp(0, size - 1)
        k.index_put_((yi, xi), wt, accumulate=True)
    return k / k.sum()


def blur_length_px(speed_mps: float, t_int_s: float, distance_m: float, dpp: float) -> float:
    return math.degrees(speed_mps * t_int_s / distance_m) / dpp


@register_stage("motion")
class Motion(Stage):
    def forward(self, state: ViewState) -> ViewState:
        c = state.cond
        key = "temporal.integration_ms_night" if c.light == "night" else "temporal.integration_ms_day"
        t_ms = float(self.param(key))
        length = blur_length_px(c.speed_mps, t_ms / 1000.0, c.distance_m, state.deg_per_px)
        applied = length >= 1.0
        if applied:
            k = line_kernel(length, c.motion_dir_deg, device=state.device)
            if state.rod is not None:
                state = state.evolve(rod=fft_conv(state.rod, k))
            else:
                state = state.evolve(cones=fft_conv(state.cones, k))
        self.record(state, speed_mps=c.speed_mps, integration_ms=t_ms, direction_deg=c.motion_dir_deg,
                    blur_px=length, applied=applied)
        return state
