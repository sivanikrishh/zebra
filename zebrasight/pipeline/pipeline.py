"""Build the ordered stage list from config and run it: photo crop -> Percept."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import kornia
import torch
from omegaconf import DictConfig
from torch import nn

import zebrasight.env  # noqa: F401  (registers scene, atmosphere)
import zebrasight.eye  # noqa: F401  (registers geometry, eye and eye sub-stages)
from zebrasight.config import load_config
from zebrasight.registry import STAGES
from zebrasight.eye.percept import make_percept
from zebrasight.types import COMPUTE_DTYPE, Percept, ViewCond, ViewState, default_device

PIPELINE_ORDER = ("geometry", "scene", "atmosphere", "eye")


def check_order(names: Sequence[str]) -> None:
    """Stages must be a subset of PIPELINE_ORDER, in that order, and must end with the eye."""
    unknown = [n for n in names if n not in PIPELINE_ORDER]
    if unknown:
        raise ValueError(f"unknown top-level stages {unknown}; allowed: {PIPELINE_ORDER}")
    idx = [PIPELINE_ORDER.index(n) for n in names]
    if idx != sorted(set(idx)):
        raise ValueError(f"stages {list(names)} break the order {PIPELINE_ORDER}")
    if not names or names[-1] != "eye":
        raise ValueError("the pipeline must end with the eye stage")


class Pipeline(nn.Module):
    def __init__(
        self,
        stages: Sequence[str],
        eye_cfg: DictConfig | None = None,
        device: torch.device | str | None = None,
    ):
        super().__init__()
        stages = list(stages)
        check_order(stages)
        self.eye_cfg = eye_cfg if eye_cfg is not None else load_config("eye")
        self.device = torch.device(device) if device is not None else default_device()
        self.stages = nn.ModuleList(STAGES[n](self.eye_cfg) for n in stages)

    @classmethod
    def from_grid(cls, phase: str = "phase1", **kwargs: Any) -> "Pipeline":
        grid = load_config("grid")
        return cls(list(grid[phase].stages), **kwargs)

    @property
    def stage_names(self) -> list[str]:
        return [s.stage_name for s in self.stages]

    @torch.no_grad()
    def run(
        self,
        img: torch.Tensor,
        cm_per_px: float,
        mask: torch.Tensor,
        cond: ViewCond,
        meta: dict[str, Any] | None = None,
        img_is_linear: bool = False,
    ) -> Percept:
        """Run all stages on a batch.

        img: (B,3,H,W) or (3,H,W) in [0,1]; sRGB unless ``img_is_linear``.
        mask: (B,1,H,W), (1,H,W) or (H,W) region mask.
        meta: IDs (zebra_id, image_id, session_id, region); copied, never read by the maths.
        """
        img = _as_bchw(img).to(self.device, COMPUTE_DTYPE)
        mask = _as_bchw(mask).to(self.device, COMPUTE_DTYPE)
        if img.shape[1] != 3:
            raise ValueError(f"img must have 3 channels, got {tuple(img.shape)}")
        if mask.shape[0] != img.shape[0] or mask.shape[-2:] != img.shape[-2:]:
            raise ValueError(f"mask {tuple(mask.shape)} does not match img {tuple(img.shape)}")
        rgb = img if img_is_linear else kornia.color.rgb_to_linear_rgb(img.clamp(0, 1))

        state = ViewState(rgb=rgb, mask=mask, cond=cond, cm_per_px=float(cm_per_px), meta=dict(meta or {}))
        state.log.append({"stage": "input", "cm_per_px": float(cm_per_px), "srgb_decoded": not img_is_linear,
                          "device": str(self.device), "cond": cond.__dict__.copy()})
        for stage in self.stages:
            state = stage(state)
        return make_percept(state, self.eye_cfg)


def _as_bchw(t: torch.Tensor) -> torch.Tensor:
    if t.ndim == 2:
        return t[None, None]
    if t.ndim == 3:
        return t[None]
    if t.ndim == 4:
        return t
    raise ValueError(f"expected 2-4 dims, got {tuple(t.shape)}")
