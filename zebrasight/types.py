"""Core data types passed between pipeline stages.

Tensors are batched (B, C, H, W), float32 for compute. Use ``Percept.for_storage()`` to get
float16 copies for disk. One ``ViewCond`` applies to the whole batch.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Literal

import torch

Light = Literal["day", "dawn", "dusk", "night"]

COMPUTE_DTYPE = torch.float32
STORAGE_DTYPE = torch.float16


def default_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass(frozen=True)
class ViewCond:
    """Viewing condition for one batch: distance, light and atmosphere."""

    distance_m: float
    cond_id: str
    light: Light
    luminance_cd_m2: float
    field_az_deg: float = 0.0  # where the target sits in the viewer's visual field
    field_el_deg: float = 0.0
    azimuth_view_deg: float = 0.0  # target body orientation away from side-on
    beta_haze: float = 0.0  # extinction coefficient, 1/m
    speed_mps: float = 0.0  # relative motion of target across the line of sight
    motion_dir_deg: float = 0.0  # image-plane direction of that motion, 0 = horizontal, CCW
    seed: int = 0


@dataclass
class ViewState:
    """Everything a stage reads and writes. Stages return a (possibly new) ViewState."""

    rgb: torch.Tensor | None  # (B,3,H,W) linear RGB; None once the eye has consumed it
    mask: torch.Tensor  # (B,1,H,W) region mask, resampled with the image
    cond: ViewCond
    deg_per_px: float | None = None
    az_map: torch.Tensor | None = None  # (B,1,H,W) visual-field azimuth, deg
    el_map: torch.Tensor | None = None  # (B,1,H,W) visual-field elevation, deg, up positive
    cones: torch.Tensor | None = None  # (B,2,H,W) [L, S]
    rod: torch.Tensor | None = None  # (B,1,H,W) night only
    acuity_map: torch.Tensor | None = None  # (B,1,H,W) local resolution limit, cycles/deg (streak -> csf)
    cm_per_px: float | None = None  # source sampling on the coat
    meta: dict[str, Any] = field(default_factory=dict)  # zebra_id, image_id, session_id, region
    log: list[dict[str, Any]] = field(default_factory=list)
    source_limited: bool = False
    display_limited: bool = False  # working-size cap coarsened the grid (geometry)

    @property
    def device(self) -> torch.device:
        return self.mask.device

    def evolve(self, **changes: Any) -> "ViewState":
        return replace(self, **changes)


@dataclass
class Percept:
    """Output of the pipeline: what the zebra eye delivers."""

    y: torch.Tensor  # (B,1,h,w) luminance signal: L cone by day, rod at night
    s: torch.Tensor | None  # (B,1,h,w) S cone; None when below the colour threshold
    rho: torch.Tensor | None  # (B,1,h,w) local black fraction (feature F2): y < Otsu, box-averaged over one acuity cycle
    mask: torch.Tensor
    deg_per_px: float
    meta: dict[str, Any]
    log: list[dict[str, Any]]
    source_limited: bool = False
    display_limited: bool = False  # storage size cap coarsened the grid below 2.5 px per acuity cycle

    def for_storage(self) -> "Percept":
        def half(t: torch.Tensor | None) -> torch.Tensor | None:
            return None if t is None else t.detach().to(STORAGE_DTYPE).cpu()

        return replace(self, y=half(self.y), s=half(self.s), rho=half(self.rho), mask=half(self.mask))
