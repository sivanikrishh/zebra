"""Scene stage: cast shadows, occlusion and background, in LINEAR RGB (colour matters here).

Runs after geometry (working grid) and before the atmosphere. Every pattern is generated on a
fixed reference grid in coordinates normalised to the region and then resized to the current
grid, from a seed hashed from (image_id, region, cond_id, base seed) - NOT the distance - so the
same tree shadow or grass sits on the same part of the coat at every distance.

Physical layering: background plate (behind) -> cast shadow on the coat -> occluders in front
(not shadowed). Conditions without an effect (c0-c9) leave the image untouched.

Cast shadow (``cond.shadow_frac``; s25, s50): a smooth random field (tree canopy / neighbour
shape) thresholded at the quantile that puts ``shadow_frac`` of the region in shadow, then a
Gaussian penumbra. Shadow light is skylight only: luminance x (1 - darkening), tinted bluish
(tint normalised to unit luminance), all grade C (configs/env.yaml scene.shadow).
"""

from __future__ import annotations

import hashlib

import torch
import torch.nn.functional as F

from zebrasight.eye.ops import gaussian_blur_fft
from zebrasight.registry import Stage, register_stage
from zebrasight.types import ViewState

SRGB_Y = (0.2126, 0.7152, 0.0722)
REF = 128  # reference grid for pattern generation (normalised region coordinates)


def scene_seed(meta: dict, i: int, cond_id: str, base_seed: int, what: str) -> int:
    def item(k):
        v = meta.get(k)
        return v[i] if isinstance(v, (list, tuple)) else v

    key = repr((item("image_id"), item("region"), str(cond_id), int(base_seed), what))
    return int.from_bytes(hashlib.blake2b(key.encode("utf-8"), digest_size=8).digest(), "little") & (2**63 - 1)


def _resize(x: torch.Tensor, size) -> torch.Tensor:
    return F.interpolate(x, size=size, mode="bilinear", align_corners=False)


def shadow_map(mask: torch.Tensor, frac: float, seed: int, blob_scale: float, penumbra: float) -> torch.Tensor:
    """(1,1,h,w) soft shadow in [0,1] (1 = full shadow) covering ``frac`` of the mask."""
    h, w = mask.shape[-2:]
    g = torch.Generator(device="cpu").manual_seed(seed)
    field = torch.randn(1, 1, REF, REF, generator=g)
    field = gaussian_blur_fft(field, blob_scale * REF)
    field = _resize(field.to(mask.device), (h, w))
    inside = field[mask > 0.5]
    if inside.numel() == 0 or frac <= 0:
        return torch.zeros_like(mask)
    thr = torch.quantile(inside.float(), 1.0 - frac) if frac < 1 else inside.min() - 1
    hard = (field > thr).to(mask.dtype)
    return gaussian_blur_fft(hard, penumbra * w).clamp(0.0, 1.0)


def apply_shadow(rgb: torch.Tensor, s: torch.Tensor, darkening: float, tint) -> torch.Tensor:
    y = sum(wc * tc for wc, tc in zip(SRGB_Y, tint))
    factor = torch.tensor([(1.0 - darkening) * tc / y for tc in tint], dtype=rgb.dtype, device=rgb.device)
    return rgb * (1.0 - s + s * factor.view(1, 3, 1, 1))


@register_stage("scene")
class Scene(Stage):
    def forward(self, state: ViewState) -> ViewState:
        c = state.cond
        if not (c.shadow_frac > 0 or c.occlusion_frac > 0 or c.background != "none"):
            self.record(state, applied=False, azimuth_view_deg=c.azimuth_view_deg)
            return state
        rgb, mask = state.rgb, state.mask
        info: dict = {"applied": True}
        if c.shadow_frac > 0:
            dark = float(self.param("env.scene.shadow.darkening"))
            tint = list(self.param("env.scene.shadow.tint_rgb"))
            outs = []
            for i in range(rgb.shape[0]):
                s = shadow_map(mask[i : i + 1], c.shadow_frac,
                               scene_seed(state.meta, i, c.cond_id, c.seed, "shadow"),
                               float(self.param("env.scene.shadow.blob_scale")),
                               float(self.param("env.scene.shadow.penumbra_sigma")))
                outs.append(apply_shadow(rgb[i : i + 1], s, dark, tint))
                if i == 0:
                    info["shadow_frac_measured"] = float(((s >= 0.5) & (mask[i : i + 1] > 0.5)).sum()
                                                         / (mask[i : i + 1] > 0.5).sum().clamp_min(1))
            rgb = torch.cat(outs)
            info.update(shadow_frac=c.shadow_frac, shadow_darkening=dark, shadow_tint=tint)
        state = state.evolve(rgb=rgb, mask=mask)
        self.record(state, **info)
        return state
