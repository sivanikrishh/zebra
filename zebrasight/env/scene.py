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

Occlusion (``cond.occlusion_frac``; o20, o40, o60): neighbour-zebra silhouettes entering from the
side (~40% of the cover) plus grass blades rising from the region's lower edge, added until the
target share of the region is hidden. Occluded pixels get (textured) dry-grass colour - a
simplification: neighbours are not given stripes - and the region mask is updated so features
use only the visible coat.

Background (``cond.background``; default "none" for crops): the masked zebra is composited onto
a procedural savanna plate (short green grass, tall dry grass, bare dust: mean colour + multi-
scale texture noise, grade C) so its edges blend into realistic surroundings. ``make_plate`` is
standalone so the VR scene can use the same plates.
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


OCC_REF = 256  # occluders are thin: finer reference grid


def occlusion_map(mask: torch.Tensor, frac: float, seed: int, blade_width: float, blade_max_height: float,
                  neighbour_share: float) -> torch.Tensor:
    """(1,1,h,w) {0,1} occluder covering ~``frac`` of the mask: neighbour-zebra silhouettes
    (ellipses entering from the left/right edge, ~``neighbour_share`` of the cover) plus grass
    blades rising from the region's lower edge. Generated on a fixed normalised grid."""
    h, w = mask.shape[-2:]
    R = OCC_REF
    m = F.interpolate(mask.float().cpu(), size=(R, R), mode="nearest")[0, 0] > 0.5
    area = int(m.sum())
    occ = torch.zeros(R, R, dtype=torch.bool)
    if area == 0 or frac <= 0:
        return torch.zeros_like(mask)
    g = torch.Generator(device="cpu").manual_seed(seed)
    rnd = lambda lo, hi: lo + (hi - lo) * float(torch.rand(1, generator=g))  # noqa: E731
    yy, xx = torch.meshgrid(torch.arange(R, dtype=torch.float32), torch.arange(R, dtype=torch.float32), indexing="ij")
    target = frac * area
    covered = lambda: int((occ & m).sum())  # noqa: E731

    def silhouette():
        side = -0.1 * R if rnd(0, 1) < 0.5 else 1.1 * R
        cy, rx, ry = rnd(0.2, 0.9) * R, rnd(0.15, 0.35) * R, rnd(0.25, 0.5) * R
        return ((xx - side) / rx) ** 2 + ((yy - cy) / ry) ** 2 <= 1.0

    # neighbour silhouettes: ~neighbour_share of the cover, never overshooting the total
    for _ in range(50):
        if covered() >= neighbour_share * target:
            break
        ell = silhouette()
        if int(((occ | ell) & m).sum()) <= target:
            occ |= ell
    # grass blades rising from the lower edge of the region in each column (vectorised raster)
    cols = torch.nonzero(m.any(dim=0)).flatten()
    rows_idx = torch.arange(R).view(R, 1).expand(R, R)
    bottom = torch.where(m, rows_idx, torch.full_like(rows_idx, -1)).max(dim=0).values  # per column
    top_y = int(torch.nonzero(m.any(dim=1)).min())
    height_span = max(1, int(bottom.max()) - top_y)
    bw = max(1.0, blade_width * R)
    stalled = 0
    for _ in range(5000):  # bounded: blades only reach the lower part of the body
        if covered() >= target:
            break
        before = covered()
        x0 = float(cols[int(torch.randint(len(cols), (1,), generator=g))])
        base = float(bottom[int(x0)])
        hgt = rnd(0.2, 1.0) * blade_max_height * height_span
        tilt = rnd(-0.3, 0.3)
        blade = (yy <= base) & (yy >= base - hgt) & ((xx - (x0 + tilt * (base - yy))).abs() <= bw / 2)
        occ |= blade
        stalled = stalled + 1 if covered() == before else 0
        if stalled >= 20:  # blades saturated the lower body: add a neighbour silhouette instead
            occ |= silhouette()
            stalled = 0
    out = F.interpolate(occ.float()[None, None], size=(h, w), mode="bilinear", align_corners=False) > 0.5
    return out.to(mask.dtype).to(mask.device)


def grass_fill(shape, colour, texture: float, seed: int, device) -> torch.Tensor:
    """(1,3,h,w) linear-RGB grass: colour x (1 + texture x smooth noise), noise on the fixed grid."""
    h, w = shape
    g = torch.Generator(device="cpu").manual_seed(seed)
    n = gaussian_blur_fft(torch.randn(1, 1, REF, REF, generator=g), 1.0)
    n = n / n.std().clamp_min(1e-6)
    n = _resize(n, (h, w)).to(device)
    col = torch.tensor(colour, dtype=torch.float32, device=device).view(1, 3, 1, 1)
    return (col * (1.0 + texture * n)).clamp_min(0.0)


PLATES = ("short_green", "tall_dry", "bare_dust")


def make_plate(h: int, w: int, kind: str, mean_rgb, texture: float, seed: int, device=None) -> torch.Tensor:
    """(1,3,h,w) linear-RGB savanna plate: mean colour x (1 + texture x multi-scale noise).
    tall_dry adds vertical grass streaks. Generated on a fixed grid and resized, so it is the
    same pattern at any size; also usable directly for VR textures."""
    if kind not in PLATES:
        raise ValueError(f"unknown plate {kind!r}; choose from {PLATES}")
    g = torch.Generator(device="cpu").manual_seed(seed)
    n = torch.zeros(1, 1, REF, REF)
    for sigma, weight in ((8.0, 0.5), (2.0, 0.3), (0.7, 0.2)):  # multi-scale noise
        layer = gaussian_blur_fft(torch.randn(1, 1, REF, REF, generator=g), sigma)
        n = n + weight * layer / layer.std().clamp_min(1e-6)
    if kind == "tall_dry":  # vertical streaks: noise stretched along y
        streak = torch.randn(1, 1, 1, REF, generator=g).expand(1, 1, REF, REF)
        n = 0.6 * n + 0.8 * gaussian_blur_fft(streak.contiguous(), 0.7)
    n = n / n.std().clamp_min(1e-6)
    n = _resize(n, (h, w))
    col = torch.tensor(list(mean_rgb), dtype=torch.float32).view(1, 3, 1, 1)
    plate = (col * (1.0 + texture * n)).clamp_min(0.0)
    return plate.to(device) if device is not None else plate


@register_stage("scene")
class Scene(Stage):
    def forward(self, state: ViewState) -> ViewState:
        c = state.cond
        if not (c.shadow_frac > 0 or c.occlusion_frac > 0 or c.background != "none"):
            self.record(state, applied=False, azimuth_view_deg=c.azimuth_view_deg)
            return state
        rgb, mask = state.rgb, state.mask
        info: dict = {"applied": True}
        if c.background != "none":  # behind everything: replaces pixels outside the region mask
            mean = list(self.param(f"env.scene.background.plates.{c.background}"))
            tex = float(self.param("env.scene.background.texture"))
            plates = torch.cat([make_plate(*rgb.shape[-2:], c.background, mean, tex,
                                           scene_seed(state.meta, i, c.cond_id, c.seed, "background"), rgb.device)
                                for i in range(rgb.shape[0])])
            rgb = mask * rgb + (1.0 - mask) * plates
            info.update(background=c.background)
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
        if c.occlusion_frac > 0:
            outs, masks = [], []
            for i in range(rgb.shape[0]):
                mi = mask[i : i + 1]
                occ = occlusion_map(mi, c.occlusion_frac, scene_seed(state.meta, i, c.cond_id, c.seed, "occlusion"),
                                    float(self.param("env.scene.occlusion.blade_width")),
                                    float(self.param("env.scene.occlusion.blade_max_height")),
                                    float(self.param("env.scene.occlusion.neighbour_share")))
                grass = grass_fill(mi.shape[-2:], list(self.param("env.scene.occlusion.grass_rgb")),
                                   float(self.param("env.scene.occlusion.grass_texture")),
                                   scene_seed(state.meta, i, c.cond_id, c.seed, "grass"), rgb.device)
                outs.append(rgb[i : i + 1] * (1 - occ) + grass * occ)
                masks.append(mi * (1 - occ))  # features use only the visible coat
                if i == 0:
                    info["occlusion_frac_measured"] = float((occ * (mi > 0.5)).sum() / (mi > 0.5).sum().clamp_min(1))
            rgb, mask = torch.cat(outs), torch.cat(masks)
            info.update(occlusion_frac=c.occlusion_frac)
        state = state.evolve(rgb=rgb, mask=mask)
        self.record(state, **info)
        return state
