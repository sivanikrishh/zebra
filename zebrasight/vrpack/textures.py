"""Texture export: one 16-layer PNG set per zebra x region x condition, from a layout-2 percept store.

Display transform (documented for V in vr/export/README.md):
    colour (L + S):  rgb_lin = LS_to_rgb @ [L, S]          (S upsampled from half resolution, bilinear)
    night (rod):     rgb_lin = (rod, rod, rod)
    exposure:        one per zebra x region over ALL its conditions and distances,
                     E = 1 / max(rgb_lin) -> no layer clips, and layers blend without brightness jumps
    png:             sRGB-encoded clip(E * rgb_lin, 0, 1); alpha = region mask (255 inside)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from zebrasight.pipeline.read_store import PerceptStore
from zebrasight.vrpack.params import matrices

CONDS = ("c0", "c8", "c9", "c1", "c2", "c3")


def srgb_encode(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1 / 2.4) - 0.055)


def display_linear(p: dict, d: np.ndarray) -> np.ndarray:
    """(h, w, 3) linear display RGB for one percept dict from PerceptStore.get."""
    y = p["y"]
    if p["s"] is None:  # night: rod, grey
        return np.repeat(y[..., None], 3, axis=-1)
    s = torch.from_numpy(p["s"])[None, None]
    s = F.interpolate(s, size=y.shape, mode="bilinear", align_corners=False)[0, 0].numpy()
    ls = np.stack([y, s], axis=-1)  # (h, w, 2)
    return ls @ d.T


def export(store_path: Path, out_dir: Path, keys: list[str], conds=CONDS, source="GZGC (Wild Me GZGC-IBEIS)",
           licence="CC BY-NC-ND 2.0") -> dict:
    st = PerceptStore(store_path)
    rows = st.rows().set_index("key")
    _, _, _, _, d, _ = matrices()
    sets = []
    for key in keys:
        r = rows.loc[key]
        layers = {c: [display_linear(st.get(key, c, i), d) for i in range(len(st.distances))] for c in conds}
        peak = max(float(a.max()) for c in conds for a in layers[c])
        exposure = 1.0 / max(peak, 1e-6)
        for c in conds:
            files = []
            for i, rgb in enumerate(layers[c]):
                p = st.get(key, c, i)
                rgba = np.concatenate([srgb_encode(exposure * rgb), p["mask"][..., None].astype(np.float32)], axis=-1)
                rel = Path(r.zebra_id) / r.region / c / f"d{i:02d}_{st.distances[i]:g}m.png"
                path = out_dir / rel
                path.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray((rgba * 255 + 0.5).astype(np.uint8), mode="RGBA").save(path, optimize=True)
                files.append({"distance_m": st.distances[i], "file": rel.as_posix(), "size_px": list(rgb.shape[1::-1]),
                              "deg_per_px": p["deg_per_px"], "display_limited": p["display_limited"],
                              "source_limited": p["source_limited"]})
            sets.append({"zebra_id": r.zebra_id, "region": r.region, "condition": c, "image_id": r.image_id,
                         "ann_id": int(r.ann_id), "cm_per_px_source": float(r.cm_per_px),
                         "exposure": exposure, "layers": files})
    return {"distance_bins_m": st.distances, "conditions": list(conds), "aliases": {"c7": "c8"},
            "source_dataset": source, "licence": licence, "sets": sets,
            "encoding": "RGBA PNG, sRGB-encoded display colour of the zebra's percept (dichromat display matrix); "
                        "alpha = region mask; linear = srgb_decode(rgb) / exposure"}


def write_json(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1), encoding="utf-8")
