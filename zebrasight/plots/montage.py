"""Distance montages from a percept store (internal: derived from CC BY-NC-ND images).

    python -m zebrasight.plots.montage <store.zarr> [n_zebras]

One figure per zebra: rows = flank / face x day (c0) / night (c9), columns = 6 of the 16
distances. Each tile is the stored luminance y on its own stored grid (shown with
nearest-neighbour upscaling, so coarse grids look blocky), scaled by the region's mean
luminance; outside the region mask is dimmed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import numpy as np
import zarr

from zebrasight.config import REPO_ROOT

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT_DIR = REPO_ROOT / "docs" / "figures"
COLS = (0, 3, 6, 9, 12, 15)  # distance indices: 2, 5.0, 12.6, 31.7, 79.6, 200 m
ROWS = (("flank", 0), ("flank", 3), ("face", 0), ("face", 3))  # (region, cond index): c0 day, c9 night
COND_NAME = {0: "day (c0)", 3: "night (c9)"}


def tile(g, ci: int, di: int) -> np.ndarray:
    h, w = (int(v) for v in g["shape"][ci, di])
    y = g["y"][ci, di, :h, :w].astype(np.float32)
    m = g["mask"][ci, di, :h, :w] > 0
    ref = float(y[m].mean()) if m.any() else float(y.mean()) or 1.0
    img = np.clip(0.5 * y / max(ref, 1e-6), 0, 1)
    return np.where(m, img, 0.35 + 0.3 * img)


def montage(store: Path, zebra_id: str, out: Path) -> Path:
    root = zarr.open_group(str(store), mode="r")
    rows = root["rows"]
    dists = root.attrs["distances_m"]
    keys = {k: rows[k] for k in rows.group_keys() if rows[k].attrs.get("zebra_id") == zebra_id}
    by_region = {}
    for k, g in sorted(keys.items()):
        by_region.setdefault(g.attrs["region"], (k, g))  # first view of each region
    fig, axes = plt.subplots(len(ROWS), len(COLS), figsize=(2.6 * len(COLS), 2.4 * len(ROWS)))
    for r, (region, ci) in enumerate(ROWS):
        key, g = by_region[region]
        for c, di in enumerate(COLS):
            ax = axes[r, c]
            ax.imshow(tile(g, ci, di), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            h, w = (int(v) for v in g["shape"][ci, di])
            flags = ("D" if g["display_limited"][ci, di] else "") + ("S" if g["source_limited"][ci, di] else "")
            ax.set_title(f"{dists[di]:.0f} m  {h}x{w}px {flags}", fontsize=9)
            ax.set_xticks([]), ax.set_yticks([])
            if c == 0:
                ax.set_ylabel(f"{region}\n{COND_NAME[ci]}", fontsize=11)
    fig.suptitle(f"{zebra_id}: zebra-eye percepts by distance   INTERNAL - NOT FOR PUBLIC DISPLAY (GZGC, CC BY-NC-ND 2.0)",
                 fontsize=12, color="#b00020")
    fig.text(0.5, 0.01, "S = source-limited (photo coarser than the eye), D = display-limited (storage cap). "
             "Tiles scaled by region mean luminance; masked-out area dimmed.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=90)
    plt.close(fig)
    return out


def main() -> None:
    store = Path(sys.argv[1])
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    root = zarr.open_group(str(store), mode="r")
    rows = root["rows"]
    regions_by_zebra = {}
    for k in rows.group_keys():
        a = rows[k].attrs
        if a.get("complete"):
            regions_by_zebra.setdefault(a["zebra_id"], set()).add(a["region"])
    zebras = sorted(z for z, r in regions_by_zebra.items() if {"flank", "face"} <= r)
    rng = np.random.default_rng(0)
    for z in rng.choice(zebras, size=min(n, len(zebras)), replace=False):
        print(montage(store, str(z), OUT_DIR / f"pilot_montage_{z}.png"))


if __name__ == "__main__":
    main()
