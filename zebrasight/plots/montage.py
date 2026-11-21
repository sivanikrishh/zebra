"""Distance montages from a percept store (internal: derived from CC BY-NC-ND images).

    python -m zebrasight.plots.montage <store.zarr> [n_zebras]

One figure per zebra: rows = flank / face x day (c0) / night (c9), columns = 6 of the 16
distances. Each tile is the stored luminance y on its own stored grid (shown with
nearest-neighbour upscaling, so coarse grids look blocky), scaled by the region's mean
luminance; outside the region mask is dimmed. Reads layout-v2 stores via PerceptStore.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import numpy as np

from zebrasight.config import REPO_ROOT
from zebrasight.pipeline.read_store import PerceptStore

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT_DIR = REPO_ROOT / "docs" / "figures"
COLS = (0, 3, 6, 9, 12, 15)  # distance indices: 2, 5.0, 12.6, 31.7, 79.6, 200 m
ROWS = (("flank", "c0"), ("flank", "c9"), ("face", "c0"), ("face", "c9"))
COND_NAME = {"c0": "day (c0)", "c9": "night (c9)"}


def tile(p: dict) -> np.ndarray:
    y, m = p["y"], p["mask"]
    ref = float(y[m].mean()) if m.any() else float(y.mean()) or 1.0
    img = np.clip(0.5 * y / max(ref, 1e-6), 0, 1)
    return np.where(m, img, 0.35 + 0.3 * img)


def montage(st: PerceptStore, rows, zebra_id: str, out: Path) -> Path:
    mine = rows[rows.zebra_id == zebra_id].sort_values("key")
    first = {r: mine[mine.region == r].key.iloc[0] for r in ("flank", "face")}
    fig, axes = plt.subplots(len(ROWS), len(COLS), figsize=(2.6 * len(COLS), 2.4 * len(ROWS)))
    for r, (region, cond) in enumerate(ROWS):
        for c, di in enumerate(COLS):
            p = st.get(first[region], cond, di)
            ax = axes[r, c]
            ax.imshow(tile(p), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            flags = ("D" if p["display_limited"] else "") + ("S" if p["source_limited"] else "")
            h, w = p["y"].shape
            ax.set_title(f"{p['distance_m']:.0f} m  {h}x{w}px {flags}", fontsize=9)
            ax.set_xticks([]), ax.set_yticks([])
            if c == 0:
                ax.set_ylabel(f"{region}\n{COND_NAME[cond]}", fontsize=11)
    fig.suptitle(f"{zebra_id}: zebra-eye percepts by distance   INTERNAL - NOT FOR PUBLIC DISPLAY (GZGC, CC BY-NC-ND 2.0)",
                 fontsize=12, color="#b00020")
    fig.text(0.5, 0.01, "S = source-limited (photo coarser than the eye), D = display-limited (size cap). "
             "Tiles scaled by region mean luminance; masked-out area dimmed.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=90)
    plt.close(fig)
    return out


def main() -> None:
    st = PerceptStore(sys.argv[1])
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    rows = st.rows()
    has = rows.groupby("zebra_id").region.agg(set)
    zebras = sorted(z for z, r in has.items() if {"flank", "face"} <= r)
    rng = np.random.default_rng(0)
    for z in rng.choice(zebras, size=min(n, len(zebras)), replace=False):
        print(montage(st, rows, str(z), OUT_DIR / f"pilot_montage_{z}.png"))


if __name__ == "__main__":
    main()
