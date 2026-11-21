"""Slide-ready environment montage for one zebra (internal: derived from CC BY-NC-ND images).

    python -m zebrasight.plots.env_montage [regions.parquet] [zebra_id] [region]

Rows: clear (c0) / hazy (c1) / dusty (c2) / dust storm (c3) / 30, 60, 75 deg view (c4-c6).
Columns: 5, 20, 50, 100, 200 m. Each tile is the zebra-eye luminance y on its stored grid
(nearest-neighbour display). One tone curve per column (linear luminance / the brightest tile at
that distance, then display gamma 2.2), so haze shows as a loss of contrast and a brightening
towards the airlight without clipping; outside the region is dimmed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

from zebrasight.config import REPO_ROOT, load_config, make_cond
from zebrasight.data.load import load_region
from zebrasight.pipeline import Pipeline

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROWS = (("c0", "clear"), ("c1", "hazy"), ("c2", "dusty"), ("c3", "dust storm"),
        ("c4", "view 30°"), ("c5", "view 60°"), ("c6", "view 75°"))
DISTANCES = (5.0, 20.0, 50.0, 100.0, 200.0)
OUT_DIR = REPO_ROOT / "docs" / "figures"


def render(row, out: Path) -> Path:
    pipe = Pipeline(list(load_config("grid").all_stages))
    img, mask, meta = load_region(row)
    res = {(c, d): pipe.run(img, float(row["cm_per_px"]), mask, make_cond(c, d), meta=meta)
           for c, _ in ROWS for d in DISTANCES}
    plt.rcParams.update({"font.size": 15})
    fig, axes = plt.subplots(len(ROWS), len(DISTANCES), figsize=(3.1 * len(DISTANCES), 2.3 * len(ROWS)))
    for j, d in enumerate(DISTANCES):
        # one tone curve per column: linear y / brightest tile at this distance, display gamma 2.2,
        # so haze shows as washed-out stripes instead of clipping to white
        ref = max(float(res[(c, d)].y.max()) for c, _ in ROWS)
        for i, (c, name) in enumerate(ROWS):
            p = res[(c, d)]
            y = p.y[0, 0].cpu().numpy()
            m = p.mask[0, 0].cpu().numpy() > 0.5
            t = np.clip(y / max(ref, 1e-6), 0, 1) ** (1 / 2.2)
            t = np.where(m, t, 0.25 + 0.5 * t)
            ax = axes[i, j]
            ax.imshow(t, cmap="gray", vmin=0, vmax=1, interpolation="nearest", aspect="equal")
            ax.set_xticks([]), ax.set_yticks([])
            if i == 0:
                ax.set_title(f"{d:.0f} m", fontsize=18)
            if j == 0:
                ax.set_ylabel(name, fontsize=17)
    fig.suptitle(f"What a zebra sees of {row['zebra_id']}'s {row['region']}: haze, dust and viewing angle\n"
                 "INTERNAL - NOT FOR PUBLIC DISPLAY (derived from GZGC, CC BY-NC-ND 2.0)",
                 fontsize=17, color="#222222")
    fig.text(0.5, 0.005, "Zebra-eye luminance (day). Haze/dust: Koschmieder + Angstrom (beta 0.002 / 0.01 / 0.03 per m);\n"
             "angle: coat compressed by cos(phi). Each column: linear luminance / brightest tile, display gamma 2.2.",
             ha="center", fontsize=11, color="#444444")
    fig.tight_layout(rect=(0, 0.035, 1, 0.96))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO_ROOT / "data" / "regions_demo5.parquet"
    reg = pd.read_parquet(path)
    zebra = sys.argv[2] if len(sys.argv) > 2 else sorted(reg.zebra_id.unique())[0]
    region = sys.argv[3] if len(sys.argv) > 3 else "flank"
    sel = reg[(reg.zebra_id == zebra) & (reg.region == region)]
    row = sel.assign(px=[(b[2] - b[0]) * (b[3] - b[1]) for b in sel.bbox_xyxy]).sort_values("px").iloc[-1]
    print(render(row.to_dict(), OUT_DIR / f"env_montage_{zebra}_{region}.png"))


if __name__ == "__main__":
    main()
