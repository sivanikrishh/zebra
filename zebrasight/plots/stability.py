"""Within-zebra stability of stripe statistics: python -m zebrasight.plots.stability [features.parquet]

For 10 random zebras (seed 0), each photo's F7 spacing (period) and F1 width at 2 m (c0), next to the spread
over all zebras. If identity lived in these numbers, each zebra's dots would sit in a narrow band of the
population spread. Panel titles give the within-zebra share of variance (pooled within-zebra var / total var;
1 = no identity information, 0 = perfect). -> outputs/figures/stability_F7_F1.png
"""

from __future__ import annotations

import sys

import matplotlib
import numpy as np
import pandas as pd

from zebrasight.config import REPO_ROOT

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

FEATS = (("F7_period_cm", "F7 stripe spacing (period, cm)"), ("F1_width_cm_median", "F1 stripe width (median, cm)"))
N_ZEBRAS, SEED = 10, 0
OUT = REPO_ROOT / "outputs" / "figures" / "stability_F7_F1.png"


def within_share(d: pd.DataFrame, col: str) -> float:
    x = d[["zebra_id", col]].dropna()
    x = x[x.groupby("zebra_id")[col].transform("size") >= 2]
    if len(x) < 3:
        return float("nan")
    resid = x[col] - x.groupby("zebra_id")[col].transform("mean")
    return float((resid**2).sum() / ((x[col] - x[col].mean()) ** 2).sum())


def main() -> None:
    src = sys.argv[1] if len(sys.argv) > 1 else REPO_ROOT / "outputs" / "features" / "features_c0_v2.parquet"
    df = pd.read_parquet(src)
    df = df[(df.cond == "c0") & (df.d_index == 0)]
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    for i, region in enumerate(("flank", "face")):
        d = df[df.region == region]
        n_photos = d.groupby("zebra_id").size()
        pool = sorted(n_photos[n_photos >= 2].index)
        pick = sorted(np.random.default_rng(SEED).choice(pool, size=min(N_ZEBRAS, len(pool)), replace=False))
        for j, (col, lab) in enumerate(FEATS):
            ax = axes[i, j]
            allv = d[col].dropna()
            lo, hi = np.percentile(allv, [5, 95])
            ax.axhspan(lo, hi, color="#999999", alpha=0.15, lw=0)
            ax.axhline(allv.median(), color="#666666", lw=1, ls="--")
            jit = np.random.default_rng(1)
            ax.scatter(jit.uniform(-0.25, 0.25, len(allv)), allv, s=10, color="#888888", alpha=0.5)
            for k, z in enumerate(pick, 1):
                zz = d[d.zebra_id == z]
                for split, mk, colr in (("gallery", "o", "#1b6ca8"), ("query", "s", "#d9822b")):
                    v = zz[zz.split == split][col].dropna()
                    ax.scatter(k + jit.uniform(-0.12, 0.12, len(v)), v, marker=mk, s=40, color=colr,
                               label=f"{split} ({'day 1' if split == 'gallery' else 'day 2'})" if k == 1 else None)
                v = zz[col].dropna()
                if len(v) >= 2:
                    ax.plot([k, k], [v.min(), v.max()], color="#333333", lw=1)
            ax.set_xticks(range(len(pick) + 1))
            ax.set_xticklabels(["all\nzebras"] + [z.replace("IBEIS_PZ_", "PZ ") for z in pick], fontsize=10, rotation=0)
            ax.set_ylabel(lab)
            ax.set_title(f"{region.capitalize()}: within-zebra share of variance = {within_share(d, col):.2f} "
                         f"(all {d.zebra_id.nunique()} zebras)", fontsize=13)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
    axes[0, 0].legend(loc="upper right", fontsize=10, frameon=False)
    fig.suptitle("Is a zebra's stripe spacing / width its own? Each photo at 2 m, clear day (c0)\n"
                 "grey band = 5-95% of all photos; 1.0 = a zebra varies as much as the whole population",
                 fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=110)
    plt.close(fig)
    print(OUT)


if __name__ == "__main__":
    main()
