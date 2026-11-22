"""Checkpoint 1: data summary, QC montage, NaN rates, size-rule vs resolution-limit distances.

    python -m zebrasight.features.qc <store> <features.parquet>

Writes outputs/qc/qc_montage_c0.png (photo-derived: internal), outputs/features/nan_rate_c0.csv,
outputs/features/size_limits_c0.csv and prints the summaries.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

from zebrasight.config import REPO_ROOT, load_config
from zebrasight.features.common import config, prepare
from zebrasight.pipeline.read_store import PerceptStore

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

QC_D = (0, 5, 10, 14)  # 2, 9.3, 43.1, 147 m


def data_summary(st: PerceptStore) -> str:
    r = st.rows()
    valid = r[r.get("valid", True) != False]  # noqa: E712
    lines = [f"zebras {valid.zebra_id.nunique()}, rows {len(valid)} (skipped valid==False: {len(r) - len(valid)})",
             valid.groupby(["region", "split"]).size().unstack().to_string(),
             "conditions present: " + ", ".join(sorted(set(sum(valid.conds.tolist(), [])))) + " (c7 = alias of c8)"]
    return "\n".join(lines)


def limited_counts(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["region", "d_index", "distance_m"])
    return g.agg(n=("key", "size"), display_limited=("display_limited", "sum"),
                 source_limited=("source_limited", "sum"), size_rule=("short_side_px", lambda s: int((s < config()["min_short_side_px"]).sum()))).reset_index()


def nan_rates(df: pd.DataFrame) -> pd.DataFrame:
    fcols = [c for c in df.columns if c[:2] in ("F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "F9") or c.startswith("F10")]
    t = df.groupby("distance_m")[fcols].apply(lambda x: x.isna().mean()).T
    t.columns = [f"{c:.3g}" for c in t.columns]
    return t


def size_limits(df: pd.DataFrame) -> pd.DataFrame:
    """Per photo x region: first distance where the short side < min size; Melin-style resolution
    limit d = 2w / tan(1/A) from the photo's stripe width w at 2 m (F1, day acuity 23.3 cpd)."""
    acuity = float(load_config("eye").acuity.photopic.value)
    mn = config()["min_short_side_px"]
    recs = []
    for (key, region), g in df.sort_values("d_index").groupby(["key", "region"]):
        small = g[g.short_side_px < mn]
        d_size = float(small.distance_m.iloc[0]) if len(small) else math.inf
        w2 = float(g[g.d_index == 0].F1_width_cm_median.iloc[0])
        d_res = (2 * w2 / 100.0) / math.tan(math.radians(1.0 / acuity)) if np.isfinite(w2) else math.nan
        recs.append({"key": key, "region": region, "d_size_rule_m": d_size, "width_2m_cm": w2, "d_resolution_m": d_res})
    return pd.DataFrame(recs)


def montage(st: PerceptStore, keys: list[str], out: Path) -> Path:
    fig, axes = plt.subplots(len(keys), 3 * len(QC_D), figsize=(2.0 * 3 * len(QC_D), 2.0 * len(keys)))
    for i, key in enumerate(keys):
        for j, di in enumerate(QC_D):
            p = st.get(key, "c0", di)
            ctx = prepare(p, st.distances[di])
            y = p["y"] / (np.median(p["y"][p["mask"]]) or 1)
            panels = [np.where(p["mask"], y, np.nan)]
            if ctx.valid:
                panels += [np.where(ctx.mask, ctx.black.astype(float), np.nan), None]
            for k in range(3):
                ax = axes[i, 3 * j + k]
                ax.set_xticks([]), ax.set_yticks([])
                if k == 0:
                    ax.imshow(panels[0], cmap="gray", interpolation="nearest")
                    ax.set_title(f"{st.distances[di]:.3g} m  {p['y'].shape[1]}x{p['y'].shape[0]}", fontsize=7)
                elif not ctx.valid:
                    ax.text(0.5, 0.5, "NaN\n" + ctx.reason, ha="center", va="center", fontsize=6, transform=ax.transAxes)
                elif k == 1:
                    ax.imshow(panels[1], cmap="gray_r", interpolation="nearest")
                    ax.set_title("Otsu binary", fontsize=7)
                else:
                    ax.imshow(ctx.skel, cmap="gray_r", interpolation="nearest")
                    if ctx.junctions:
                        jy, jx = zip(*ctx.junctions)
                        ax.plot(jx, jy, "o", ms=2, color="#d62728")
                    if ctx.endpoints:
                        ey, ex = zip(*ctx.endpoints)
                        ax.plot(ex, ey, "s", ms=1.5, color="#1f77b4")
                    ax.set_title(f"skeleton J{len(ctx.junctions)} E{len(ctx.endpoints)}", fontsize=7)
    fig.suptitle("QC c0 flank: percept | binary | pruned skeleton (red = junctions, blue = endings)  "
                 "INTERNAL (GZGC-derived)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=80)
    plt.close(fig)
    return out


def main() -> None:
    st = PerceptStore(sys.argv[1])
    df = pd.read_parquet(sys.argv[2])
    out = REPO_ROOT / "outputs"
    print("== data summary ==\n" + data_summary(st))
    lc = limited_counts(df)
    print("\n== display / source / size-rule limited percepts per distance (c0) ==")
    print(lc.pivot_table(index="distance_m", columns="region", values=["display_limited", "source_limited", "size_rule"]).to_string())
    nr = nan_rates(df)
    nr.to_csv(out / "features" / "nan_rate_c0.csv")
    print("\n== NaN rate per feature x distance (c0, flank+face) ==\n" + nr.round(2).to_string())
    f8 = df.groupby("distance_m").F8_shadow_count_per100cm2.agg(nan=lambda s: s.isna().mean(), zero=lambda s: (s == 0).mean())
    print("\n== F8 shadow stripes: NaN and zero rates ==\n" + f8.round(2).T.to_string())
    sl = size_limits(df)
    sl.to_csv(out / "features" / "size_limits_c0.csv", index=False)
    print("\n== size rule vs resolution limit (per photo; m) ==")
    for region, g in sl.groupby("region"):
        fin = g.d_size_rule_m[np.isfinite(g.d_size_rule_m)]
        print(f"{region:5s}: size rule (short side < {config()['min_short_side_px']} px) first at median "
              f"{fin.median():.0f} m (range {fin.min():.0f}-{fin.max():.0f}; {int((~np.isfinite(g.d_size_rule_m)).sum())} photos never); "
              f"resolution limit 2w/tan(1/A) median {g.d_resolution_m.median():.0f} m (range {g.d_resolution_m.min():.0f}-{g.d_resolution_m.max():.0f}), "
              f"stripe width at 2 m median {g.width_2m_cm.median():.1f} cm")
    keys = sorted(df[df.region == "flank"].key.unique())
    rng = np.random.default_rng(0)
    pick = list(rng.choice(keys, size=6, replace=False))
    print("\nQC montage:", montage(st, pick, out / "qc" / "qc_montage_c0.png"))


if __name__ == "__main__":
    main()
