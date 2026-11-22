"""Recognition metrics: python -m zebrasight.metrics.run <features.parquet> <out_dir> [--cond c0] [--n-boot 200]

Writes <out_dir>/metrics_<cond>.parquet (tidy: region x cond x distance x observer x metric, with CI
columns), <out_dir>/dstar_<cond>.csv and, for c0, the Checkpoint-2 chart. Needs only the Parquet.
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd

from zebrasight.config import REPO_ROOT
from zebrasight.metrics.core import (auc, boot_auc, ci, eer_dprime_tar, fano_bits, feature_columns, herd_top1,
                                     pairs, similarity, split)

OBSERVERS = ("raw_pixel", "interpretable_nn", "interpretable_lda", "spatial_nn", "spatial_lda")
HERDS = (5, 20, 50)
SEED = 0


def evaluate(df: pd.DataFrame, cond: str, n_boot: int, draws: int = 200) -> tuple[pd.DataFrame, dict]:
    cols = feature_columns(df)
    recs, boots = [], {}
    for region in sorted(df.region.unique()):
        for (di, d), sub in df[(df.region == region) & (df.cond == cond)].groupby(["d_index", "distance_m"]):
            sp = split(sub)
            zebras = np.intersect1d(np.unique(sp.gz), np.unique(sp.qz))
            base = {"region": region, "cond": cond, "d_index": int(di), "distance_m": float(d),
                    "display_limited_frac": float(sub.display_limited.mean()),
                    "source_limited_frac": float(sub.source_limited.mean()),
                    "invalid_frac": float(sub[cols["F"]].isna().all(axis=1).mean())}
            for obs in OBSERVERS:
                rng = np.random.default_rng(SEED)
                S = similarity(sp, obs, cols)
                s, l, qz = pairs(sp, S, rng)
                a = auc(s, l)
                b = boot_auc(s, l, qz, zebras, n_boot, np.random.default_rng(SEED + 1))
                boots[(region, int(di), obs)] = (b, zebras, s, l, qz)
                lo, hi = ci(b)
                # label-shuffle null (query labels permuted across zebras)
                sp_null = split(sub)
                sp_null.qz = np.random.default_rng(SEED + 2).permutation(sp_null.qz)
                sn, ln, _ = pairs(sp_null, S, np.random.default_rng(SEED))
                null = auc(sn, ln)
                extra = eer_dprime_tar(s, l)
                rows = [("auc", a, lo, hi), ("auc_null", null, math.nan, math.nan),
                        ("eer", extra["eer"], math.nan, math.nan), ("dprime", extra["dprime"], math.nan, math.nan),
                        ("tar_far1", extra["tar_at_far"], math.nan, math.nan)]
                for n in HERDS:
                    acc = herd_top1(sp, S, n, draws, np.random.default_rng(SEED + 3))
                    nn = min(n, len(zebras))
                    bits = np.array([fano_bits(x, nn) for x in acc])
                    rows += [(f"top1_n{n}", float(acc.mean()), *ci(acc)), (f"bits_n{n}", float(bits.mean()), *ci(bits)),
                             (f"chance_n{n}", 1.0 / nn, math.nan, math.nan)]
                for metric, v, lo_, hi_ in rows:
                    recs.append({**base, "observer": obs, "metric": metric, "value": v, "ci_low": lo_, "ci_high": hi_})
    return pd.DataFrame(recs), boots


def farthest(dist: np.ndarray, ok: np.ndarray) -> float:
    """Farthest distance whose criterion holds (NaN if never)."""
    idx = np.flatnonzero(ok)
    return float(dist[idx.max()]) if idx.size else math.nan


def crossings(dist, ok) -> list[float]:
    """Every distance where the criterion switches from true to false (curve dips and recovers)."""
    return [float(dist[i]) for i in range(len(ok) - 1) if ok[i] and not ok[i + 1]]


def dstar_table(m: pd.DataFrame, boots: dict, size_rule: dict) -> pd.DataFrame:
    out = []
    for (region, obs), g in m.groupby(["region", "observer"]):
        a = g[g.metric == "auc"].sort_values("distance_m")
        dist = a.distance_m.to_numpy()
        rec = {"region": region, "observer": obs,
               "dstar_auc_m (lower CI >= 0.80)": farthest(dist, a.ci_low.to_numpy() >= 0.80),
               "crossings_0.80": crossings(dist, a.ci_low.to_numpy() >= 0.80),
               "d_reliable_m (lower CI >= 0.95)": farthest(dist, a.ci_low.to_numpy() >= 0.95)}
        for n in (5, 20):
            bt = g[g.metric == f"bits_n{n}"].sort_values("distance_m")
            rec[f"dstar_bits_n{n}_m"] = farthest(bt.distance_m.to_numpy(), bt.ci_low.to_numpy() >= math.log2(n) - 1e-9)
        t = g[g.metric == "top1_n20"].sort_values("distance_m")
        rec["d75_n20_m (mean top-1 >= 0.75)"] = farthest(t.distance_m.to_numpy(), t.value.to_numpy() >= 0.75)
        rec["d75_ci_m"] = (farthest(t.distance_m.to_numpy(), t.ci_low.to_numpy() >= 0.75),
                           farthest(t.distance_m.to_numpy(), t.ci_high.to_numpy() >= 0.75))
        rec["size_rule_median_m"] = size_rule.get(region)
        out.append(rec)
    return pd.DataFrame(out)


def crossover(boots: dict, dist: list[float], obs: str, n_boot: int) -> dict:
    """Paired bootstrap over zebras present in both regions: dAUC(d) = face - flank; crossover distance."""
    zebras = np.intersect1d(boots[("face", 0, obs)][1], boots[("flank", 0, obs)][1])
    rng = np.random.default_rng(SEED + 4)
    delta = np.full((n_boot, len(dist)), np.nan)
    draws = [rng.choice(zebras, size=len(zebras), replace=True) for _ in range(n_boot)]
    for di in range(len(dist)):
        per = {}
        for region in ("face", "flank"):
            _, _, s, l, qz = boots[(region, di, obs)]
            idx_by_z = {z: np.flatnonzero(qz == z) for z in zebras}
            per[region] = [auc(s[np.concatenate([idx_by_z[z] for z in d])], l[np.concatenate([idx_by_z[z] for z in d])])
                           for d in draws]
        delta[:, di] = np.array(per["face"]) - np.array(per["flank"])
    logd = np.log(np.asarray(dist))
    xs = []
    for b in range(n_boot):
        x = math.nan
        for i in range(len(dist) - 1):
            if delta[b, i] > 0 >= delta[b, i + 1]:
                f = delta[b, i] / (delta[b, i] - delta[b, i + 1])
                x = math.exp(logd[i] + f * (logd[i + 1] - logd[i]))
                break
        xs.append(x)
    xs = np.array(xs)
    return {"observer": obs, "crossover_median_m": float(np.nanmedian(xs)) if np.isfinite(xs).any() else math.nan,
            "crossover_ci_m": ci(xs), "share_boot_with_crossing": float(np.isfinite(xs).mean()),
            "p_face_better": (delta > 0).mean(axis=0).round(3).tolist(),
            "delta_auc_median": np.nanmedian(delta, axis=0).round(3).tolist()}


def chart(m: pd.DataFrame, size_rule: dict, out: Path, main_obs="interpretable_nn") -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colour = {"face": "#eb6834", "flank": "#2a78d6"}
    plt.rcParams.update({"font.size": 14})
    fig, ax = plt.subplots(figsize=(12, 7))
    for region in ("face", "flank"):
        for obs, ls, lab in ((main_obs, "-", "F1-F10"), ("spatial_nn", ":", "F + layout + minutiae"),
                             ("raw_pixel", "--", "raw pixels")):
            a = m[(m.region == region) & (m.observer == obs) & (m.metric == "auc")].sort_values("distance_m")
            ax.plot(a.distance_m, a.value, ls, color=colour[region], lw=2.4 if obs == main_obs else 1.6,
                    label=f"{region}: {lab}")
            if obs == main_obs:
                ax.fill_between(a.distance_m, a.ci_low, a.ci_high, color=colour[region], alpha=0.18, lw=0)
        nl = m[(m.region == region) & (m.observer == main_obs) & (m.metric == "auc_null")].sort_values("distance_m")
        ax.plot(nl.distance_m, nl.value, color=colour[region], lw=0.8, alpha=0.6)
        if size_rule.get(region):
            ax.axvline(size_rule[region], color=colour[region], lw=1, ls="-.", alpha=0.8)
            ax.text(size_rule[region] * 0.97, 0.43, f"{region} size rule\n(median photo)", rotation=90,
                    ha="right", va="bottom", fontsize=10, color=colour[region])
    # shade distance bins where most percepts are display- or source-limited (pooled over regions)
    lim = m[(m.observer == main_obs) & (m.metric == "auc")].groupby("distance_m")[["display_limited_frac", "source_limited_frac"]].mean()
    dist = lim.index.to_numpy()
    edges = np.sqrt(dist[:-1] * dist[1:])
    edges = np.r_[dist[0] / 1.15, edges, dist[-1] * 1.15]
    for i, dd in enumerate(dist):
        if lim.source_limited_frac.iloc[i] > 0.5 or lim.display_limited_frac.iloc[i] > 0.5:
            ax.axvspan(edges[i], edges[i + 1], color="#999999", alpha=0.15, lw=0)
    ax.axhline(0.5, color="#555555", lw=1)
    ax.text(2.1, 0.505, "chance / label-shuffle null", fontsize=10, color="#555555")
    for yv, txt in ((0.80, "usable 0.80"), (0.95, "reliable 0.95")):
        ax.axhline(yv, color="#888888", lw=1, ls="--")
        ax.text(160, yv + 0.005, txt, fontsize=10, color="#555555")
    ax.set_xscale("log")
    ax.set_ylim(0.4, 1.01)
    ax.set_xlim(1.8, 230)
    ax.set_xlabel("Viewing distance (m)")
    ax.set_ylabel("ROC-AUC, genuine vs impostor (95% CI)")
    ax.set_title("Identity in the zebra-eye percept: face vs flank, clear day (c0), 50 zebras\n"
                 "grey bands: >50% of percepts source/display-limited; dash-dot: size rule bites (median photo)")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(loc="lower left", fontsize=11, ncol=2, frameon=False)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("features")
    ap.add_argument("out_dir")
    ap.add_argument("--cond", default="c0")
    ap.add_argument("--n-boot", type=int, default=200)
    a = ap.parse_args()
    t0 = time.perf_counter()
    df = pd.read_parquet(a.features)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    m, boots = evaluate(df, a.cond, a.n_boot)
    m.to_parquet(out / f"metrics_{a.cond}.parquet", index=False)
    sl = pd.read_csv(REPO_ROOT / "outputs" / "features" / "size_limits_c0.csv")
    size_rule = {r: float(g.d_size_rule_m.replace(np.inf, np.nan).median()) for r, g in sl.groupby("region")}
    tab = dstar_table(m, boots, size_rule)
    dist = sorted(df.distance_m.unique())
    xo = pd.DataFrame([crossover(boots, dist, o, a.n_boot) for o in OBSERVERS])
    tab.to_csv(out / f"dstar_{a.cond}.csv", index=False)
    xo.to_csv(out / f"crossover_{a.cond}.csv", index=False)
    if a.cond == "c0":
        print("chart:", chart(m, size_rule, REPO_ROOT / "outputs" / "figures" / "auc_face_flank_c0.png"))
    pd.set_option("display.width", 250)
    print(tab.to_string(index=False))
    print(xo[["observer", "crossover_median_m", "crossover_ci_m", "share_boot_with_crossing"]].to_string(index=False))
    print(f"done in {time.perf_counter() - t0:.0f} s")


if __name__ == "__main__":
    main()
