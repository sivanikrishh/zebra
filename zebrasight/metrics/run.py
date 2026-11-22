"""Recognition metrics: python -m zebrasight.metrics.run <features.parquet> <out_dir> --cond c0
       [--embeddings megadesc_c0.parquet ...] [--pairs hotspotter_c0.parquet] [--n-boot 200]

Writes <out_dir>/metrics_<cond>.parquet (tidy: region x cond x distance x observer x metric, with CI
columns), boot_<cond>.npz (per-zebra-draw AUCs, paired across distances), family_<cond>.parquet
(single-family AUC = leading clues), dstar_<cond>.csv and crossover_<cond>.csv.
--pairs: HotSpotter pair table (amendment A1) -> observer "hotspotter", similarity = log1p(inliers);
its layout_r column (A4) -> observer "registered_layout".
Pseudo-region "face_owlv2" = face rows whose mask came from OWLv2 (not the silhouette-tip fallback).
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd

from zebrasight.metrics.core import (auc, boot_auc, ci, eer_dprime_tar, fano_bits, feature_columns, herd_top1,
                                     pairs, similarity, similarity_from_pairs, split)

OBSERVERS = ("raw_pixel", "interpretable_nn", "interpretable_lda", "spatial_nn", "megadesc", "hotspotter", "registered_layout")
FAMILIES = [f"F{i}" for i in range(1, 11)]
HERDS = (5, 20, 50)
SEED = 0


def regions_of(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    out = {r: df[df.region == r] for r in ("face", "flank") if (df.region == r).any()}
    if "face" in out:
        out["face_owlv2"] = out["face"][out["face"].mask_method == "owlv2"]
    return out


def evaluate(df: pd.DataFrame, cond: str, n_boot: int, draws: int = 200, observers=OBSERVERS,
             hs: pd.DataFrame | None = None):
    cols = feature_columns(df)
    has_e = any(c.startswith("E_") for c in df.columns)
    observers = [o for o in observers if (o != "megadesc" or has_e) and (o not in ("hotspotter", "registered_layout") or hs is not None)]
    if hs is not None and "layout_r" not in hs:
        observers = [o for o in observers if o != "registered_layout"]
    if hs is not None:
        hs = hs[hs.cond == cond]
    recs, fam, boots = [], [], {}
    for region, rdf in regions_of(df[df.cond == cond]).items():
        for (di, d), sub in rdf.groupby(["d_index", "distance_m"]):
            sp = split(sub)
            zebras = np.intersect1d(np.unique(sp.gz), np.unique(sp.qz))
            base = {"region": region, "cond": cond, "d_index": int(di), "distance_m": float(d),
                    "display_limited_frac": float(sub.display_limited.mean()),
                    "source_limited_frac": float(sub.source_limited.mean()),
                    "invalid_frac": float(sub[cols["F"]].isna().all(axis=1).mean())}
            for obs in observers:
                if obs in ("hotspotter", "registered_layout"):
                    S = similarity_from_pairs(sp, hs[(hs.region == region.split("_")[0]) & (hs.d_index == di)],
                                              "inliers" if obs == "hotspotter" else "layout_r")
                else:
                    S = similarity(sp, obs, cols)
                s, l, qz = pairs(sp, S, np.random.default_rng(SEED))
                b = boot_auc(s, l, qz, zebras, n_boot, np.random.default_rng(SEED + 1))  # same draws at every d
                boots[(region, int(di), obs)] = (b, zebras, s, l, qz)
                sp_null = split(sub)
                sp_null.qz = np.random.default_rng(SEED + 2).permutation(sp_null.qz)
                sn, ln, _ = pairs(sp_null, S, np.random.default_rng(SEED))
                extra = eer_dprime_tar(s, l)
                rows = [("auc", auc(s, l), *ci(b)), ("auc_null", auc(sn, ln), math.nan, math.nan),
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
            if region in ("face", "flank"):  # single-family AUC (nearest neighbour): which clue survives
                for f in FAMILIES + ["L", "M"]:
                    S = similarity(sp, "interpretable_nn", cols, families=[f])
                    s, l, _ = pairs(sp, S, np.random.default_rng(SEED))
                    fam.append({**base, "family": f, "auc": auc(s, l)})
    return pd.DataFrame(recs), pd.DataFrame(fam), boots


def farthest(dist: np.ndarray, ok: np.ndarray) -> float:
    idx = np.flatnonzero(ok)
    return float(dist[idx.max()]) if idx.size else math.nan


def dstar_table(m: pd.DataFrame, boots: dict) -> pd.DataFrame:
    """d*_AUC = farthest distance with lower CI >= 0.80 (pre-registered). Its CI: the same rule applied to
    each bootstrap draw (farthest distance whose draw-AUC >= 0.80; draws are paired across distances;
    a draw that is never usable counts as 0 m)."""
    out = []
    for (region, obs), g in m.groupby(["region", "observer"]):
        a = g[g.metric == "auc"].sort_values("distance_m")
        dist, dis = a.distance_m.to_numpy(), a.d_index.to_numpy()
        B = np.stack([boots[(region, int(i), obs)][0] for i in dis], axis=1)  # (n_boot, n_dist)
        per = np.array([farthest(dist, B[k] >= 0.80) for k in range(B.shape[0])])
        lo, hi = ci(np.where(np.isnan(per), 0.0, per))
        rec = {"region": region, "observer": obs,
               "dstar_auc_m": farthest(dist, a.ci_low.to_numpy() >= 0.80),
               "dstar_auc_ci_low_m": lo, "dstar_auc_ci_high_m": hi,
               "d_reliable_m": farthest(dist, a.ci_low.to_numpy() >= 0.95)}
        for n in (5, 20):
            bt = g[g.metric == f"bits_n{n}"].sort_values("distance_m")
            rec[f"dstar_bits_n{n}_m"] = farthest(bt.distance_m.to_numpy(), bt.ci_low.to_numpy() >= math.log2(n) - 1e-9)
        t = g[g.metric == "top1_n20"].sort_values("distance_m")
        rec["d75_n20_m"] = farthest(t.distance_m.to_numpy(), t.value.to_numpy() >= 0.75)
        out.append(rec)
    return pd.DataFrame(out)


def crossover(boots: dict, n_dist: int, dist, obs: str, n_boot: int) -> dict:
    """Paired bootstrap over zebras in both regions: dAUC(d) = face - flank, first sign change face>flank -> <=."""
    if ("face", 0, obs) not in boots or ("flank", 0, obs) not in boots:
        return {"observer": obs}
    zebras = np.intersect1d(boots[("face", 0, obs)][1], boots[("flank", 0, obs)][1])
    rng = np.random.default_rng(SEED + 4)
    draws = [rng.choice(zebras, size=len(zebras), replace=True) for _ in range(n_boot)]
    delta = np.full((n_boot, n_dist), np.nan)
    for di in range(n_dist):
        per = {}
        for region in ("face", "flank"):
            _, _, s, l, qz = boots[(region, di, obs)]
            by = {z: np.flatnonzero(qz == z) for z in zebras}
            per[region] = [auc(s[np.concatenate([by[z] for z in d])], l[np.concatenate([by[z] for z in d])]) for d in draws]
        delta[:, di] = np.array(per["face"]) - np.array(per["flank"])
    logd = np.log(np.asarray(dist))
    xs = []
    for b in range(n_boot):
        x = math.nan
        for i in range(n_dist - 1):
            if delta[b, i] > 0 >= delta[b, i + 1]:
                f = delta[b, i] / (delta[b, i] - delta[b, i + 1])
                x = math.exp(logd[i] + f * (logd[i + 1] - logd[i]))
                break
        xs.append(x)
    xs = np.array(xs)
    return {"observer": obs, "crossover_median_m": float(np.nanmedian(xs)) if np.isfinite(xs).any() else math.nan,
            "crossover_ci": ci(xs), "share_boot_with_crossing": float(np.isfinite(xs).mean()),
            "p_face_better": (delta > 0).mean(axis=0).round(3).tolist(),
            "delta_auc_median": np.nanmedian(delta, axis=0).round(3).tolist()}


def load(features: str, embeddings: list[str]) -> pd.DataFrame:
    df = pd.read_parquet(features)
    if embeddings:
        E = pd.concat([pd.read_parquet(p) for p in embeddings], ignore_index=True)
        df = df.merge(E, on=["key", "cond", "d_index"], how="left")
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("features")
    ap.add_argument("out_dir")
    ap.add_argument("--cond", default="c0")
    ap.add_argument("--embeddings", nargs="*", default=[])
    ap.add_argument("--pairs", default=None, help="HotSpotter pair table (observer 'hotspotter')")
    ap.add_argument("--n-boot", type=int, default=200)
    a = ap.parse_args()
    t0 = time.perf_counter()
    df = load(a.features, a.embeddings)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    hs = pd.read_parquet(a.pairs) if a.pairs else None
    m, fam, boots = evaluate(df, a.cond, a.n_boot, hs=hs)
    m.to_parquet(out / f"metrics_{a.cond}.parquet", index=False)
    fam.to_parquet(out / f"family_{a.cond}.parquet", index=False)
    np.savez_compressed(out / f"boot_{a.cond}.npz", **{f"{r}|{d}|{o}": v[0] for (r, d, o), v in boots.items()})
    tab = dstar_table(m, boots)
    dist = sorted(df[df.cond == a.cond].distance_m.unique())
    xo = pd.DataFrame([crossover(boots, len(dist), dist, o, a.n_boot) for o in m.observer.unique()])
    tab.to_csv(out / f"dstar_{a.cond}.csv", index=False)
    xo.to_csv(out / f"crossover_{a.cond}.csv", index=False)
    pd.set_option("display.width", 250)
    print(tab.round(1).to_string(index=False))
    print(xo[["observer", "crossover_median_m", "share_boot_with_crossing"]].to_string(index=False))
    print(f"done in {time.perf_counter() - t0:.0f} s")


if __name__ == "__main__":
    main()
