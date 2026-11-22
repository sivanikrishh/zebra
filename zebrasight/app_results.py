"""Real numbers for app_data/results.json (schema: app_data/README.md). Used by app_export (not --dummy)."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from zebrasight.app_export import DISTANCES, FAMILIES, OBSERVERS, PHOTO_LIMITED_MAX_M, _r
from zebrasight.config import REPO_ROOT

MET = REPO_ROOT / "outputs" / "metrics"
FEAT = REPO_ROOT / "outputs" / "features"
CONDS = ("c0", "c8", "c9")
MIN_SHORT_SIDE_PX = 12
# same-day pairs (same zebra, same session, different photo), gallery only, c0 at 2 m: Checkpoint-2 diagnostic
SAME_DAY = [("flank", "interpretable_nn", 0.755), ("flank", "spatial_nn", 0.680), ("flank", "raw_pixel", 0.684),
            ("face", "interpretable_nn", 0.657), ("face", "spatial_nn", 0.625), ("face", "raw_pixel", 0.614)]


def _size_share() -> pd.DataFrame:
    parts = []
    for f in ("features_c0_v2.parquet", "features_c8c9_v2.parquet"):
        p = FEAT / f
        if p.exists():
            d = pd.read_parquet(p, columns=["region", "cond", "d_index", "short_side_px"])
            parts.append(d.assign(small=d.short_side_px < MIN_SHORT_SIDE_PX)
                         .groupby(["region", "cond", "d_index"]).small.mean().reset_index())
    return pd.concat(parts) if parts else pd.DataFrame(columns=["region", "cond", "d_index", "small"])


def _per_draw_dstar(boot: dict, region: str, obs: str) -> np.ndarray | None:
    """Per-bootstrap-draw d* (draw AUC >= 0.80 farthest; 0 if never). Draws are the same zebra resamples
    for every distance and condition (fixed seed, same zebras), so ratios between conditions are paired."""
    cols = [boot.get(f"{region}|{i}|{obs}") for i in range(len(DISTANCES))]
    if any(c is None for c in cols):
        return None
    B = np.stack(cols, axis=1) >= 0.80
    d = np.asarray(DISTANCES)
    return np.array([d[np.flatnonzero(b).max()] if b.any() else 0.0 for b in B])


def _ci(a):
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    return [_r(np.percentile(a, 2.5)), _r(np.percentile(a, 97.5))] if a.size else [None, None]


def real() -> dict:
    share = _size_share()
    obs_ids = [o["id"] for o in OBSERVERS]
    curves, dstar, fam_out, loss = [], [], [], []
    boots = {}
    for cond in CONDS:
        mp = MET / f"metrics_{cond}.parquet"
        if not mp.exists():
            continue
        m = pd.read_parquet(mp)
        bp = MET / f"boot_{cond}.npz"
        boots[cond] = dict(np.load(bp)) if bp.exists() else {}
        for region in ("face", "flank"):
            for oid in obs_ids:
                g = m[(m.region == region) & (m.observer == oid)]
                if g.empty:
                    continue
                pts = []
                for di, d in enumerate(DISTANCES):
                    r = g[g.d_index == di].set_index("metric")
                    if r.empty:
                        continue

                    def v(k, col="value"):
                        return _r(r.at[k, col]) if k in r.index else None

                    sh = share[(share.region == region) & (share.cond == cond) & (share.d_index == di)].small
                    lo = v("auc", "ci_low")
                    pts.append({"d_index": di, "distance_m": d, "auc": v("auc"), "auc_ci": [lo, v("auc", "ci_high")],
                                "top1_n5": v("top1_n5"), "top1_n5_ci": [v("top1_n5", "ci_low"), v("top1_n5", "ci_high")],
                                "chance_n5": v("chance_n5"),
                                "top1_n20": v("top1_n20"), "top1_n20_ci": [v("top1_n20", "ci_low"), v("top1_n20", "ci_high")],
                                "chance_n20": v("chance_n20"),
                                "usable": bool(lo is not None and lo >= 0.80), "photo_limited": d <= PHOTO_LIMITED_MAX_M,
                                "size_rule_share": _r(sh.iloc[0]) if len(sh) else None})
                curves.append({"region": region, "condition": cond, "observer": oid, "points": pts})
        tab = pd.read_csv(MET / f"dstar_{cond}.csv")
        for _, r in tab[tab.region.isin(["face", "flank"])].iterrows():
            ds = r.dstar_auc_m if math.isfinite(r.dstar_auc_m) else None
            dstar.append({"region": r.region, "condition": cond, "observer": r.observer, "dstar_auc_m": _r(ds),
                          "dstar_auc_ci_m": [_r(r.dstar_auc_ci_low_m), _r(r.dstar_auc_ci_high_m)],
                          "never_usable": ds is None})
        fp = MET / f"family_{cond}.parquet"
        if fp.exists():
            fam = pd.read_parquet(fp)
            for (region, d), g in fam.groupby(["region", "distance_m"]):
                ranked = sorted(({"family": f, "label": FAMILIES[f], "auc": _r(a)} for f, a in zip(g.family, g.auc)),
                                key=lambda x: -(x["auc"] if x["auc"] is not None else 0))
                fam_out.append({"region": region, "condition": cond, "distance_m": _r(d), "families": ranked})
    # distance loss ratio = d*(cond) / d*(c0), paired over bootstrap draws (pre-registered: with CI)
    for region in ("face", "flank"):
        for oid in ("megadesc", "hotspotter"):
            base = _per_draw_dstar(boots.get("c0", {}), region, oid)
            for cond in ("c8", "c9"):
                other = _per_draw_dstar(boots.get(cond, {}), region, oid)
                if base is None or other is None:
                    continue
                with np.errstate(divide="ignore", invalid="ignore"):
                    ratio = np.where(base > 0, other / base, np.nan)
                ok = np.isfinite(ratio)
                loss.append({"region": region, "observer": oid, "condition": cond, "vs": "c0",
                             "ratio": _r(np.median(ratio[ok])) if ok.any() else None, "ratio_ci": _ci(ratio),
                             "draws_with_usable_day": _r(ok.mean())})
    orig = pd.read_csv(MET / "originals.csv") if (MET / "originals.csv").exists() else pd.DataFrame()
    originals = [{"region": r.region, "observer": r.observer, "auc": _r(r.auc), "auc_ci": [_r(r.auc_ci_low), _r(r.auc_ci_high)],
                  "top1_n20": _r(r.top1_n20), "chance_n20": _r(r.chance_n20)}
                 for r in orig.itertuples() if r.region in ("face", "flank")]
    at2 = [{"region": c["region"], "observer": c["observer"], "auc": c["points"][0]["auc"],
            "auc_ci": c["points"][0]["auc_ci"]} for c in curves if c["condition"] == "c0" and c["points"]]
    return {
        "dstar": dstar, "curves": curves, "leading_features": fam_out, "distance_loss_ratio": loss,
        "original_crops": originals,
        "statistics_vs_positions": {
            "headline": "Summary stripe statistics (F1-F10 scalars, coarse layout) do not identify individuals across "
                        "days even at 2 m; identity is positional.",
            "at_2m_c0": at2,
            "same_day_pairs_auc": [{"region": r, "observer": o, "auc": a} for r, o, a in SAME_DAY]},
    }
