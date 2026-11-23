"""ZebraSight prototype: Recognition horizon + Stripe identity explorer (internal demo; GZGC-derived images,
CC BY-NC-ND).

    python scripts/build_app_demo.py [--n-jobs 4]     -> outputs/app_demo/ (open index.html in a browser)

Recognition horizon: the stored zebra-eye textures (app_data/textures) and the study's measured identification
accuracy (top-1 in a herd of 20, 95% CI; app_data/results.json). Copied, not computed.

Stripe identity explorer (explorer.js): two flank photos side by side (a later-day and an earliest-day photo, so
every pair is a study pair), from a pool of the 5 demo zebras + 3 seeded decoy zebras each. Everything is
existing study data or the study's own code:
  original photos   prepared exactly as the study's original-photo HotSpotter run (hotspotter.originals)
  zebra-eye images  stored percepts (cache/percepts_50.zarr), display colour as the HotSpotter/MegaDescriptor runs
  scores            official tables: MegaDescriptor embeddings (cosine), HotSpotter inliers; HotSpotter match
                    points recomputed with the study's rules and CHECKED equal to the official inlier counts
  stripe map        features.common.prepare (the study's lighting-corrected binary, skeleton, junctions, endings);
                    junction density CHECKED equal to the stored F5 values
  characteristics   stored F1-F10 values (outputs/features/features_*_v2.parquet)
  AUC per observer  app_data/results.json
"""

from __future__ import annotations

import argparse
import json
import shutil

import cv2
import numpy as np
import pandas as pd
from PIL import Image

from zebrasight.config import REPO_ROOT
from zebrasight.metrics.hotspotter import describe, grey_u8

APP = REPO_ROOT / "app_data"
OUT = REPO_ROOT / "outputs" / "app_demo"
STORE = REPO_ROOT / "cache" / "percepts_50.zarr"
REGIONS = REPO_ROOT / "data" / "regions_50.parquet"
FEAT = REPO_ROOT / "outputs" / "features"
DEMO_ZEBRAS = ["IBEIS_PZ_0334", "IBEIS_PZ_0395", "IBEIS_PZ_1267", "IBEIS_PZ_1648", "IBEIS_PZ_1725"]
CONDS = ["c0", "c8", "c9"]
OBS = ("megadesc", "hotspotter")
N_DECOYS, SEED = 3, 0


# ---- recognition horizon ----------------------------------------------------------------------------------

def horizon_data(results: dict) -> dict:
    tex = json.loads((APP / "textures" / "textures.json").read_text(encoding="utf-8"))
    textures = {}
    for s in tex["sets"]:
        if s["condition"] not in CONDS:
            continue
        files = []
        for L in s["layers"]:
            dst = OUT / "tex" / L["file"]
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(APP / "textures" / L["file"], dst)
            files.append("tex/" + L["file"])
        textures.setdefault(s["zebra_id"], {}).setdefault(s["region"], {})[s["condition"]] = files
    return {
        "status": results["status"], "distances_m": results["distances_m"],
        "conditions": [c for c in results["conditions"] if c["id"] in CONDS], "regions": results["regions"],
        "observers": [o for o in results["observers"] if o["id"] in OBS],
        "curves": [{"region": c["region"], "condition": c["condition"], "observer": c["observer"],
                    "points": [{k: p[k] for k in ("distance_m", "top1_n20", "top1_n20_ci", "chance_n20")} for p in c["points"]]}
                   for c in results["curves"] if c["observer"] in OBS and c["condition"] in CONDS],
        "zebras": sorted(textures), "textures": textures,
    }


# ---- stripe identity explorer -----------------------------------------------------------------------------

def tables():
    E = pd.concat([pd.read_parquet(FEAT / "megadesc_c0.parquet"), pd.read_parquet(FEAT / "megadesc_c8c9.parquet")])
    E = E[E.key.str.endswith("_flank")]
    H = pd.read_parquet(FEAT / "hotspotter_percepts.parquet")
    return E, H[H.region == "flank"]


def score_matrices(E, H, cond, d, qk, gk) -> dict:
    """(n_query x n_gallery) official scores per observer; NaN = no score (invalid percept)."""
    ec = [c for c in E.columns if c.startswith("E_")]
    e = E[(E.cond == cond) & (E.d_index == d)].set_index("key")
    Sm = e.loc[qk, ec].to_numpy(float) @ e.loc[gk, ec].to_numpy(float).T
    h = H[(H.cond == cond) & (H.d_index == d)]
    lut = dict(zip(zip(h.qkey, h.gkey), h.inliers))
    return {"megadesc": Sm, "hotspotter": np.array([[lut.get((q, g), 0) for g in gk] for q in qk], float)}


def percept_png(st, dm, key, cond, d) -> np.ndarray:
    from zebrasight.metrics.megadesc import percept_rgb

    p = st.get(key, cond, d)
    return (np.clip(percept_rgb(p, dm), 0, 1) * 255).astype(np.uint8) * p["mask"][..., None].astype(np.uint8)


def stripe_map(p: dict, distance_m: float) -> dict:
    """The study's own stripe representation (features.common.prepare): palette image 0 = outside / white
    stripe, 1 = black stripe, 2 = skeleton; junction and ending points; their densities (F5, F6)."""
    from zebrasight.features.common import prepare

    c = prepare({**p, "region": "flank"}, distance_m)
    if not c.valid:
        return {"valid": False, "reason": c.reason}
    lab = np.zeros(c.mask.shape, np.uint8)
    lab[c.black] = 1
    lab[c.skel] = 2
    return {"valid": True, "lab": lab, "junctions": [[int(r), int(q)] for r, q in c.junctions],
            "endings": [[int(r), int(q)] for r, q in c.endpoints],
            "F5": 100.0 * len(c.junctions) / c.area_cm2, "F6": 100.0 * len(c.endpoints) / c.area_cm2}


def explorer_pairs(feats, pool_q, pool_g, S, qk, gk) -> dict:
    """Every pool (later-day, earliest-day) pair: official scores + HotSpotter inlier points for the lines."""
    from zebrasight.metrics.hotspotter import RANSAC_PX, RATIO

    qi, gi = {k: i for i, k in enumerate(qk)}, {k: i for i, k in enumerate(gk)}
    out = {}
    for q in pool_q:
        for g in pool_g:
            (pa, da), (pb, db) = feats[q], feats[g]
            qp = gp = np.zeros((0, 2))
            if da is not None and db is not None and len(da) >= 2 and len(db) >= 2:
                m = cv2.BFMatcher(cv2.NORM_L2).knnMatch(da, db, k=2)
                good = [x[0] for x in m if len(x) == 2 and x[0].distance < RATIO * x[1].distance]
                if len(good) >= 3:
                    src, dst = pa[[x.queryIdx for x in good]], pb[[x.trainIdx for x in good]]
                    _, inl = cv2.estimateAffine2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=RANSAC_PX)
                    if inl is not None:
                        k_ = inl.ravel().astype(bool)
                        qp, gp = src[k_], dst[k_]
            md = S["megadesc"][qi[q], gi[g]]
            out[f"{q}|{g}"] = {"md": None if np.isnan(md) else round(float(md), 4),
                               "hs": int(S["hotspotter"][qi[q], gi[g]]),
                               "q_pts": np.round(qp).astype(int).tolist(), "g_pts": np.round(gp).astype(int).tolist()}
    return out


def hist(S, qz, gz) -> dict:
    """Score distributions of ALL study pairs at one setting, same vs different zebra (context only)."""
    same = qz[:, None] == gz[None, :]
    out = {}
    for o, M in S.items():
        ok = np.isfinite(M)
        v = M[ok]
        if not v.size:
            out[o] = None
            continue
        edges = (np.linspace(np.floor(v.min() * 20) / 20, np.ceil(v.max() * 20) / 20, 31) if o == "megadesc"
                 else np.arange(0, min(int(v.max()), 60) + 2) - 0.5)
        hs_, _ = np.histogram(np.clip(M[same & ok], edges[0], edges[-1]), edges)
        hd, _ = np.histogram(np.clip(M[~same & ok], edges[0], edges[-1]), edges)
        out[o] = {"edges": np.round(edges, 4).tolist(), "same": hs_.tolist(), "diff": hd.tolist()}
    return out


def setting_job(cond, d, pool, qk, gk, qz, gz, E, H, pool_q, pool_g):
    from zebrasight.metrics.megadesc import percept_rgb
    from zebrasight.pipeline.read_store import PerceptStore
    from zebrasight.vrpack.params import matrices

    st, dm = PerceptStore(STORE), matrices()[4]
    imgs = {k: percept_png(st, dm, k, cond, d) for k in pool}
    S = score_matrices(E, H, cond, d, qk, gk)
    feats, maps = {}, {}
    for k in pool:
        p = st.get(k, cond, d)
        feats[k] = describe(grey_u8(percept_rgb(p, dm)), p["mask"])
        maps[k] = stripe_map(p, st.distances[d])
    return cond, d, imgs, explorer_pairs(feats, pool_q, pool_g, S, qk, gk), maps, hist(S, qz, gz)


def originals_job(reg, pool, pool_q, pool_g, qk, gk, qz, gz):
    """Original photos prepared exactly as hotspotter.originals (crop, longest side <= MAX_SIDE, grey SIFT in
    the mask). Scores from the official original-photo tables."""
    from zebrasight.data.load import load_region
    from zebrasight.metrics.hotspotter import MAX_SIDE

    rows = reg.set_index("key")
    imgs, feats = {}, {}
    for k in pool:
        img, mask, _ = load_region(rows.loc[k].to_dict())
        rgb = img[0].permute(1, 2, 0).numpy()
        m = mask[0, 0].numpy() > 0.5
        s = MAX_SIDE / max(m.shape)
        if s < 1:
            size = (round(m.shape[1] * s), round(m.shape[0] * s))
            rgb = cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)
            m = cv2.resize(m.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST) > 0
        feats[k] = describe(grey_u8(rgb), m)
        # load_region gives the photo's own sRGB values / 255; written as-is, like app_export.human_views
        imgs[k] = np.dstack([(np.clip(rgb, 0, 1) * 255).astype(np.uint8), (m * 255).astype(np.uint8)])
    Eo = pd.read_parquet(FEAT / "megadesc_orig.parquet").set_index("key")
    Ho = pd.read_parquet(FEAT / "hotspotter_orig.parquet")
    Ho = Ho[Ho.region == "flank"]
    ec = [c for c in Eo.columns if c.startswith("E_")]
    lut = dict(zip(zip(Ho.qkey, Ho.gkey), Ho.inliers))
    S = {"megadesc": Eo.loc[qk, ec].to_numpy(float) @ Eo.loc[gk, ec].to_numpy(float).T,
         "hotspotter": np.array([[lut.get((q, g), 0) for g in gk] for q in qk], float)}
    return imgs, explorer_pairs(feats, pool_q, pool_g, S, qk, gk), hist(S, qz, gz)


def feature_table(keys) -> dict:
    """Stored F1-F10 values for 4 interpretable characteristics, all flank photos of the 50 zebras."""
    cols = {"F1_width_cm_median": "width", "F7_period_cm": "spacing", "F5_junctions_per100cm2": "junctions",
            "F2_rho_mean": "black"}
    F = pd.concat([pd.read_parquet(FEAT / "features_c0_v2.parquet", columns=["key", "cond", "d_index", *cols]),
                   pd.read_parquet(FEAT / "features_c8c9_v2.parquet", columns=["key", "cond", "d_index", *cols])])
    F = F[F.key.isin(keys)]
    out = {}
    for (c, d), g in F.groupby(["cond", "d_index"]):
        g = g.set_index("key").reindex(keys)
        out.setdefault(c, [None] * 16)[int(d)] = {
            v: [None if not np.isfinite(x) else round(float(x), 4) for x in g[k].to_numpy(float)] for k, v in cols.items()}
    return out


def explorer_data(results: dict, n_jobs: int) -> dict:
    from joblib import Parallel, delayed

    reg = pd.read_parquet(REGIONS)
    reg = reg[reg.region == "flank"].copy()
    reg["key"] = reg.ann_id.astype(int).astype(str) + "_flank"
    reg = reg.sort_values("ann_id")
    gal, qry = reg[reg.split == "gallery"], reg[reg.split == "query"]
    gk, qk = gal.key.tolist(), qry.key.tolist()
    gz, qz = gal.zebra_id.to_numpy(), qry.zebra_id.to_numpy()
    # pool: the demo zebras + 3 seeded decoy zebras each (same seed as before), ALL their flank photos
    rng, others = np.random.default_rng(SEED), sorted(set(gz))
    pool_z = set(DEMO_ZEBRAS)
    for z in DEMO_ZEBRAS:
        pool_z |= {str(x) for x in rng.choice([o for o in others if o != z], size=N_DECOYS, replace=False)}
        for _ in range(N_DECOYS):
            rng.random()                                   # keep the original seeded sequence (one draw per round)
    pool_q = qry[qry.zebra_id.isin(pool_z)].key.tolist()
    pool_g = gal[gal.zebra_id.isin(pool_z)].key.tolist()
    pool = sorted(set(pool_q) | set(pool_g))
    E, H = tables()
    dists = results["distances_m"]
    res = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(setting_job)(c, d, pool, qk, gk, qz, gz, E, H, pool_q, pool_g) for c, d in [("c0", 0)])   # explorer shows 2 m, clear day
    o_imgs, o_pairs, o_hist = originals_job(reg, pool, pool_q, pool_g, qk, gk, qz, gz)

    (OUT / "img").mkdir(exist_ok=True)
    (OUT / "map").mkdir(exist_ok=True)
    for k, im in o_imgs.items():
        Image.fromarray(im, "RGBA").save(OUT / "img" / f"{k}_orig.png")
    pairs = {"orig": o_pairs, **{c: [None] * len(dists) for c in CONDS}}
    hists = {"orig": o_hist, **{c: [None] * len(dists) for c in CONDS}}
    maps_out = {c: [None] * len(dists) for c in CONDS}
    palette = [0, 0, 0, 230, 80, 40, 255, 230, 60] + [0] * (256 * 3 - 9)
    F5 = pd.concat([pd.read_parquet(FEAT / f, columns=["key", "cond", "d_index", "F5_junctions_per100cm2"])
                    for f in ("features_c0_v2.parquet", "features_c8c9_v2.parquet")])
    F5 = F5.set_index(["key", "cond", "d_index"]).F5_junctions_per100cm2
    hs_mismatch, f5_check = sum(len(v["q_pts"]) != v["hs"] for v in o_pairs.values()), []
    for cond, d, imgs, ex, maps, hst in res:
        for k, im in imgs.items():
            Image.fromarray(im).save(OUT / "img" / f"{k}_{cond}_d{d:02d}.png")
        pairs[cond][d], hists[cond][d] = ex, hst
        hs_mismatch += sum(len(v["q_pts"]) != v["hs"] for v in ex.values())
        mm = {}
        for k, m in maps.items():
            if m["valid"]:
                im = Image.fromarray(m.pop("lab"), "P")
                im.putpalette(palette)
                im.save(OUT / "map" / f"{k}_{cond}_d{d:02d}.png", transparency=0)
                ref = F5.get((k, cond, d), np.nan)
                if np.isfinite(ref):
                    f5_check.append(abs(m["F5"] - ref) / max(abs(ref), 1e-9))
                m["F5"], m["F6"] = round(m["F5"], 3), round(m["F6"], 3)
            mm[k] = m
        maps_out[cond][d] = mm
    print(f"CHECK HotSpotter: recomputed inlier counts differing from the official tables: {hs_mismatch}")
    print(f"CHECK stripe map: junction density vs stored F5, max rel. diff {max(f5_check):.4f} over {len(f5_check)} percepts")
    return {
        "distances_m": dists, "conditions": [c for c in results["conditions"] if c["id"] in CONDS],
        "thresholds": results["thresholds"], "observers_all": results["observers"],
        "curves": [c for c in results["curves"] if c["region"] == "flank" and c["condition"] in CONDS],
        "original_crops": results.get("original_crops", []), "stats_vs_pos": results["statistics_vs_positions"],
        "photos": [{"key": r.key, "zebra": r.zebra_id, "split": r.split, "date": str(r.date)[:10]}
                   for r in reg[reg.key.isin(pool)].itertuples(index=False)],
        "pool_q": pool_q, "pool_g": pool_g, "pairs": pairs, "hist": hists, "maps": maps_out,
        "feat_keys": reg.key.tolist(), "feat_zebra": reg.zebra_id.tolist(), "features": feature_table(reg.key.tolist()),
        "demo_zebra": DEMO_ZEBRAS[0], "hs_mismatch": int(hs_mismatch),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n-jobs", type=int, default=4)
    a = ap.parse_args()
    results = json.loads((APP / "results.json").read_text(encoding="utf-8"))
    OUT.mkdir(parents=True, exist_ok=True)
    for child in OUT.iterdir():                              # clear contents (the folder may be open)
        shutil.rmtree(child) if child.is_dir() else child.unlink()
    (OUT / "data.js").write_text("window.DEMO = " + json.dumps(horizon_data(results)) + ";\n", encoding="utf-8")
    (OUT / "explorer.js").write_text("window.EX = " + json.dumps(explorer_data(results, a.n_jobs)) + ";\n", encoding="utf-8")
    import sys

    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    from build_pattern_lab import build as build_lab   # Pattern Lab (science in zebrasight/patternlab.py)

    build_lab(OUT)
    shutil.copy(REPO_ROOT / "scripts" / "app_demo.html", OUT / "index.html")
    print("open", OUT / "index.html")


if __name__ == "__main__":
    main()
