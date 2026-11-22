"""HotSpotter-style local matcher (amendment A1 in configs/metrics.yaml).

    python -m zebrasight.metrics.hotspotter percepts  <store.zarr> <out.parquet> --conds c0 [--n-jobs 6]
    python -m zebrasight.metrics.hotspotter originals <regions.parquet> <out.parquet>

SIFT keypoints inside the mask on the grey display image -> Lowe ratio 0.8 -> RANSAC affine
(threshold 3 px) -> inliers. Output is a long pair table: region, cond, d_index, qkey, gkey, inliers
(query = split "query", gallery = split "gallery", same region, same distance).

Percepts also get layout_r (amendment A4, "registered layout"): the query rho map (and mask) is warped into
the gallery frame with the RANSAC affine above (identity if < 4 inliers), then Pearson r of rho over the
overlapping mask; NaN (-> a miss) if the overlap has < 16 px or no variance.
"""

from __future__ import annotations

import argparse
import time

import cv2
import numpy as np
import pandas as pd

RATIO = 0.8
RANSAC_PX = 3.0
MAX_SIDE = 512
MIN_INLIERS_AFFINE = 4
MIN_OVERLAP_PX = 16


def grey_u8(rgb: np.ndarray) -> np.ndarray:
    g = rgb[..., :3] @ np.array([0.299, 0.587, 0.114])
    return np.clip(g * 255 + 0.5, 0, 255).astype(np.uint8)


def describe(grey: np.ndarray, mask: np.ndarray):
    if grey.shape[0] < 8 or grey.shape[1] < 8 or not mask.any():
        return np.zeros((0, 2), np.float32), None
    kp, des = cv2.SIFT_create().detectAndCompute(grey, mask.astype(np.uint8) * 255)
    return np.array([k.pt for k in kp], np.float32).reshape(-1, 2), des


def match(a, b):
    """-> (inliers, 2x3 affine query->gallery or None)."""
    (pa, da), (pb, db) = a, b
    if da is None or db is None or len(da) < 2 or len(db) < 2:
        return 0, None
    m = cv2.BFMatcher(cv2.NORM_L2).knnMatch(da, db, k=2)
    good = [x[0] for x in m if len(x) == 2 and x[0].distance < RATIO * x[1].distance]
    if len(good) < 3:
        return 0, None
    src = pa[[g.queryIdx for g in good]]
    dst = pb[[g.trainIdx for g in good]]
    M, inl = cv2.estimateAffine2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=RANSAC_PX)
    return (int(inl.sum()), M) if inl is not None else (0, None)


def inliers(a, b) -> int:
    return match(a, b)[0]


def layout_r(q, g, M, n_inl: int) -> float:
    """q, g: (rho, mask). Warp query into the gallery frame, Pearson r of rho inside the overlap."""
    (rq, mq), (rg, mg) = q, g
    if M is None or n_inl < MIN_INLIERS_AFFINE:
        M = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    size = (rg.shape[1], rg.shape[0])
    wr = cv2.warpAffine(rq.astype(np.float32), M, size, flags=cv2.INTER_LINEAR)
    wm = cv2.warpAffine(mq.astype(np.uint8), M, size, flags=cv2.INTER_NEAREST) > 0
    ov = wm & mg
    if ov.sum() < MIN_OVERLAP_PX:
        return float("nan")
    a, b = wr[ov].astype(float), rg[ov].astype(float)
    if a.std() < 1e-9 or b.std() < 1e-9:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def match_group(group: dict, meta: pd.DataFrame, tag: dict, rho: dict | None = None) -> list[dict]:
    """group: key -> (pts, des); rho (optional): key -> (rho, mask) for layout_r.
    All query x gallery pairs within one region/distance."""
    q = meta[meta.split == "query"].key
    g = meta[meta.split == "gallery"].key
    out = []
    for qk in q:
        for gk in g:
            n, M = match(group[qk], group[gk])
            rec = {**tag, "qkey": qk, "gkey": gk, "inliers": n}
            if rho is not None:
                rec["layout_r"] = layout_r(rho[qk], rho[gk], M, n)
            out.append(rec)
    return out


def _percept_job(store, keys, meta, region, cond, di):
    from zebrasight.metrics.megadesc import percept_rgb
    from zebrasight.pipeline.read_store import PerceptStore
    from zebrasight.vrpack.params import matrices

    st = PerceptStore(store)
    dm = matrices()[4]
    group, rho = {}, {}
    for k in keys:
        p = st.get(k, cond, di)
        group[k] = describe(grey_u8(percept_rgb(p, dm)), p["mask"])
        rho[k] = (p["rho"], p["mask"])
    return match_group(group, meta, {"region": region, "cond": cond, "d_index": di}, rho)


def percepts(store, out, conds, n_jobs):
    from joblib import Parallel, delayed

    from zebrasight.pipeline.read_store import PerceptStore

    st = PerceptStore(store)
    rows = st.rows()
    rows = rows[rows.region.isin(["flank", "face"])]
    if "valid" in rows:
        rows = rows[rows.valid != False]  # noqa: E712
    jobs = [(r, c, di) for c in conds for r in ("flank", "face") for di in range(len(st.distances))]
    t0 = time.perf_counter()
    res = []
    gen = Parallel(n_jobs=n_jobs, return_as="generator_unordered")(
        delayed(_percept_job)(store, rows[rows.region == r].key.tolist(), rows[rows.region == r][["key", "split"]], r, c, di)
        for r, c, di in jobs)
    for i, part in enumerate(gen, 1):
        res += part
        print(f"{i}/{len(jobs)} groups  {time.perf_counter() - t0:.0f} s", flush=True)
    pd.DataFrame(res).to_parquet(out, index=False)


def originals(regions_path, out):
    from zebrasight.data.load import load_region

    reg = pd.read_parquet(regions_path)
    reg = reg[reg.region.isin(["flank", "face"])].copy()
    reg["key"] = reg.ann_id.astype(str) + "_" + reg.region
    res = []
    for region, rr in reg.groupby("region"):
        group = {}
        for _, r in rr.iterrows():
            img, mask, _ = load_region(r)
            rgb = img[0].permute(1, 2, 0).numpy()
            m = mask[0, 0].numpy() > 0.5
            s = MAX_SIDE / max(m.shape)
            if s < 1:
                size = (round(m.shape[1] * s), round(m.shape[0] * s))
                rgb = cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)
                m = cv2.resize(m.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST) > 0
            group[r.key] = describe(grey_u8(rgb), m)
        res += match_group(group, rr[["key", "split"]], {"region": region, "cond": "orig", "d_index": -1})
    pd.DataFrame(res).to_parquet(out, index=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["originals", "percepts"])
    ap.add_argument("src")
    ap.add_argument("out")
    ap.add_argument("--conds", default="c0")
    ap.add_argument("--n-jobs", type=int, default=6)
    a = ap.parse_args()
    t0 = time.perf_counter()
    if a.mode == "originals":
        originals(a.src, a.out)
    else:
        percepts(a.src, a.out, a.conds.split(","), a.n_jobs)
    print(f"wrote {a.out} in {time.perf_counter() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
