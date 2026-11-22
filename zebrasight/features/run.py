"""Feature runner: python -m zebrasight.features.run <store> <out.parquet> [--regions flank,face] [--conds c0]
                                                [--n-jobs 4]

One row per (row key, cond, distance index): IDs, split, cond, d_index, distance_m, display_limited,
source_limited, size columns (short_side_px, mask_px, nan_reason) and every feature column
(F1-F10, L, M) plus the raw-pixel family R (Weber-normalised percept on a fixed grid, computed
at ANY size: the raw-pixel baseline does not use the min-size rule).
Rows flagged valid == False (rule-v1 faces) are skipped. Resumable: one Parquet part per
(key, cond) in <out>.parts/; existing parts are skipped; the parts are concatenated into <out>
at the end. The metrics step needs only that Parquet file, not the store.
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd

META = ("zebra_id", "image_id", "session_id", "encounter_id", "region", "split", "ann_id", "mask_method")


def raw_pixels(p: dict, grid=(16, 24)) -> dict[str, float]:
    """Raw-pixel baseline vector: y / median(y[mask]) over the mask box, outside the mask = 1,
    resampled to a fixed grid. No minimum size (works on a 1-px percept)."""
    from skimage.transform import resize

    y, m = np.asarray(p["y"], float), np.asarray(p["mask"], bool)
    names = [f"R_{i:03d}" for i in range(grid[0] * grid[1])]
    if not m.any():
        return {n: float("nan") for n in names}
    rows, cols = np.nonzero(m.any(1))[0], np.nonzero(m.any(0))[0]
    r0, r1, c0, c1 = rows[0], rows[-1] + 1, cols[0], cols[-1] + 1
    med = float(np.median(y[m])) or 1.0
    sub = np.where(m[r0:r1, c0:c1], y[r0:r1, c0:c1] / med, 1.0)
    g = resize(sub, grid, order=1, anti_aliasing=sub.shape[0] > grid[0] and sub.shape[1] > grid[1], mode="edge")
    return dict(zip(names, g.ravel().astype(float)))


def process_key(store_path: str, key: str, attrs: dict, cond: str, out_dir: str) -> int:
    from zebrasight.features import extract
    from zebrasight.features.common import prepare
    from zebrasight.pipeline.read_store import PerceptStore

    st = PerceptStore(store_path)
    recs = []
    for di, d in enumerate(st.distances):
        p = st.get(key, cond, di)
        p["region"] = attrs.get("region")  # selects the region's illumination sigma (D1)
        m = p["mask"]
        rows, cols = np.nonzero(m.any(1))[0], np.nonzero(m.any(0))[0]
        short = int(min(rows[-1] + 1 - rows[0], cols[-1] + 1 - cols[0])) if rows.size else 0
        ctx = prepare(p, d)
        feats = {}
        from zebrasight.features.registry import FEATURES

        for name, fn in FEATURES.items():
            feats.update(fn(p, d, ctx))
        recs.append({"key": key, **{k: attrs.get(k) for k in META}, "cond": cond, "d_index": di, "distance_m": d,
                     "display_limited": p["display_limited"], "source_limited": p["source_limited"],
                     "deg_per_px": p["deg_per_px"], "short_side_px": short, "mask_px": int(m.sum()),
                     "nan_reason": ctx.reason, **feats, **raw_pixels(p)})
    df = pd.DataFrame(recs)
    float_cols = df.select_dtypes("float64").columns
    df[float_cols] = df[float_cols].astype("float32")
    df.to_parquet(Path(out_dir) / f"{key}__{cond}.parquet", index=False)
    return len(df)


def main() -> None:
    from joblib import Parallel, delayed

    from zebrasight.pipeline.read_store import PerceptStore

    ap = argparse.ArgumentParser()
    ap.add_argument("store")
    ap.add_argument("out")
    ap.add_argument("--regions", default="flank,face")
    ap.add_argument("--conds", default="c0")
    ap.add_argument("--n-jobs", type=int, default=4)
    a = ap.parse_args()
    out = Path(a.out)
    parts = out.with_suffix(".parts")
    parts.mkdir(parents=True, exist_ok=True)
    st = PerceptStore(a.store)
    rows = st.rows()
    rows = rows[rows.region.isin(a.regions.split(","))]
    if "valid" in rows:
        skipped = int((rows.valid == False).sum())  # noqa: E712
        rows = rows[rows.valid != False]  # noqa: E712
        print(f"skipping {skipped} rows with valid == False", flush=True)
    jobs = []
    for cond in a.conds.split(","):
        for r in rows.itertuples():
            if st.resolve(cond) not in r.conds:
                continue
            if (parts / f"{r.key}__{cond}.parquet").exists():
                continue
            jobs.append((r.key, {k: getattr(r, k, None) for k in META}, cond))
    print(f"{len(jobs)} (key, cond) jobs to run, n_jobs={a.n_jobs}", flush=True)
    t0 = time.perf_counter()
    done = 0
    gen = Parallel(n_jobs=a.n_jobs, return_as="generator_unordered")(
        delayed(process_key)(str(a.store), k, at, c, str(parts)) for k, at, c in jobs)
    for _ in gen:
        done += 1
        if done % 20 == 0 or done == len(jobs):
            el = time.perf_counter() - t0
            print(f"{done:5d}/{len(jobs)}  {el:6.0f} s  ~{el / done * (len(jobs) - done):5.0f} s left", flush=True)
    files = sorted(parts.glob("*.parquet"))
    conds = set(a.conds.split(","))
    df = pd.concat([pd.read_parquet(f) for f in files if f.stem.split("__")[-1] in conds], ignore_index=True)
    df.to_parquet(out, index=False)
    print(f"wrote {out}: {len(df)} rows x {df.shape[1]} cols in {time.perf_counter() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
