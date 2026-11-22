"""Loading for the positional study: region rows, original crops, per-photo frames, small percept sets.

Region rows come from data/regions_50_positional.parquet (A1). A row's key is "<ann_id>_<region>".
Frames are fitted on each photo's full Mask R-CNN body (v2 label PNG > 0), even when the photo is
excluded from FULL-BODY analysis (cut or overlapped bodies still give a usable axis for flank/rump).
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from zebrasight.config import REPO_ROOT, load_config, make_cond
from zebrasight.data.load import load_region
from zebrasight.positional.frame import Frame, fit_frame, head_check


def cfg():
    return load_config("positional")


def out_dir() -> Path:
    p = REPO_ROOT / cfg().data.out_dir
    p.mkdir(parents=True, exist_ok=True)
    return p


def regions() -> pd.DataFrame:
    r = pd.read_parquet(REPO_ROOT / cfg().data.out_regions)
    r["key"] = r.ann_id.astype(int).astype(str) + "_" + r.region
    return r


def crop_hw(row) -> tuple[int, int]:
    x0, y0, x1, y1 = (int(v) for v in row["bbox_xyxy"])
    return y1 - y0, x1 - x0


def load_crop(row) -> tuple[np.ndarray, np.ndarray]:
    """-> (rgb (h, w, 3) sRGB [0, 1], mask (h, w) bool) for one region row."""
    img, mask, _ = load_region(row)
    return img[0].permute(1, 2, 0).numpy(), mask[0, 0].numpy() > 0.5


def frames(reg: pd.DataFrame, cache: bool = True) -> dict[int, Frame]:
    """Frame per photo (ann_id), cached in outputs/positional/frames.parquet (with the head check)."""
    path = out_dir() / "frames.parquet"
    if cache and path.exists():
        t = pd.read_parquet(path)
        if set(reg.ann_id.astype(int)) <= set(t.ann_id):
            return {int(r.ann_id): Frame.from_dict(r._asdict()) for r in t.itertuples(index=False)}
    c = cfg()
    heads = pd.read_parquet(REPO_ROOT / "data" / "interim" / "masks_v2_index.parquet").set_index("ann_id")
    rows, out = [], {}
    for r in reg.drop_duplicates("ann_id").itertuples(index=False):
        lab = np.asarray(Image.open(REPO_ROOT / r.mask_path))  # v2 label PNG or body PNG: > 0 = whole body either way
        x0, y0 = (float(v) for v in r.mask_bbox_xyxy[:2])
        fr = fit_frame(lab > 0, (x0, y0), str(r.viewpoint) in set(c.frame.head_left_viewpoints))
        hb = heads.loc[int(r.ann_id), "head_box"] if int(r.ann_id) in heads.index else None
        hb_photo = None if hb is None or len(hb) != 4 else np.asarray(hb, float) + np.array([x0, y0, x0, y0])
        out[int(r.ann_id)] = fr
        rows.append({"ann_id": int(r.ann_id), **fr.to_dict(), "head_check": head_check(fr, hb_photo)})
    pd.DataFrame(rows).to_parquet(path, index=False)
    return out


def smoke_zebras(reg: pd.DataFrame) -> list[str]:
    """The fixed 5 smoke-test zebras: full body usable on both dates (seeded choice)."""
    c = cfg().smoke
    body = reg[reg.region == "body"]
    both = body.groupby("zebra_id").split.nunique()
    pool = np.array(sorted(both[both == 2].index))
    return sorted(np.random.default_rng(int(c.seed)).choice(pool, size=int(c.n_zebras), replace=False).tolist())


def make_percepts(rows: pd.DataFrame, distance_indices, cond: str = "c0", path: Path | None = None) -> dict:
    """Percepts for a FEW rows at a few distances via the unchanged pipeline (geometry -> scene ->
    atmosphere -> eye). -> {(key, d_index): percept dict like PerceptStore.get}. Cached as .npz."""
    import torch

    from zebrasight.pipeline.pipeline import Pipeline

    grid = load_config("grid")
    dist = list(grid.distances_m["values"])
    if path is not None and path.exists():
        z = np.load(path, allow_pickle=True)
        return z["percepts"].item()
    pipe = Pipeline(list(grid.all_stages), eye_cfg=load_config("eye"))
    out = {}
    t0 = time.perf_counter()
    for k, r in enumerate(rows.to_dict("records"), 1):
        img, mask, meta = load_region(r)
        ds = [dist[i] for i in distance_indices]
        ps = pipe.run_distances(img, float(r["cm_per_px"]), mask, make_cond(cond, ds[0]), ds, meta=meta)
        for i, p in zip(distance_indices, ps):
            out[(r["key"], i)] = {
                "y": p.y[0, 0].float().cpu().numpy(), "s": None if p.s is None else p.s[0, 0].float().cpu().numpy(),
                "mask": p.mask[0, 0].cpu().numpy() > 0.5, "deg_per_px": float(p.deg_per_px),
                "distance_m": float(dist[i]), "region": r["region"], "cond": cond}
        if k % 10 == 0 or k == len(rows):
            print(f"  percepts {k}/{len(rows)} rows  {time.perf_counter() - t0:.0f} s", flush=True)
        del img, mask
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if path is not None:
        np.savez_compressed(path, percepts=np.array(out, dtype=object))
    return out
