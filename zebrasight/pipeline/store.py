"""Run percepts for region rows and write them to a Zarr store.

    python -m zebrasight.pipeline.store <regions.parquet> <out.zarr> [--regions flank,face] [--low-priority]

Layout (zarr v3, blosc-zstd clevel 5 bit-shuffle, one shard file per array):
    /                    attrs: conds, distances_m, eye config snapshot, regions file, created
    /rows/<ann_id>_<region>/
        y, s, rho        float16 (4 conds, 16 distances, Hmax, Wmax), zero-padded; s empty at night
        mask             uint8   (4, 16, Hmax, Wmax)
        shape            int32   (4, 16, 2)  stored (h, w) of each percept
        deg_per_px       float32 (4, 16)
        display_limited, source_limited   bool (4, 16)
        attrs: the region row's IDs and scale, complete=True when written
Each percept's valid area is [:h, :w] of its (cond, distance) slice. Rows already marked
complete are skipped, so an interrupted run resumes. Everything is regenerable from the
regions parquet + configs (noise seeds are hashes of the IDs).
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import zarr
from omegaconf import OmegaConf
from zarr.codecs import BloscCodec

from zebrasight.config import REPO_ROOT, load_config, make_cond
from zebrasight.data.load import load_region
from zebrasight.pipeline.pipeline import Pipeline

CONDS = ("c0", "c7", "c8", "c9")
CODEC = BloscCodec(cname="zstd", clevel=5, shuffle="bitshuffle")


def lower_priority() -> None:
    """Below-normal CPU priority so a concurrent job keeps the machine (Windows and Linux)."""
    if os.name == "nt":
        import ctypes

        BELOW_NORMAL = 0x00004000
        ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), BELOW_NORMAL)
    else:
        os.nice(10)


def row_key(row) -> str:
    return f"{row['ann_id']}_{row['region']}"


def percepts_for_row(pipe: Pipeline, row, distances) -> dict:
    img, mask, meta = load_region(row)
    out = {}
    for ci, c in enumerate(CONDS):
        for di, d in enumerate(distances):
            p = pipe.run(img, float(row["cm_per_px"]), mask, make_cond(c, d), meta=meta)
            out[(ci, di)] = p
    return out


def write_row(group: zarr.Group, key: str, row, percepts: dict, n_d: int) -> int:
    hmax = max(p.y.shape[-2] for p in percepts.values())
    wmax = max(p.y.shape[-1] for p in percepts.values())
    full = (len(CONDS), n_d, hmax, wmax)
    arrays = {k: np.zeros(full, np.float16) for k in ("y", "s", "rho")}
    arrays["mask"] = np.zeros(full, np.uint8)
    shape = np.zeros((len(CONDS), n_d, 2), np.int32)
    dpp = np.zeros((len(CONDS), n_d), np.float32)
    disp = np.zeros((len(CONDS), n_d), bool)
    src = np.zeros((len(CONDS), n_d), bool)
    for (ci, di), p in percepts.items():
        h, w = p.y.shape[-2:]
        arrays["y"][ci, di, :h, :w] = p.y[0, 0].cpu().numpy()
        arrays["rho"][ci, di, :h, :w] = p.rho[0, 0].cpu().numpy()
        arrays["mask"][ci, di, :h, :w] = p.mask[0, 0].cpu().numpy() > 0.5
        if p.s is not None:
            arrays["s"][ci, di, :h, :w] = p.s[0, 0].cpu().numpy()
        shape[ci, di] = (h, w)
        dpp[ci, di] = p.deg_per_px
        disp[ci, di] = p.display_limited
        src[ci, di] = p.source_limited
    g = group.require_group(key)
    for name, a in arrays.items():
        arr = g.create_array(name, shape=a.shape, chunks=(1, 1, hmax, wmax), shards=a.shape, dtype=a.dtype,
                             fill_value=0, compressors=CODEC, overwrite=True)
        arr[:] = a
    for name, a in (("shape", shape), ("deg_per_px", dpp), ("display_limited", disp), ("source_limited", src)):
        g.create_array(name, shape=a.shape, dtype=a.dtype, overwrite=True)[:] = a
    attrs = {k: (v.item() if hasattr(v, "item") else v) for k, v in row.items()
             if k in ("zebra_id", "image_id", "session_id", "encounter_id", "region", "split", "ann_id",
                      "cm_per_px", "cm_per_px_unc", "scale_method", "mask_method", "date")}
    g.attrs.update({**attrs, "complete": True})
    return sum(a.nbytes for a in arrays.values())


def run(regions_path: Path, out: Path, regions=("flank", "face"), low_priority: bool = False, every: int = 10) -> dict:
    if low_priority:
        lower_priority()
    reg = pd.read_parquet(regions_path)
    reg = reg[reg.region.isin(regions)].reset_index(drop=True)
    distances = list(load_config("grid").distances_m["values"])
    eye = load_config("eye")
    root = zarr.open_group(str(out), mode="a")
    if "conds" not in root.attrs:
        root.attrs.update({"conds": list(CONDS), "distances_m": distances,
                           "eye_config": OmegaConf.to_container(eye, resolve=True),
                           "regions_file": Path(regions_path).resolve().relative_to(REPO_ROOT).as_posix(),
                           "created": dt.datetime.now().isoformat(timespec="seconds"),
                           "licence": "derived from GZGC CC BY-NC-ND 2.0: internal, not for public display"})
    rows = root.require_group("rows")
    pipe = Pipeline(["geometry", "eye"], eye_cfg=eye)
    done = {k for k in rows.group_keys() if rows[k].attrs.get("complete")}
    todo = [r for _, r in reg.iterrows() if row_key(r) not in done]
    print(f"{len(reg)} rows, {len(done)} already complete, {len(todo)} to run "
          f"({len(todo) * len(CONDS) * len(distances)} percepts)", flush=True)
    t0 = time.perf_counter()
    raw = 0
    for k, row in enumerate(todo, 1):
        percepts = percepts_for_row(pipe, row, distances)
        raw += write_row(rows, row_key(row), row, percepts, len(distances))
        if k % every == 0 or k == len(todo):
            el = time.perf_counter() - t0
            print(f"{k:5d}/{len(todo)} rows  {el:6.0f} s  {el / k:5.1f} s/row  ~{el / k * (len(todo) - k):5.0f} s left",
                  flush=True)
    el = time.perf_counter() - t0
    return {"rows": len(todo), "seconds": el, "raw_bytes_padded": raw}


def store_size(path: Path) -> int:
    return sum(p.stat().st_size for p in Path(path).rglob("*") if p.is_file())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("regions", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--regions-used", default="flank,face")
    ap.add_argument("--low-priority", action="store_true")
    a = ap.parse_args()
    res = run(a.regions, a.out, tuple(a.regions_used.split(",")), a.low_priority)
    size = store_size(a.out)
    print(f"done: {res['rows']} rows in {res['seconds']:.0f} s; store {size / 1e9:.3f} GB on disk "
          f"(padded raw {res['raw_bytes_padded'] / 1e9:.2f} GB)", flush=True)


if __name__ == "__main__":
    main()
