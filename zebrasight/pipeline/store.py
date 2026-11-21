"""Run percepts for region rows and write them to a Zarr store (layout v2: one group per condition).

    python -m zebrasight.pipeline.store <regions.parquet> <out.zarr> [--conds c0,c8,c9] [--regions-used flank,face]
                                        [--low-priority]
    python -m zebrasight.pipeline.store --migrate-v1 <old.zarr> <new.zarr>

Layout v2 (zarr v3, blosc-zstd clevel 5 bit-shuffle, one shard file per array):
    /                          attrs: layout_version=2, distances_m, eye config snapshot, aliases, regions file
    /rows/<ann_id>_<region>/   attrs: the region row's IDs and scale
        <cond_id>/             one group per condition, ADDITIVE: new conditions never touch existing ones
            y, rho             float16 (16 distances, H, W), zero-padded; valid area [:h, :w] from `shape`
            s                  float16 (16, ceil(H/2), ceil(W/2)) S cone at half resolution (absent at night);
                               valid area from `s_shape`
            mask               uint8 (16, H, W)
            shape, s_shape     int32 (16, 2)
            deg_per_px         float32 (16)      (s pixels are 2x this)
            display_limited, source_limited   bool (16)
            attrs: complete=True
        <alias_id>/            attrs: alias_of=<cond_id> (no arrays), e.g. c7 -> c8 (configs/grid.yaml store_aliases)
A condition group marked complete is skipped, so an interrupted run resumes and a later run with
more conditions only computes the new ones. Everything is regenerable from the regions parquet +
mask PNGs + configs (noise seeds are hashes of the IDs).
"""

from __future__ import annotations

import argparse
import datetime as dt
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import zarr
from omegaconf import OmegaConf
from zarr.codecs import BloscCodec

from zebrasight.config import REPO_ROOT, load_config, make_cond
from zebrasight.data.load import load_region
from zebrasight.eye.ops import gaussian_resize
from zebrasight.pipeline.pipeline import Pipeline

CODEC = BloscCodec(cname="zstd", clevel=5, shuffle="bitshuffle")
LAYOUT_VERSION = 2


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


def peak_rss_gb() -> float:
    try:
        import psutil

        info = psutil.Process().memory_info()
        return getattr(info, "peak_wset", info.rss) / 1e9
    except ImportError:
        return float("nan")


def _put(g: zarr.Group, name: str, a: np.ndarray, chunked: bool) -> None:
    if chunked:
        arr = g.create_array(name, shape=a.shape, chunks=(1, *a.shape[1:]), shards=a.shape, dtype=a.dtype,
                             fill_value=0, compressors=CODEC, overwrite=True)
    else:
        arr = g.create_array(name, shape=a.shape, dtype=a.dtype, overwrite=True)
    arr[:] = a


def write_condition(g: zarr.Group, percepts: list) -> int:
    """percepts: one Percept per distance (index order). Returns raw bytes written."""
    n = len(percepts)
    hmax = max(p.y.shape[-2] for p in percepts)
    wmax = max(p.y.shape[-1] for p in percepts)
    y = np.zeros((n, hmax, wmax), np.float16)
    rho = np.zeros_like(y)
    mask = np.zeros((n, hmax, wmax), np.uint8)
    has_s = percepts[0].s is not None
    s = np.zeros((n, math.ceil(hmax / 2), math.ceil(wmax / 2)), np.float16) if has_s else None
    shape = np.zeros((n, 2), np.int32)
    s_shape = np.zeros((n, 2), np.int32)
    dpp = np.zeros(n, np.float32)
    disp = np.zeros(n, bool)
    src = np.zeros(n, bool)
    for i, p in enumerate(percepts):
        h, w = p.y.shape[-2:]
        y[i, :h, :w] = p.y[0, 0].cpu().numpy()
        rho[i, :h, :w] = p.rho[0, 0].cpu().numpy()
        mask[i, :h, :w] = p.mask[0, 0].cpu().numpy() > 0.5
        if has_s:
            hs, ws = math.ceil(h / 2), math.ceil(w / 2)
            s_small = gaussian_resize(p.s, (hs, ws)) if (hs, ws) != (h, w) else p.s
            s[i, :hs, :ws] = s_small[0, 0].cpu().numpy()
            s_shape[i] = (hs, ws)
        shape[i] = (h, w)
        dpp[i] = p.deg_per_px
        disp[i] = p.display_limited
        src[i] = p.source_limited
    arrays = {"y": y, "rho": rho, "mask": mask} | ({"s": s} if has_s else {})
    for name, a in arrays.items():
        _put(g, name, a, chunked=True)
    for name, a in (("shape", shape), ("s_shape", s_shape), ("deg_per_px", dpp), ("display_limited", disp),
                    ("source_limited", src)):
        _put(g, name, a, chunked=False)
    g.attrs["complete"] = True
    return sum(a.nbytes for a in arrays.values())


ROW_ATTRS = ("zebra_id", "image_id", "session_id", "encounter_id", "region", "split", "ann_id", "cm_per_px",
             "cm_per_px_unc", "scale_method", "mask_method", "date")


def _row_attrs(row) -> dict:
    return {k: (row[k].item() if hasattr(row[k], "item") else row[k]) for k in ROW_ATTRS if k in row}


def _portable_path(p) -> str:
    """Repo-relative POSIX path when inside the repo (portable Windows <-> Linux), else absolute."""
    p = Path(p).resolve()
    try:
        return p.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return p.as_posix()


def open_root(out: Path, regions_path: Path | None, eye, distances, aliases: dict) -> zarr.Group:
    root = zarr.open_group(str(out), mode="a")
    if "layout_version" not in root.attrs:
        root.attrs.update({
            "layout_version": LAYOUT_VERSION, "distances_m": distances,
            "eye_config": OmegaConf.to_container(eye, resolve=True), "aliases": aliases,
            "regions_file": _portable_path(regions_path) if regions_path else None,
            "created": dt.datetime.now().isoformat(timespec="seconds"),
            "licence": "derived from GZGC CC BY-NC-ND 2.0: internal, not for public display"})
    elif root.attrs["layout_version"] != LAYOUT_VERSION:
        raise ValueError(f"{out} has layout {root.attrs['layout_version']}; migrate it first (--migrate-v1)")
    return root


def run(regions_path: Path, out: Path, conds: list[str] | None = None, regions=("flank", "face"),
        low_priority: bool = False, every: int = 10, profile=None) -> dict:
    if low_priority:
        lower_priority()
    grid = load_config("grid")
    eye = load_config("eye")
    conds = conds or list(grid.conditions.keys())
    aliases = {k: v for k, v in OmegaConf.to_container(grid.get("store_aliases", {})).items()}
    distances = list(grid.distances_m["values"])
    reg = pd.read_parquet(regions_path)
    reg = reg[reg.region.isin(regions)].reset_index(drop=True)
    root = open_root(out, regions_path, eye, distances, aliases)
    rows = root.require_group("rows")
    pipe = Pipeline(["geometry", "eye"], eye_cfg=eye)
    computed = [c for c in conds if c not in aliases]
    for a, src in aliases.items():
        if a in conds and src not in computed:
            computed.append(src)  # an alias needs its source
    t0 = time.perf_counter()
    n_rows = n_percepts = 0
    raw = 0
    writer = ThreadPoolExecutor(max_workers=1)  # zarr/blosc writes overlap the next GPU batch
    pending = []
    for k, row in enumerate(reg.itertuples(index=False), 1):
        row = row._asdict()
        g = rows.require_group(row_key(row))
        if "zebra_id" not in g.attrs:
            g.attrs.update(_row_attrs(row))
        todo = [c for c in computed if not (c in g.group_keys() and g[c].attrs.get("complete"))]
        if todo:
            img, mask, meta = load_region(row)
            geo_cache: dict = {}  # geometry shared by conditions with the same distance/angle/regime
            for c in todo:
                percepts = pipe.run_distances(img, float(row["cm_per_px"]), mask, make_cond(c, distances[0]),
                                              distances, meta=meta, geo_cache=geo_cache)
                n_percepts += len(percepts)
                pending.append(writer.submit(write_condition, g.require_group(c), percepts))
                del percepts
            for fut in pending:  # bounded memory: at most one crop's conditions in flight
                raw += fut.result()
            pending.clear()
            del img, mask, geo_cache
            n_rows += 1
        for a, src in aliases.items():
            if a in conds and a not in g.group_keys():
                g.require_group(a).attrs.update({"alias_of": src})
        if k % every == 0 or k == len(reg):
            el = time.perf_counter() - t0
            rate = el / max(n_rows, 1)
            left = rate * (len(reg) - k)
            print(f"{k:5d}/{len(reg)} rows ({n_rows} computed)  {el:6.0f} s  {rate:5.1f} s/row  ~{left:5.0f} s left  "
                  f"peak RAM {peak_rss_gb():.2f} GB", flush=True)
    writer.shutdown(wait=True)
    return {"rows_computed": n_rows, "percepts": n_percepts, "seconds": time.perf_counter() - t0,
            "raw_bytes": raw, "peak_rss_gb": peak_rss_gb()}


def migrate_v1(src: Path, dst: Path) -> dict:
    """v1 (conditions as an array axis) -> v2 (one group per condition), no recomputation.
    S is downsampled to half resolution (Gaussian pre-filter) to match v2; aliases (c7) are dropped
    in favour of the alias record."""
    old = zarr.open_group(str(src), mode="r")
    grid = load_config("grid")
    aliases = dict(OmegaConf.to_container(grid.get("store_aliases", {})))
    conds = list(old.attrs["conds"])
    new = zarr.open_group(str(dst), mode="a")
    new.attrs.update({"layout_version": LAYOUT_VERSION, "distances_m": old.attrs["distances_m"],
                      "eye_config": old.attrs["eye_config"], "aliases": aliases,
                      "regions_file": old.attrs.get("regions_file"), "created": old.attrs.get("created"),
                      "migrated_from": f"{src.as_posix()} (layout v1)", "licence": old.attrs.get("licence")})
    rows_new = new.require_group("rows")
    n = 0
    for key in old["rows"].group_keys():
        g_old = old["rows"][key]
        if not g_old.attrs.get("complete"):
            continue
        g = rows_new.require_group(key)
        g.attrs.update({k: v for k, v in g_old.attrs.items() if k != "complete"})
        shape_all = g_old["shape"][:]
        for ci, c in enumerate(conds):
            if c in aliases:
                g.require_group(c).attrs.update({"alias_of": aliases[c]})
                continue
            gc = g.require_group(c)
            if gc.attrs.get("complete"):
                continue
            shp = shape_all[ci]
            hmax, wmax = int(shp[:, 0].max()), int(shp[:, 1].max())
            y = g_old["y"][ci][:, :hmax, :wmax]
            rho = g_old["rho"][ci][:, :hmax, :wmax]
            mask = g_old["mask"][ci][:, :hmax, :wmax]
            arrays = {"y": y, "rho": rho, "mask": mask}
            s_shape = np.zeros_like(shp)
            if c != "c9":
                s_full = g_old["s"][ci]
                s = np.zeros((len(shp), math.ceil(hmax / 2), math.ceil(wmax / 2)), np.float16)
                for i, (h, w) in enumerate(shp):
                    hs, ws = math.ceil(h / 2), math.ceil(w / 2)
                    t = torch.from_numpy(s_full[i, :h, :w].astype(np.float32))[None, None]
                    s[i, :hs, :ws] = gaussian_resize(t, (hs, ws))[0, 0].numpy() if (hs, ws) != (h, w) else t[0, 0]
                    s_shape[i] = (hs, ws)
                arrays["s"] = s
            for name, a in arrays.items():
                _put(gc, name, a, chunked=True)
            for name, a in (("shape", shp.astype(np.int32)), ("s_shape", s_shape.astype(np.int32)),
                            ("deg_per_px", g_old["deg_per_px"][ci]), ("display_limited", g_old["display_limited"][ci]),
                            ("source_limited", g_old["source_limited"][ci])):
                _put(gc, name, np.asarray(a), chunked=False)
            gc.attrs["complete"] = True
        n += 1
    return {"rows": n}


def store_size(path: Path) -> int:
    return sum(p.stat().st_size for p in Path(path).rglob("*") if p.is_file())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("regions", type=Path, nargs="?")
    ap.add_argument("out", type=Path, nargs="?")
    ap.add_argument("--conds", default=None, help="comma list; default all conditions in configs/grid.yaml")
    ap.add_argument("--regions-used", default="flank,face")
    ap.add_argument("--low-priority", action="store_true")
    ap.add_argument("--migrate-v1", nargs=2, type=Path, metavar=("OLD", "NEW"))
    a = ap.parse_args()
    if a.migrate_v1:
        t0 = time.perf_counter()
        res = migrate_v1(*a.migrate_v1)
        print(f"migrated {res['rows']} rows in {time.perf_counter() - t0:.0f} s; "
              f"{store_size(a.migrate_v1[1]) / 1e9:.3f} GB on disk")
        return
    conds = a.conds.split(",") if a.conds else None
    res = run(a.regions, a.out, conds, tuple(a.regions_used.split(",")), a.low_priority)
    print(f"done: {res['rows_computed']} rows, {res['percepts']} percepts in {res['seconds']:.0f} s; "
          f"store {store_size(a.out) / 1e9:.3f} GB on disk; peak RAM {res['peak_rss_gb']:.2f} GB", flush=True)


if __name__ == "__main__":
    main()
