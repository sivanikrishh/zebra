"""Estimate the Zarr store size for percepts: subset x {flank, face} x 16 distances x {c0, c7, c8, c9}.

    python -m zebrasight.data.storage_estimate [n_sample_rows]
    python -m zebrasight.data.storage_estimate --from-pilot <regions_pilot.parquet> <pilot.zarr> <seconds>

Uncompressed size: exact, from each region's pixel size and cm_per_px through the geometry
and storage-grid rules (no simulation). Stored per percept: y, s (not at night), rho as float16,
mask as uint8. Compression: blosc-zstd (clevel 5, bit-shuffle) ratios measured on real percepts
of a random sample of region rows (full eye, noise on), per array kind, then applied to the total.
Two layouts: A = spec (max_side 1024, s full resolution); B = capped (max_side 512, s at half
resolution).
"""

from __future__ import annotations

import math
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import torch
from numcodecs import Blosc

from zebrasight.config import load_config, make_cond
from zebrasight.data.load import load_region
from zebrasight.data.regions import OUT as REGIONS
from zebrasight.eye.geometry import deg_per_px
from zebrasight.eye.percept import storage_size
from zebrasight.eye.streak import LIGHT_TO_REGIME
from zebrasight.pipeline import Pipeline

CONDS = ("c0", "c7", "c8", "c9")
REGIONS_USED = ("flank", "face")
CODEC = Blosc(cname="zstd", clevel=5, shuffle=Blosc.BITSHUFFLE)
LAYOUTS = {"A_spec": {"max_side": 1024, "s_half": False}, "B_capped": {"max_side": 512, "s_half": True}}


def stored_shape(h: int, w: int, cm_per_px: float, d: float, acuity: float, grid: float, spc: float,
                 max_side: int) -> tuple[tuple[int, int], bool]:
    dpp = deg_per_px(cm_per_px, d)
    if dpp < grid:
        nh, nw = max(1, round(h * dpp / grid)), max(1, round(w * dpp / grid))
        dpp = dpp * h / nh
        h, w = nh, nw
    return storage_size(h, w, dpp, acuity, grid, spc, max_side)


def raw_bytes(reg: pd.DataFrame, max_side: int, s_half: bool) -> dict:
    eye, grid_cfg = load_config("eye"), load_config("grid")
    grid = float(eye.working_grid.deg_per_px.value)
    spc = float(eye.percept.samples_per_cycle.value)
    dists = list(grid_cfg.distances_m["values"])
    tot = {"y": 0, "s": 0, "rho": 0, "mask": 0, "n": 0, "display_limited": 0}
    for r in reg.itertuples():
        x0, y0, x1, y1 = r.bbox_xyxy
        h, w = y1 - y0, x1 - x0
        for c in CONDS:
            light = grid_cfg.conditions[c].light
            acuity = float(eye.acuity[LIGHT_TO_REGIME[light]].value)
            for d in dists:
                (sh, sw), lim = stored_shape(h, w, r.cm_per_px, d, acuity, grid, spc, max_side)
                px = sh * sw
                tot["y"] += 2 * px
                tot["rho"] += 2 * px
                tot["mask"] += px
                if light != "night":
                    tot["s"] += 2 * (math.ceil(sh / 2) * math.ceil(sw / 2) if s_half else px)
                tot["n"] += 1
                tot["display_limited"] += int(lim)
    return tot


def measured_ratios(reg: pd.DataFrame, n_rows: int, seed: int = 0) -> dict:
    sample = reg.sample(n_rows, random_state=seed)
    pipe = Pipeline(["geometry", "eye"])
    dists = list(load_config("grid").distances_m["values"])
    raw = {"y": 0, "s": 0, "rho": 0, "mask": 0}
    comp = dict(raw)
    for _, row in sample.iterrows():
        img, mask, meta = load_region(row)
        for c in CONDS:
            for d in dists:
                p = pipe.run(img, float(row.cm_per_px), mask, make_cond(c, d), meta=meta)
                arrays = {"y": p.y, "rho": p.rho, "mask": p.mask}
                if p.s is not None:
                    arrays["s"] = p.s
                for k, t in arrays.items():
                    a = t[0, 0].detach().cpu().numpy()
                    a = a.astype(np.uint8) if k == "mask" else a.astype(np.float16)
                    raw[k] += a.nbytes
                    comp[k] += len(CODEC.encode(np.ascontiguousarray(a)))
    return {k: raw[k] / max(comp[k], 1) for k in raw}


def from_pilot(pilot_regions: Path, pilot_store: Path, seconds: float) -> None:
    """Project the full run from a finished pilot: measured on-disk bytes and seconds per row,
    scaled by the analytic raw size (exact per-row pixel counts) of pilot vs full rows."""
    from zebrasight.pipeline.store import store_size

    full = pd.read_parquet(REGIONS)
    pilot = pd.read_parquet(pilot_regions)
    full, pilot = (d[d.region.isin(REGIONS_USED)] for d in (full, pilot))
    disk = store_size(pilot_store)
    out = {}
    for name, lay in LAYOUTS.items():
        rp = raw_bytes(pilot, lay["max_side"], lay["s_half"])
        rf = raw_bytes(full, lay["max_side"], lay["s_half"])
        raw_p = sum(rp[k] for k in ("y", "s", "rho", "mask"))
        raw_f = sum(rf[k] for k in ("y", "s", "rho", "mask"))
        out[name] = (raw_p, raw_f, rf["display_limited"], rf["n"])
    raw_pA, raw_fA, _, _ = out["A_spec"]
    ratio = raw_pA / disk  # the pilot ran layout A
    print(f"pilot: {len(pilot)} rows, {len(pilot) * 64} percepts, {seconds:.0f} s ({seconds / len(pilot):.1f} s/row), "
          f"{disk / 1e9:.3f} GB on disk (raw {raw_pA / 1e9:.2f} GB, ratio {ratio:.2f}x)")
    print(f"full subset: {len(full)} rows, {len(full) * 64} percepts; "
          f"time ~{seconds / len(pilot) * len(full) / 3600:.1f} h at the pilot rate")
    for name, (raw_p, raw_f, lim, n) in out.items():
        print(f"  {name}: raw {raw_f / 1e9:.1f} GB -> ~{raw_f / ratio / 1e9:.1f} GB on disk; "
              f"display_limited {lim}/{n} percepts")


def main() -> None:
    if "--from-pilot" in sys.argv:
        i = sys.argv.index("--from-pilot")
        from_pilot(Path(sys.argv[i + 1]), Path(sys.argv[i + 2]), float(sys.argv[i + 3]))
        return
    n_rows = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    reg = pd.read_parquet(REGIONS)
    reg = reg[reg.region.isin(REGIONS_USED)]
    t0 = time.perf_counter()
    ratios = measured_ratios(reg, n_rows)
    t_sample = time.perf_counter() - t0
    print(f"rows {len(reg)} ({reg.region.value_counts().to_dict()}), percepts {len(reg) * 16 * len(CONDS)}")
    print("measured blosc-zstd ratios:", {k: round(v, 2) for k, v in ratios.items()},
          f"({n_rows} rows x 64 percepts in {t_sample:.0f} s -> {t_sample / (n_rows * 64):.3f} s/percept)")
    for name, lay in LAYOUTS.items():
        tot = raw_bytes(reg, lay["max_side"], lay["s_half"])
        raw_gb = sum(tot[k] for k in ("y", "s", "rho", "mask")) / 1e9
        comp_gb = sum(tot[k] / ratios[k] for k in ("y", "s", "rho", "mask")) / 1e9
        print(f"{name}: raw {raw_gb:.1f} GB -> compressed ~{comp_gb:.1f} GB; per kind raw GB "
              f"{ {k: round(tot[k] / 1e9, 2) for k in ('y', 's', 'rho', 'mask')} }; "
              f"display_limited percepts {tot['display_limited']}/{tot['n']}")
    print(f"est. full run time at the sampled rate: {t_sample / (n_rows * 64) * len(reg) * 64 / 3600:.1f} h")


if __name__ == "__main__":
    main()
