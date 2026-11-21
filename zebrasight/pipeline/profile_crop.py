"""Profile one region crop: 16 distances x unique conditions, time per stage, zarr write, peak memory.

    python -m zebrasight.pipeline.profile_crop [regions.parquet] [row_index] [--looped]

Default crop: the largest flank in data/regions.parquet (worst case). Stage times use CUDA
synchronisation, so they are wall-clock on the GPU.
"""

from __future__ import annotations

import collections
import sys
import tempfile
import time
from pathlib import Path

import pandas as pd
import torch
import zarr
from omegaconf import OmegaConf

import zebrasight.registry as registry
from zebrasight.config import REPO_ROOT, load_config, make_cond
from zebrasight.data.load import load_region
from zebrasight.pipeline.pipeline import Pipeline
from zebrasight.pipeline.store import peak_rss_gb, write_condition

TIMES: dict[str, float] = collections.defaultdict(float)


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _timed_call(orig):
    def call(self, *a, **k):
        _sync()
        t = time.perf_counter()
        out = orig(self, *a, **k)
        _sync()
        if self.stage_name != "eye":  # the eye wrapper's time is the sum of its sub-stages
            TIMES[self.stage_name] += time.perf_counter() - t
        return out

    return call


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    path = Path(args[0]) if args else REPO_ROOT / "data" / "regions.parquet"
    reg = pd.read_parquet(path)
    flank = reg[reg.region == "flank"].copy()
    flank["px"] = [(b[2] - b[0]) * (b[3] - b[1]) for b in flank.bbox_xyxy]
    pos = args[1:]
    row = flank.loc[flank.px.idxmax()] if not pos else reg.iloc[int(pos[0])]
    grid, eye = load_config("grid"), load_config("eye")
    aliases = OmegaConf.to_container(grid.get("store_aliases", {}))
    conds = [c for c in grid.conditions.keys() if c not in aliases]
    distances = list(grid.distances_m["values"])
    pipe = Pipeline(["geometry", "eye"], eye_cfg=eye)
    registry.Stage.__call__ = _timed_call(torch.nn.Module.__call__)
    from zebrasight.eye.csf import CSF

    orig_batch = CSF.forward_batch

    def timed_batch(self, states):  # the batched CSF bypasses __call__
        _sync()
        t0 = time.perf_counter()
        out = orig_batch(self, states)
        _sync()
        TIMES["csf"] += time.perf_counter() - t0
        return out

    if "--looped" not in sys.argv:  # looped mode is already timed through __call__
        CSF.forward_batch = timed_batch

    t = time.perf_counter()
    img, mask, meta = load_region(row)
    t_load = time.perf_counter() - t
    pipe.run(img, float(row.cm_per_px), mask, make_cond("c0", 50.0), meta=meta)  # warm-up
    TIMES.clear()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t_percept = t_write = 0.0
    out = Path(tempfile.mkdtemp()) / "profile.zarr"
    g = zarr.open_group(str(out), mode="a")
    batched = "--looped" not in sys.argv
    geo_cache: dict = {}
    for c in conds:
        _sync()
        t = time.perf_counter()
        if batched:
            percepts = pipe.run_distances(img, float(row.cm_per_px), mask, make_cond(c, distances[0]), distances,
                                          meta=meta, geo_cache=geo_cache)
        else:
            percepts = [pipe.run(img, float(row.cm_per_px), mask, make_cond(c, d), meta=meta) for d in distances]
        _sync()
        t_percept += time.perf_counter() - t
        t = time.perf_counter()
        write_condition(g.require_group(c), percepts)
        t_write += time.perf_counter() - t
    n = len(conds) * len(distances)
    x0, y0, x1, y1 = row.bbox_xyxy
    print(f"crop {row.zebra_id} {row.region} {y1 - y0}x{x1 - x0} px, cm/px {row.cm_per_px:.3f}; "
          f"{len(conds)} unique conditions x {len(distances)} distances = {n} percepts")
    total = t_load + t_percept + t_write
    print(f"load {t_load:.2f} s | percepts {t_percept:.2f} s ({t_percept / n * 1000:.0f} ms each) | "
          f"zarr write {t_write:.2f} s | total {total:.2f} s")
    stage_sum = sum(TIMES.values())
    for k, v in sorted(TIMES.items(), key=lambda kv: -kv[1]):
        print(f"  {k:10s} {v:6.2f} s  {100 * v / stage_sum:5.1f}%")
    print(f"  (pipeline overhead outside stages: {t_percept - stage_sum:.2f} s)")
    gpu = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else float("nan")
    print(f"peak RAM {peak_rss_gb():.2f} GB, peak GPU {gpu:.2f} GB")


if __name__ == "__main__":
    main()
