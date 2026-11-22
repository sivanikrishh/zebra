"""Build cache/percepts_50.zarr for the 50 pilot zebras (rule-v2 regions), in priority order.

    python scripts/run_percepts_50.py

Passes (each resumable: finished condition groups are skipped on rerun):
  1 c0 (flank + face)  2 c8 + c7 alias  3 c9  4 c2  5 rump c0
Prints "PASS <name> DONE" after each pass so downstream jobs (features on c0) can start.
"""

from __future__ import annotations

import time
from pathlib import Path

from zebrasight.config import REPO_ROOT
from zebrasight.pipeline.store import run, store_size

REGIONS = REPO_ROOT / "data" / "regions_50.parquet"
OUT = REPO_ROOT / "cache" / "percepts_50.zarr"
PASSES = [
    ("c0", ["c0"], ("flank", "face")),
    ("c8", ["c7", "c8"], ("flank", "face")),
    ("c9", ["c9"], ("flank", "face")),
    ("c2", ["c2"], ("flank", "face")),
    ("rump_c0", ["c0"], ("rump",)),
]


def main() -> None:
    t0 = time.perf_counter()
    for name, conds, regions in PASSES:
        print(f"PASS {name} START ({conds}, {regions})", flush=True)
        res = run(Path(REGIONS), Path(OUT), conds, regions, low_priority=False)
        print(f"PASS {name} DONE: {res['rows_computed']} rows, {res['percepts']} percepts, {res['seconds']:.0f} s, "
              f"peak RAM {res['peak_rss_gb']:.2f} GB", flush=True)
    print(f"ALL DONE in {time.perf_counter() - t0:.0f} s; store {store_size(OUT) / 1e9:.2f} GB", flush=True)


if __name__ == "__main__":
    main()
