"""Exit 0 if every valid flank/face row of the store has the given condition, else print the count and exit 1.

    python scripts/check_store.py cache/percepts_50.zarr c9
"""

import sys

from zebrasight.pipeline.read_store import PerceptStore

if __name__ == "__main__":
    store, cond = sys.argv[1], sys.argv[2]
    r = PerceptStore(store).rows()
    r = r[r.region.isin(["flank", "face"])]
    if "valid" in r:
        r = r[r.valid != False]  # noqa: E712
    missing = int((~r.conds.apply(lambda c: cond in c)).sum())
    print(f"{cond}: {len(r) - missing}/{len(r)} rows complete")
    sys.exit(1 if missing else 0)
