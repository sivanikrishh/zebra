"""Pack every NON-git file the positional DGX run needs into one zip (paths relative to the repo root).

    python scripts/pack_handover.py [--out zebrasight_positional_handover.zip] [--dry-run]

Unzip it at the root of a fresh clone (branch positional). Contents:
  cache/percepts_50.zarr                    percept store (flank c0 etc.; A2 adds body + rump)
  data/*.parquet used here, heldout csv     region tables are gitignored (data/ is not in git)
  data/interim/*.parquet                    mask index (head boxes) used by the canonical frame
  data/masks/                               Mask R-CNN v2 label PNGs + body PNGs
  data/raw/gzgc/gzgc.coco/annotations       COCO annotations (A1 body check on new photos)
  original photos of the 50 + held-out zebras only (not the whole 20 GB data/raw)
  outputs/positional/                       frames cache, A1 table, smoke results (for comparison)
"""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

import pandas as pd

from zebrasight.config import REPO_ROOT

FILES = ["data/regions_50_positional.parquet", "data/regions_50.parquet", "data/regions.parquet",
         "data/heldout_zebras.csv"]
DIRS = ["cache/percepts_50.zarr", "data/interim", "data/masks", "data/raw/gzgc/gzgc.coco/annotations",
        "outputs/positional"]


def file_list() -> list[Path]:
    out = [REPO_ROOT / f for f in FILES]
    for d in DIRS:
        out += [p for p in (REPO_ROOT / d).rglob("*") if p.is_file()]
    photos = pd.concat([pd.read_parquet(REPO_ROOT / "data/regions_50_positional.parquet").image_path,
                        pd.read_csv(REPO_ROOT / "data/heldout_zebras.csv").image_path]).drop_duplicates()
    out += [REPO_ROOT / p for p in photos]
    missing = [p for p in out if not p.exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} files missing, e.g. {missing[:3]}")
    return sorted(set(out))


def main() -> None:
    ap = argparse.ArgumentParser(description="Pack the non-git files for the positional DGX run")
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "zebrasight_positional_handover.zip")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    files = file_list()
    size = sum(p.stat().st_size for p in files)
    print(f"{len(files)} files, {size / 1e9:.2f} GB")
    if a.dry_run:
        return
    with zipfile.ZipFile(a.out, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as z:  # jpg/zarr already compressed
        for i, p in enumerate(files, 1):
            z.write(p, p.relative_to(REPO_ROOT).as_posix())
            if i % 5000 == 0:
                print(f"  {i}/{len(files)}", flush=True)
    print(f"wrote {a.out} ({a.out.stat().st_size / 1e9:.2f} GB)")


if __name__ == "__main__":
    main()
