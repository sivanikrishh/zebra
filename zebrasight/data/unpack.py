"""Unpack the GZGC COCO archive safely, with progress.

    python -m zebrasight.data.unpack [archive] [out_dir]

Defaults: data/raw/gzgc.coco.tar.gz -> data/raw/gzgc/. Uses tarfile's "data" extraction
filter (rejects absolute paths, "..", links and device files). Skips files that already
exist with the right size, so an interrupted run can be resumed.
"""

from __future__ import annotations

import sys
import tarfile
import time
from pathlib import Path

from zebrasight.config import REPO_ROOT

ARCHIVE = REPO_ROOT / "data" / "raw" / "gzgc.coco.tar.gz"
OUT = REPO_ROOT / "data" / "raw" / "gzgc"


def unpack(archive: Path = ARCHIVE, out: Path = OUT, every: int = 250) -> tuple[int, int]:
    out.mkdir(parents=True, exist_ok=True)
    total = archive.stat().st_size
    t0 = time.perf_counter()
    n = skipped = 0
    with archive.open("rb") as raw, tarfile.open(fileobj=raw, mode="r|gz") as tf:
        for m in tf:
            target = out / m.name
            if m.isfile() and target.exists() and target.stat().st_size == m.size:
                skipped += 1
            else:
                tf.extract(m, out, filter="data")
            if m.isfile():
                n += 1
                if n % every == 0:
                    done = raw.tell() / total
                    el = time.perf_counter() - t0
                    eta = el / done - el if done > 0 else float("nan")
                    print(f"{n:5d} files  {100 * done:5.1f}% of archive  {el:5.0f} s elapsed  ~{eta:4.0f} s left",
                          flush=True)
    print(f"done: {n} files ({skipped} already present) in {time.perf_counter() - t0:.0f} s -> {out}", flush=True)
    return n, skipped


if __name__ == "__main__":
    args = [Path(a) for a in sys.argv[1:3]]
    unpack(*args)
