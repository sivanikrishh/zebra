"""Pack / unpack the small derived-data bundle B needs to regenerate percepts on another machine.

    python -m zebrasight.data.handoff pack   [out.zip]    # default cache/zebrasight_data_handoff.zip
    python -m zebrasight.data.handoff unpack <in.zip>

Bundle: data/masks/gzgc_v2/*.png (region rule v2: OWLv2 faces, trunk flank/rump; used by
data/regions.parquet), data/masks/gzgc/*.png (rule v1, kept for the pilot), data/regions.parquet
(full-run subset, rule v2), data/regions_v1.parquet, data/regions_all.parquet, data/regions_pilot.parquet,
data/regions_demo5_v2.parquet, data/interim/*.parquet. Paths inside the
zip are POSIX and repo-relative, so it unpacks identically on Linux and Windows. The images
themselves are NOT included (licence; B unpacks the organiser's archive with
``python -m zebrasight.data.unpack``).
"""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path, PurePosixPath

from zebrasight.config import REPO_ROOT

DEFAULT = REPO_ROOT / "cache" / "zebrasight_data_handoff.zip"
INCLUDE = ("data/masks/gzgc", "data/masks/gzgc_v2", "data/regions.parquet", "data/regions_v1.parquet",
           "data/regions_all.parquet", "data/regions_pilot.parquet", "data/regions_demo5_v2.parquet", "data/interim")


def pack(out: Path = DEFAULT) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for rel in INCLUDE:
            p = REPO_ROOT / rel
            files = [p] if p.is_file() else sorted(f for f in p.rglob("*") if f.is_file()) if p.exists() else []
            for f in files:
                if f.suffix == ".jsonl":
                    continue  # resumable mask log, not needed
                z.write(f, f.relative_to(REPO_ROOT).as_posix())
    return out


def unpack(src: Path) -> int:
    n = 0
    with zipfile.ZipFile(src) as z:
        for name in z.namelist():
            p = PurePosixPath(name)
            if p.is_absolute() or ".." in p.parts or p.parts[0] != "data":
                raise ValueError(f"refusing unsafe path in bundle: {name}")
        z.extractall(REPO_ROOT)
        n = len(z.namelist())
    return n


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "pack"
    if cmd == "pack":
        out = pack(Path(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT)
        print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")
    elif cmd == "unpack":
        print(f"unpacked {unpack(Path(sys.argv[2]))} files into {REPO_ROOT / 'data'}")
    else:
        raise SystemExit("usage: python -m zebrasight.data.handoff pack [out.zip] | unpack <in.zip>")
