"""Build the team zip for B and V: python -m zebrasight.team_pack [out.zip]

Contents: code/ (git archive HEAD: tracked files only, so no .venv, data/raw, cache, logs,
__pycache__), data_handoff/ (cache/zebrasight_data_handoff.zip), vr_export/ (vr/export/vr_export.zip),
figures/ (docs/figures/*.png + docs/examples/masks_v2/qa_grid_v2.png), START_HERE.md at the top.
Commit first: the code part is exactly HEAD.
"""

from __future__ import annotations

import io
import subprocess
import sys
import zipfile
from pathlib import Path

from zebrasight.config import REPO_ROOT

OUT = REPO_ROOT / "zebrasight_team.zip"
EXCLUDE_PARTS = {".venv", "__pycache__", "logs"}


def build(out: Path = OUT) -> Path:
    archive = subprocess.run(["git", "archive", "--format=zip", "HEAD"], cwd=REPO_ROOT, check=True,
                             capture_output=True).stdout
    extras = {
        "data_handoff/zebrasight_data_handoff.zip": REPO_ROOT / "cache" / "zebrasight_data_handoff.zip",
        "vr_export/vr_export.zip": REPO_ROOT / "vr" / "export" / "vr_export.zip",
        "figures/qa_grid_v2.png": REPO_ROOT / "docs" / "examples" / "masks_v2" / "qa_grid_v2.png",
    }
    for f in sorted((REPO_ROOT / "docs" / "figures").glob("*.png")):
        extras[f"figures/{f.name}"] = f
    missing = [str(p) for p in extras.values() if not p.exists()]
    if missing:
        raise SystemExit(f"missing inputs: {missing}")
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.write(REPO_ROOT / "START_HERE.md", "START_HERE.md")
        with zipfile.ZipFile(io.BytesIO(archive)) as code:
            for info in code.infolist():
                parts = set(Path(info.filename).parts)
                if info.is_dir() or parts & EXCLUDE_PARTS or info.filename.startswith(("data/raw/", "cache/")):
                    continue
                z.writestr(f"code/{info.filename}", code.read(info))
        for arc, src in extras.items():  # already-compressed zips/PNGs: store without recompressing
            z.write(src, arc, compress_type=zipfile.ZIP_STORED)
    return out


if __name__ == "__main__":
    out = build(Path(sys.argv[1]) if len(sys.argv) > 1 else OUT)
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")
