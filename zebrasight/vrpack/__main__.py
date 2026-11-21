"""python -m zebrasight.vrpack [--params-only]: build vr/export/ and vr_export.zip."""

from __future__ import annotations

import sys
import zipfile

import pandas as pd

from zebrasight.config import REPO_ROOT
from zebrasight.vrpack import params
from zebrasight.vrpack.textures import export, write_json

OUT = REPO_ROOT / "vr" / "export"
STORE = REPO_ROOT / "cache" / "percepts_demo_env.zarr"
ROWS = REPO_ROOT / "data" / "regions_vr5.parquet"  # best photo per demo zebra, flank + face
FEATURES = [f"F{i}" for i in range(1, 11)]


def write_params() -> None:
    for name, build in params.ALL.items():
        write_json(build(), OUT / "params" / name)
    write_json(params.dichromat(), REPO_ROOT / "vr" / "shaders" / "dichromat_matrix.json")  # keep in sync


def placeholder_lut(zebras: list[str], distances: list[float]) -> dict:
    """Dummy identification LUT so V can build the HUD; B replaces it with real feature values."""
    import numpy as np

    rng = np.random.default_rng(0)
    return {
        "PLACEHOLDER": "dummy values - schema proposed by A, not yet agreed (docs/contracts.md has no vr_lut schema yet)",
        "schema": {"zebras[].zebra_id": "str", "zebras[].region": "flank|face",
                   "zebras[].condition": "condition id", "zebras[].features": "{feature: [16 values, one per distance bin]}",
                   "zebras[].id_probability": "[16 values in 0..1]", "features": "list of feature names"},
        "distance_bins_m": distances, "features": FEATURES,
        "zebras": [{"zebra_id": z, "region": reg, "condition": c,
                    "features": {f: np.round(np.clip(rng.random() * np.exp(-np.array(distances) / 80), 0, 1), 3).tolist()
                                 for f in FEATURES},
                    "id_probability": np.round(1 / (1 + np.exp((np.array(distances) - 60) / 15)), 3).tolist()}
                   for z in zebras for reg in ("flank", "face") for c in ("c0", "c8", "c9", "c1", "c2", "c3")],
    }


def build_zip() -> None:
    zpath = OUT / "vr_export.zip"
    with zipfile.ZipFile(zpath, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for f in sorted(OUT.rglob("*")):
            if f.is_file() and f != zpath:
                z.write(f, f.relative_to(OUT).as_posix())
    print(f"wrote {zpath} ({zpath.stat().st_size / 1e6:.1f} MB)")


def main() -> None:
    write_params()
    if "--params-only" in sys.argv:
        return
    reg = pd.read_parquet(ROWS)
    keys = [f"{a}_{r}" for a, r in zip(reg.ann_id, reg.region)]
    tex = export(STORE, OUT / "textures", keys)
    size = {(int(a), r): (b, float(c)) for a, r, b, c in zip(reg.ann_id, reg.region, reg.bbox_xyxy, reg.cm_per_px)}
    for s in tex["sets"]:  # physical size of the textured region, for sizing the quad on the coat
        (x0, y0, x1, y1), cmpp = size[(s["ann_id"], s["region"])]
        s["region_size_cm"] = [round((x1 - x0) * cmpp, 2), round((y1 - y0) * cmpp, 2)]
    write_json(tex, OUT / "textures" / "textures.json")
    write_json(placeholder_lut(sorted(reg.zebra_id.unique()), tex["distance_bins_m"]), OUT / "vr_lut.json")
    n = sum(len(s["layers"]) for s in tex["sets"])
    print(f"{len(tex['sets'])} texture sets, {n} PNGs")
    build_zip()


if __name__ == "__main__":
    main()
