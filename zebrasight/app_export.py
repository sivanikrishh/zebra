"""Web-app data pack: python -m zebrasight.app_export [--dummy]

Writes app_data/results.json (schema in app_data/README.md). --dummy fills every field with
made-up but correctly shaped numbers (status "DUMMY") so the app can be built before the results.
The app only plays back this file; it computes nothing.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

from zebrasight.config import REPO_ROOT

SCHEMA_VERSION = "1.0"
DISTANCES = [2.0, 2.719, 3.696, 5.024, 6.829, 9.283, 12.619, 17.154, 23.318, 31.698, 43.089, 58.573,
             79.621, 108.234, 147.128, 200.0]
CONDITIONS = [{"id": "c0", "label": "Clear day"}, {"id": "c8", "label": "Dusk"}, {"id": "c9", "label": "Night (moonlight)"}]
REGIONS = [{"id": "face", "label": "Face"}, {"id": "flank", "label": "Flank"}]
OBSERVERS = [
    {"id": "megadesc", "label": "Learned re-ID model (MegaDescriptor)", "kind": "positional", "main": True},
    {"id": "hotspotter", "label": "Local stripe matcher (HotSpotter-style)", "kind": "positional", "main": False},
    {"id": "registered_layout", "label": "Registered stripe layout (aligned black-fraction map)", "kind": "positional",
     "main": False},
    {"id": "spatial_nn", "label": "Stripe statistics + coarse layout", "kind": "summary_statistics", "main": False},
    {"id": "interpretable_nn", "label": "Stripe statistics (F1-F10)", "kind": "summary_statistics", "main": False},
    {"id": "raw_pixel", "label": "Raw pixels (no size rule)", "kind": "baseline", "main": False},
]
FAMILIES = {"F1": "Stripe width", "F2": "Black fraction", "F3": "Curvature", "F4": "Orientation",
            "F5": "Junctions", "F6": "Endings", "F7": "Spacing / period", "F8": "Shadow stripes",
            "F9": "Contrast & sharpness", "F10": "Topology", "L": "Coarse layout", "M": "Minutiae map"}
PHOTO_LIMITED_MAX_M = 9.283
N_BOOT = 200
OUT = REPO_ROOT / "app_data"


def _r(x, k=3):
    return None if x is None or (isinstance(x, float) and not math.isfinite(x)) else round(float(x), k)


def dummy() -> dict:
    rng = np.random.default_rng(0)
    curves, dstar, fam = [], [], []
    for c_i, c in enumerate(CONDITIONS):
        for r in REGIONS:
            for o in OBSERVERS:
                top = {"megadesc": 0.87, "spatial_nn": 0.53, "interpretable_nn": 0.53, "raw_pixel": 0.54}.get(o["id"], 0.75)
                top = 0.5 + (top - 0.5) * (0.4 if r["id"] == "face" else 1.0) * (1.0, 0.8, 0.5)[c_i]
                pts = []
                for i, d in enumerate(DISTANCES):
                    a = 0.5 + (top - 0.5) / (1 + (d / (60 / (c_i + 1))) ** 2)
                    lo, hi = a - 0.04, min(a + 0.04, 1.0)
                    t5 = 0.2 + (a - 0.5) * 1.4
                    t20 = 0.05 + (a - 0.5) * 1.0
                    pts.append({"d_index": i, "distance_m": d, "auc": _r(a), "auc_ci": [_r(lo), _r(hi)],
                                "top1_n5": _r(t5), "top1_n5_ci": [_r(t5 - 0.1), _r(t5 + 0.1)], "chance_n5": 0.2,
                                "top1_n20": _r(t20), "top1_n20_ci": [_r(t20 - 0.08), _r(t20 + 0.08)], "chance_n20": 0.05,
                                "usable": bool(lo >= 0.80), "photo_limited": d <= PHOTO_LIMITED_MAX_M,
                                "size_rule_share": _r(0.0 if d < 100 else (d - 100) / 100)})
                curves.append({"region": r["id"], "condition": c["id"], "observer": o["id"], "points": pts})
                usable = [p["distance_m"] for p in pts if p["usable"]]
                ds = max(usable) if usable else None
                dstar.append({"region": r["id"], "condition": c["id"], "observer": o["id"], "dstar_auc_m": ds,
                              "dstar_auc_ci_m": [None if ds is None else _r(ds * 0.7), None if ds is None else _r(ds * 1.4)],
                              "never_usable": ds is None})
            for d in DISTANCES:
                ranked = sorted(({"family": f, "label": FAMILIES[f], "auc": _r(0.5 + rng.uniform(0, 0.05))}
                                 for f in FAMILIES), key=lambda x: -x["auc"])
                fam.append({"region": r["id"], "condition": c["id"], "distance_m": d, "families": ranked})
    loss = [{"region": r["id"], "observer": "megadesc", "condition": c, "vs": "c0", "ratio": _r(v),
             "ratio_ci": [_r(v * 0.7), _r(v * 1.3)]} for r in REGIONS for c, v in (("c8", 0.6), ("c9", 0.25))]
    return {
        "dstar": dstar, "curves": curves, "leading_features": fam, "distance_loss_ratio": loss,
        "original_crops": [{"region": r["id"], "observer": "megadesc", "auc": v, "auc_ci": [_r(v - 0.04), _r(v + 0.04)],
                            "top1_n20": t, "chance_n20": 0.05} for r, v, t in ((REGIONS[0], 0.555, 0.18), (REGIONS[1], 0.867, 0.49))],
        "statistics_vs_positions": {
            "headline": "Summary stripe statistics do not identify individuals across days even at 2 m; identity is positional.",
            "at_2m_c0": [{"region": r["id"], "observer": o["id"], "auc": 0.5} for r in REGIONS for o in OBSERVERS],
            "same_day_pairs_auc": [{"region": "flank", "observer": "interpretable_nn", "auc": 0.755},
                                   {"region": "face", "observer": "interpretable_nn", "auc": 0.657}]},
    }


def envelope(body: dict, status: str) -> dict:
    return {
        "schema_version": SCHEMA_VERSION, "status": status,
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "n_zebras": 50, "n_bootstrap": N_BOOT,
        "bootstrap_note": "95% CIs from 200 bootstrap resamples over zebras (final for this demo)",
        "distances_m": DISTANCES, "conditions": CONDITIONS, "regions": REGIONS, "observers": OBSERVERS,
        "families": [{"id": k, "label": v} for k, v in FAMILIES.items()],
        "thresholds": {"usable_auc_lower_ci": 0.80, "reliable_auc_lower_ci": 0.95, "chance_auc": 0.5},
        "markers": {"photo_limited_max_m": PHOTO_LIMITED_MAX_M,
                    "size_rule_median_m": {"face": 147.1, "flank": 200.0}},
        **body,
    }


def write(data: dict, out: Path = OUT) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    p = out / "results.json"
    p.write_text(json.dumps(data, indent=1), encoding="utf-8")
    return p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dummy", action="store_true")
    a = ap.parse_args()
    if a.dummy:
        print("wrote", write(envelope(dummy(), "DUMMY - made-up numbers, schema only")))
    else:
        from zebrasight.app_results import real  # filled from outputs/metrics/*

        print("wrote", write(envelope(real(), "final")))
        copy_assets()
        print("wrote", make_zip())


def copy_assets(out: Path = OUT) -> None:
    """figures/, params/, textures/ (+ human-view originals) into app_data/."""
    import shutil

    figs = out / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    srcs = sorted((REPO_ROOT / "outputs" / "figures").glob("*.png"))
    srcs += [REPO_ROOT / "docs" / "figures" / "calibration.png", REPO_ROOT / "docs" / "figures" / "haze_resolution.png"]
    for s in srcs:
        if s.exists():
            shutil.copy2(s, figs / s.name)
        else:
            print("missing figure:", s)
    shutil.copytree(REPO_ROOT / "vr" / "export" / "params", out / "params", dirs_exist_ok=True)
    tex = out / "textures"
    shutil.copytree(REPO_ROOT / "vr" / "export" / "textures", tex, dirs_exist_ok=True)
    human_views(tex)
    wits = REPO_ROOT / "data" / "raw" / "wits"
    if wits.exists():
        shutil.copytree(wits, tex / "wits", dirs_exist_ok=True)
    else:
        print("no Wits scan textures (data/raw/wits absent): GZGC textures only")


def human_views(tex: Path, max_side: int = 512) -> None:
    """Add <zebra>/<region>/human.png (the original photo region, RGBA, alpha = mask) + 'human_view_file'."""
    import pandas as pd
    from PIL import Image

    from zebrasight.data.load import load_region

    idx = json.loads((tex / "textures.json").read_text(encoding="utf-8"))
    reg = pd.read_parquet(REPO_ROOT / "data" / "regions.parquet")
    done = {}
    for s in idx["sets"]:
        k = (s["zebra_id"], s["region"])
        if k not in done:
            row = reg[(reg.ann_id == s["ann_id"]) & (reg.region == s["region"])]
            if row.empty:
                done[k] = None
            else:
                img, mask, _ = load_region(row.iloc[0])
                rgb = (img[0].permute(1, 2, 0).numpy().clip(0, 1) * 255).astype(np.uint8)
                a = ((mask[0, 0].numpy() > 0.5) * 255).astype(np.uint8)
                im = Image.fromarray(np.dstack([rgb, a]), "RGBA")
                im.thumbnail((max_side, max_side), Image.LANCZOS)
                rel = f"{s['zebra_id']}/{s['region']}/human.png"
                (tex / rel).parent.mkdir(parents=True, exist_ok=True)
                im.save(tex / rel)
                done[k] = rel
        s["human_view_file"] = done[k]
    idx["human_view_note"] = "human.png = the original photo region as a human sees it (sRGB, alpha = mask)"
    (tex / "textures.json").write_text(json.dumps(idx, indent=1), encoding="utf-8")


def make_zip(out: Path = OUT) -> Path:
    import shutil

    return Path(shutil.make_archive(str(REPO_ROOT / "app_data"), "zip", root_dir=out.parent, base_dir=out.name))


if __name__ == "__main__":
    main()
