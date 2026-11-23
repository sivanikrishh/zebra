"""Pattern Lab data for the 2D research prototype (the science lives in zebrasight/patternlab.py).

    python scripts/build_pattern_lab.py      (also run by scripts/build_app_demo.py) -> outputs/app_demo/lab.js, lab/

Experiment 1 "Move away": like-with-like distance measurements (study's stored inference for that exact pair)
  + the zebra-eye images; the full-body image in the scene is a VISUAL only (full-body results are pending).
Experiment 2 "Change the stripes": every slider step of every control is a real modified photo with real
  MegaDescriptor and HotSpotter inference (one control at a time; configs/positional.yaml pattern_lab).
Experiment 3 needs horse photos, which the dataset does not have: nothing is generated for it.
Transformed images are written only under outputs/app_demo/lab/ (never next to the study data).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

from zebrasight import patternlab as PL
from zebrasight.config import REPO_ROOT

OUT = REPO_ROOT / "outputs" / "app_demo"
LAB_ZEBRAS = ["IBEIS_PZ_0334", "IBEIS_PZ_0395", "IBEIS_PZ_1648", "IBEIS_PZ_1725"]   # later-day full-body photo exists
CONDS = ["c0", "c8", "c9"]
MAX_SHOW = 512


def png(path: Path, rgb: np.ndarray, mask: np.ndarray) -> str:
    im = Image.fromarray(np.dstack([(np.clip(rgb, 0, 1) * 255).astype(np.uint8), (mask * 255).astype(np.uint8)]), "RGBA")
    im.thumbnail((MAX_SHOW, MAX_SHOW), Image.LANCZOS)
    path.parent.mkdir(parents=True, exist_ok=True)
    im.save(path)
    return path.relative_to(OUT).as_posix()


def percept_rgba(p: dict, dm) -> tuple[np.ndarray, np.ndarray]:
    from zebrasight.metrics.megadesc import percept_rgb

    return np.clip(percept_rgb(p, dm), 0, 1), p["mask"]


def pick_rows(zebra: str):
    """The zebra's later-day photo that has a full-body row, and its earliest-day reference photo."""
    import pandas as pd

    pos = pd.read_parquet(REPO_ROOT / "data" / "regions_50_positional.parquet")
    body = pos[(pos.region == "body") & (pos.zebra_id == zebra) & (pos.split == "query")].sort_values("ann_id").iloc[0]
    rows = {}
    for region in PL.AVAILABLE_REGIONS:
        t = PL.region_table(region)
        q = t[(t.zebra_id == zebra) & (t.split == "query")]
        q = q[q.ann_id == body.ann_id] if (q.ann_id == body.ann_id).any() else q
        rows[region] = {"query": q.iloc[0].to_dict(), "ref": t[(t.zebra_id == zebra) & (t.split == "gallery")].iloc[0].to_dict()}
    return body.to_dict(), rows


def build(out: Path = OUT) -> dict:
    from zebrasight.pipeline.read_store import PerceptStore
    from zebrasight.positional import data_io as dio
    from zebrasight.vrpack.params import matrices

    t0 = time.time()
    c = PL.cfg()
    st, dm = PerceptStore(REPO_ROOT / "cache" / "percepts_50.zarr"), matrices()[4]
    results = json.loads((REPO_ROOT / "app_data" / "results.json").read_text(encoding="utf-8"))
    lab = {"zebras": LAB_ZEBRAS, "regions": list(PL.AVAILABLE_REGIONS), "pending_regions": ["body"],
           "controls": {k: list(c[k]) for k in PL.CONTROLS}, "distances_m": results["distances_m"],
           "conditions": [x for x in results["conditions"] if x["id"] in CONDS], "zebra": {}, "checks": {},
           "study": [{k: cv[k] for k in ("region", "condition", "observer")} |
                     {"points": [{k: p[k] for k in ("auc", "auc_ci", "top1_n20", "top1_n20_ci", "chance_n20")} for p in cv["points"]]}
                     for cv in results["curves"] if cv["observer"] in ("megadesc", "hotspotter") and cv["condition"] in CONDS]}
    picks = {z: pick_rows(z) for z in LAB_ZEBRAS}
    # ---- experiment 1: scene visual (full body, VISUAL ONLY) + like-with-like measurements per region
    for z, (body, rows) in picks.items():
        zz = lab["zebra"].setdefault(z, {"exp1": {}, "exp2": {}})
        h_m = (body["bbox_xyxy"][3] - body["bbox_xyxy"][1]) * body["cm_per_px"] / 100.0
        zz["body"] = {"height_m": round(float(h_m), 2), "date": str(body["date"])[:10], "img": {}}
        for cond in CONDS:
            P = dio.make_percepts(__import__("pandas").DataFrame([body | {"key": f"{body['ann_id']}_body"}]), range(16), cond)
            zz["body"]["img"][cond] = [png(out / "lab" / "body" / z / f"{cond}_d{d:02d}.png", *percept_rgba(P[(f"{body['ann_id']}_body", d)], dm))
                                       for d in range(16)]
        for region, rr in rows.items():
            qk, rk = rr["query"]["key"], rr["ref"]["key"]
            e1 = {"query": qk, "ref": rk, "query_date": str(rr["query"]["date"])[:10], "ref_date": str(rr["ref"]["date"])[:10],
                  "measure": PL.distance_measurements(qk, rk, z, region, CONDS), "img": {}}
            for cond in CONDS:
                e1["img"][cond] = {who: [png(out / "lab" / "eye" / region / z / f"{who}_{cond}_d{d:02d}.png",
                                             *percept_rgba(st.get(k, cond, d), dm)) for d in range(16)]
                                   for who, k in (("query", qk), ("ref", rk))}
            zz["exp1"][region] = e1
    print(f"  experiment 1 done ({time.time() - t0:.0f} s)", flush=True)
    # ---- experiment 2: real inference on every modification step
    for region in PL.AVAILABLE_REGIONS:
        obs = PL.Observers(region=region)
        for z, (_, rows) in picks.items():
            q, r = rows[region]["query"], rows[region]["ref"]
            rgb, m = PL.load_crop(q)
            bl = PL.body_length_px(int(q["ann_id"]))
            base = obs.identify(rgb, m, z)
            lab["checks"][f"{region}/{z}"] = obs.check_against_study(q["key"], base)
            ref_rgb, ref_m = PL.load_crop(r)
            e2 = {"query": q["key"], "ref": r["key"], "original": png(out / "lab" / "mod" / region / z / "original.png", rgb, m),
                  "reference": png(out / "lab" / "mod" / region / z / "reference.png", ref_rgb, ref_m), "steps": {}}
            strip = lambda d: {k: v for k, v in d.items() if not k.startswith("_")}
            for ctl in PL.CONTROLS:
                steps = []
                for i, v in enumerate(c[ctl]):
                    neutral = (ctl in ("thickness", "arrangement") and v == 0) or (ctl in ("contrast", "spacing") and v == 1)
                    mod = rgb if neutral else PL.modify(rgb, m, bl, ctl, float(v))
                    idn = base if neutral else obs.identify(mod, m, z)
                    steps.append({"value": v, "neutral": neutral,
                                  "img": e2["original"] if neutral else png(out / "lab" / "mod" / region / z / f"{ctl}_{i}.png", mod, m),
                                  "self": {"megadesc": 1.0, "hotspotter": None} if neutral else obs.self_similarity(idn, base),
                                  **{o: strip(idn[o]) for o in ("megadesc", "hotspotter")}})
                e2["steps"][ctl] = steps
            lab["zebra"][z]["exp2"][region] = e2
        print(f"  experiment 2 {region} done ({time.time() - t0:.0f} s)", flush=True)
    bad = {k: v for k, v in lab["checks"].items() if v["megadesc_cosine_to_stored"] < 0.999 or v["hotspotter_mismatches"]}
    print(f"CHECK Pattern Lab: unmodified photos reproduce the study's stored scores: {'yes' if not bad else bad}")
    (out / "lab.js").write_text("window.LAB = " + json.dumps(lab) + ";\n", encoding="utf-8")
    return lab


if __name__ == "__main__":
    build()
