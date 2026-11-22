"""Part A pre-flight for the positional study (configs/positional.yaml).

    python -m zebrasight.positional.preflight quality    # A1: body-quality report + body masks + region table
    python -m zebrasight.positional.preflight percepts   # A2: missing c0 percepts for body + rump (resumable)
    python -m zebrasight.positional.preflight heldout    # A3: freeze the held-out zebra set

A1 reads the v2 label PNGs (label > 0 = Mask R-CNN body) and the COCO annotations. A photo's full body
is usable when its annotation box does not touch the frame, its mask fills enough of the box, no other
zebra's box covers much of it, and it has a real mask (not the bbox_rule fallback). Bad photos stay in
for flank and rump. It writes body-only PNGs (1 = body) and data/regions_50_positional.parquet with
body rows (good photos only) plus the existing flank and rump rows; no code under data/ changes.

A2 calls the existing percept store (zebrasight.pipeline.store.run) on that table, regions body and rump,
condition c0: rows already complete are skipped, so it resumes after an interruption.

A3 freezes the held-out set: zebras of data/regions.parquet outside the 50, left side, with >= 1
gallery and >= 1 query photo, at most 4 photos each (seeded). Nothing is computed on them.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from zebrasight.config import REPO_ROOT, load_config

COCO = REPO_ROOT / "data" / "raw" / "gzgc" / "gzgc.coco" / "annotations" / "instances_train2020.json"


def cfg():
    return load_config("positional")


def _coco():
    d = json.loads(COCO.read_text(encoding="utf-8"))
    cats = {c["id"]: c["name"] for c in d["categories"]}
    size = {i["id"]: (i["width"], i["height"]) for i in d["images"]}
    anns = {a["id"]: a for a in d["annotations"]}
    by_img: dict[int, list] = {}
    for a in d["annotations"]:
        if "zebra" in cats[a["category_id"]]:
            by_img.setdefault(a["image_id"], []).append(a)
    return anns, size, by_img


def body_quality(regions: pd.DataFrame, q) -> pd.DataFrame:
    """One row per photo (ann_id): edge distance, mask fill, other-zebra cover, verdict and reasons."""
    anns, size, by_img = _coco()
    out = []
    for r in regions.drop_duplicates("ann_id").itertuples(index=False):
        a = anns[int(r.ann_id)]
        W, H = size[a["image_id"]]
        bx, by, bw, bh = a["bbox"]
        x0, y0, x1, y1 = bx, by, bx + bw, by + bh
        edge = min(x0, y0, W - x1, H - y1)
        body = np.asarray(Image.open(REPO_ROOT / r.mask_path)) > 0
        fill = float(body.mean())
        cover = 0.0
        for b in by_img.get(a["image_id"], []):
            if b["id"] == a["id"]:
                continue
            ox, oy, ow, oh = b["bbox"]
            ix = max(0.0, min(x1, ox + ow) - max(x0, ox))
            iy = max(0.0, min(y1, oy + oh) - max(y0, oy))
            cover = max(cover, ix * iy / (bw * bh))
        reasons = []
        if edge <= q.edge_margin_px:
            reasons.append("cut_by_frame")
        if q.require_mask and r.mask_method == "bbox_rule":
            reasons.append("no_body_mask")
        elif fill < q.min_mask_fill:
            reasons.append("low_mask_fill")
        if cover > q.max_other_zebra_cover:
            reasons.append("overlaps_other_zebra")
        out.append({"ann_id": int(r.ann_id), "zebra_id": r.zebra_id, "split": r.split, "image_id": r.image_id,
                    "edge_px": round(float(edge), 1), "mask_fill": round(fill, 3), "other_zebra_cover": round(cover, 3),
                    "body_ok": not reasons, "reasons": ";".join(reasons)})
    return pd.DataFrame(out)


def body_rows(regions: pd.DataFrame, quality: pd.DataFrame, mask_dir: Path) -> pd.DataFrame:
    """Body region rows (good photos only), with body-only PNGs written to mask_dir (1 = body)."""
    mask_dir.mkdir(parents=True, exist_ok=True)
    ok = set(quality.ann_id[quality.body_ok])
    src = regions[regions.region == "flank"].drop_duplicates("ann_id")  # one template row per photo
    rows = []
    for r in src.itertuples(index=False):
        if int(r.ann_id) not in ok:
            continue
        lab = np.asarray(Image.open(REPO_ROOT / r.mask_path))
        body = (lab > 0).astype(np.uint8)
        p = mask_dir / f"{int(r.ann_id)}.png"
        Image.fromarray(body).save(p)
        ys, xs = np.nonzero(body)
        mx0, my0 = int(r.mask_bbox_xyxy[0]), int(r.mask_bbox_xyxy[1])
        row = r._asdict()
        row.update(region="body", mask_path=p.relative_to(REPO_ROOT).as_posix(), mask_label=1,
                   bbox_xyxy=[mx0 + int(xs.min()), my0 + int(ys.min()), mx0 + int(xs.max()) + 1, my0 + int(ys.max()) + 1],
                   mask_method="maskrcnn_body", region_px=int(body.sum()))
        rows.append(row)
    return pd.DataFrame(rows)


def quality_cmd() -> None:
    c = cfg()
    reg = pd.read_parquet(REPO_ROOT / c.data.regions_50)
    q = body_quality(reg, c.body_quality)
    out_dir = REPO_ROOT / c.data.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    q.to_csv(out_dir / "a1_body_quality.csv", index=False)
    body = body_rows(reg, q, REPO_ROOT / c.data.body_mask_dir)
    pos = pd.concat([body, reg[reg.region.isin(["flank", "rump"])]], ignore_index=True)
    pos = pos[reg.columns.tolist()]
    pos.to_parquet(REPO_ROOT / c.data.out_regions, index=False)
    print(f"photos: {len(q)}  body_ok: {int(q.body_ok.sum())}  flagged: {int((~q.body_ok).sum())}")
    for reason in ("cut_by_frame", "no_body_mask", "low_mask_fill", "overlaps_other_zebra"):
        print(f"  {reason}: {int(q.reasons.str.contains(reason).sum())}")
    counts = pos.groupby(["region", "split"]).agg(photos=("ann_id", "size"), zebras=("zebra_id", "nunique"))
    print(counts.to_string())
    both = pos.groupby(["region", "zebra_id"]).split.nunique().eq(2).groupby("region").sum()
    print("zebras with both gallery and query photos, per region:", both.to_dict())
    print(f"wrote {c.data.out_regions}, {c.data.body_mask_dir}/, {c.data.out_dir}/a1_body_quality.csv")


def percepts_cmd() -> None:
    from zebrasight.pipeline import store

    c = cfg()
    store.run(REPO_ROOT / c.data.out_regions, REPO_ROOT / c.data.store, conds=[c.data.condition],
              regions=("body", "rump"), low_priority=bool(c.run.low_priority))


def heldout(full: pd.DataFrame, fifty: set, h) -> pd.DataFrame:
    rng = np.random.default_rng(int(h.seed))
    f = full[(full.region == "flank") & (full.viewpoint == h.viewpoint) & ~full.zebra_id.isin(fifty)]
    keep = []
    for z, g in f.groupby("zebra_id", sort=True):
        gal, qry = g[g.split == "gallery"], g[g.split == "query"]
        if len(gal) < h.min_gallery or len(qry) < h.min_query:
            continue
        if len(g) > h.max_photos:  # keep >= 1 of each split, fill the rest at random
            first = pd.concat([gal.sample(1, random_state=rng.integers(1 << 31)),
                               qry.sample(1, random_state=rng.integers(1 << 31))])
            rest = g.drop(first.index).sample(int(h.max_photos) - 2, random_state=rng.integers(1 << 31))
            g = pd.concat([first, rest])
        keep.append(g)
    out = pd.concat(keep)[["zebra_id", "ann_id", "image_id", "split", "date", "session_id", "image_path"]]
    return out.sort_values(["zebra_id", "split", "ann_id"]).reset_index(drop=True)


def heldout_cmd() -> None:
    c = cfg()
    path = REPO_ROOT / c.data.heldout_csv
    if path.exists():
        raise SystemExit(f"{path} already exists: the held-out set is frozen; delete it only by a dated amendment")
    full = pd.read_parquet(REPO_ROOT / c.data.regions_full)
    fifty = set(pd.read_parquet(REPO_ROOT / c.data.regions_50).zebra_id)
    h = heldout(full, fifty, c.heldout)
    h.to_csv(path, index=False)
    print(f"froze {path}: {h.zebra_id.nunique()} zebras, {len(h)} photos "
          f"({(h.split == 'gallery').sum()} gallery, {(h.split == 'query').sum()} query)")


def main() -> None:
    ap = argparse.ArgumentParser(description="Positional study, Part A pre-flight")
    ap.add_argument("cmd", choices=["quality", "percepts", "heldout"])
    a = ap.parse_args()
    {"quality": quality_cmd, "percepts": percepts_cmd, "heldout": heldout_cmd}[a.cmd]()


if __name__ == "__main__":
    main()
