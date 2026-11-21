"""Rebuild region tables with region rule v2 (zebrasight.data.heads): real faces, trunk-based flank/rump.

    python -m zebrasight.data.regions_v2

- data/regions.parquet        <- full-run subset, rule v2 (the v1 table is kept as data/regions_v1.parquet)
- data/regions_demo5_v2.parquet <- env-demo zebras (all their pilot photos), rule v2, for the demo store

mask_method per region row: face = "owlv2" | "silhouette_tip" (photos with neither get no face row);
flank / rump = "maskrcnn_trunk" (rule v2 on the Mask R-CNN body mask) | "bbox_rule" (no body mask).
"""

from __future__ import annotations

import shutil

import pandas as pd

from zebrasight.config import REPO_ROOT
from zebrasight.data.gzgc import load_annotations
from zebrasight.data.heads import INDEX as V2_INDEX
from zebrasight.data.masks import INDEX as V1_INDEX
from zebrasight.data.regions import OUT as REGIONS, build, check_splits

V1_COPY = REPO_ROOT / "data" / "regions_v1.parquet"
DEMO_V2 = REPO_ROOT / "data" / "regions_demo5_v2.parquet"


def v2_index() -> pd.DataFrame:
    v2 = pd.read_parquet(V2_INDEX)
    v1 = pd.read_parquet(V1_INDEX)[["ann_id", "mask_iou", "mask_score", "mask_frac"]]
    idx = v2.merge(v1, on="ann_id", how="left")
    idx["mask_method"] = idx.body_method.map({"maskrcnn_rule": "maskrcnn_trunk"}).fillna("bbox_rule")
    return idx


def build_v2(ann_ids, out) -> pd.DataFrame:
    ann = load_annotations()
    sub = ann[ann.ann_id.isin(set(ann_ids))]
    idx = v2_index()
    reg = build(sub, idx[["ann_id", "mask_method", "mask_iou", "mask_score", "mask_frac", "mask_path", "mask_bbox_xyxy"]],
                out)
    face_method = dict(zip(idx.ann_id, idx.face_method))
    is_face = reg.region == "face"
    reg.loc[is_face, "mask_method"] = reg.loc[is_face, "ann_id"].map(face_method)
    reg = reg[~(is_face & (reg.mask_method == "none"))].reset_index(drop=True)
    reg["region_rule"] = "v2"
    reg.to_parquet(out, index=False)
    return reg


def main() -> None:
    if not V1_COPY.exists():
        shutil.copy2(REGIONS, V1_COPY)
    subset_ids = pd.read_parquet(V1_COPY).ann_id.unique()
    reg = build_v2(subset_ids, REGIONS)
    photos = len(subset_ids)
    idx = v2_index().set_index("ann_id").loc[subset_ids]
    share = idx.face_method.value_counts().reindex(["owlv2", "silhouette_tip", "none"], fill_value=0)
    print(f"subset photos {photos}: face from " + ", ".join(f"{k} {v} ({100 * v / photos:.1f}%)" for k, v in share.items()))
    faces = reg[reg.region == "face"]
    sess = faces.groupby("zebra_id").session_id.nunique()
    print(f"zebras with face crops in >= 2 sessions: {int((sess >= 2).sum())} of {reg.zebra_id.nunique()}")
    print(reg.groupby(["region", "split"]).size().unstack().to_string())
    print(check_splits(reg))
    demo = REPO_ROOT / "data" / "regions_demo5.parquet"
    if demo.exists():
        d = build_v2(pd.read_parquet(demo).ann_id.unique(), DEMO_V2)
        print(f"demo zebras v2: {len(d)} rows, faces {int((d.region == 'face').sum())}")


if __name__ == "__main__":
    main()
