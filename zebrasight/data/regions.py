"""Build data/regions.parquet: one row per (annotation image x region).

    python -m zebrasight.data.regions

Inputs: GZGC annotations (zebrasight.data.gzgc), the mask index and label PNGs
(zebrasight.data.masks). A region row exists when the region has >= MIN_REGION_PX pixels.

Columns: zebra_id, image_id, session_id, encounter_id, dataset, region, bbox_xyxy (region's
tight box, image px), mask_path (label PNG of the whole annotation box), mask_bbox_xyxy (the
PNG's frame in image px), mask_label (1 face / 2 flank / 3 rump), cm_per_px, cm_per_px_unc,
scale_method, mask_method, viewpoint, date, split (gallery / query), plus ann_id, image_path,
theta, mask_iou, region_px.

Load a region: crop the image at bbox_xyxy; mask = (label PNG == mask_label), cropped from
mask_bbox_xyxy to bbox_xyxy.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from PIL import Image

from zebrasight.config import REPO_ROOT
from zebrasight.data.gzgc import IMAGES, assign_splits, load_annotations, left_subset
from zebrasight.data.masks import INDEX, LABELS

OUT = REPO_ROOT / "data" / "regions.parquet"
MIN_REGION_PX = 400

COLUMNS = ["zebra_id", "image_id", "session_id", "encounter_id", "dataset", "region", "bbox_xyxy", "mask_path",
           "mask_bbox_xyxy", "mask_label", "cm_per_px", "cm_per_px_unc", "scale_method", "mask_method", "viewpoint",
           "date", "split", "ann_id", "image_path", "theta", "mask_iou", "region_px"]


def build(sub: pd.DataFrame | None = None, index: pd.DataFrame | None = None, out_path=OUT) -> pd.DataFrame:
    """Region rows for ``sub`` (default: the full left subset) using the mask ``index``."""
    if sub is None:
        sub = left_subset(load_annotations())
    sub = sub.copy()
    sub["split"] = assign_splits(sub)
    index = pd.read_parquet(INDEX) if index is None else index
    df = sub.merge(index, on="ann_id", how="inner", validate="one_to_one")
    rows = []
    for r in df.itertuples():
        label = np.asarray(Image.open(REPO_ROOT / r.mask_path))
        mx0, my0, _, _ = r.mask_bbox_xyxy
        for region, code in LABELS.items():
            sel = label == code
            n = int(sel.sum())
            if n < MIN_REGION_PX:
                continue
            ys, xs = np.nonzero(sel)
            rows.append({
                "zebra_id": r.zebra_id, "image_id": r.image_id, "session_id": r.session_id,
                "encounter_id": r.encounter_id, "dataset": r.dataset, "region": region,
                "bbox_xyxy": [int(mx0 + xs.min()), int(my0 + ys.min()), int(mx0 + xs.max() + 1), int(my0 + ys.max() + 1)],
                "mask_path": r.mask_path, "mask_bbox_xyxy": [int(v) for v in r.mask_bbox_xyxy], "mask_label": code,
                "cm_per_px": r.cm_per_px, "cm_per_px_unc": r.cm_per_px_unc, "scale_method": r.scale_method,
                "mask_method": r.mask_method, "viewpoint": r.viewpoint, "date": r.date, "split": r.split,
                "ann_id": r.ann_id, "image_path": (IMAGES / r.file_name).relative_to(REPO_ROOT).as_posix(),
                "theta": r.theta, "mask_iou": r.mask_iou, "region_px": n,
            })
    out = pd.DataFrame(rows, columns=COLUMNS)
    out.to_parquet(out_path, index=False)
    return out


def check_splits(df: pd.DataFrame) -> dict:
    enc = df.groupby("encounter_id")["split"].nunique()
    both = df.groupby("zebra_id")["split"].nunique()
    return {"encounters_in_both_splits": int((enc > 1).sum()), "zebras_with_gallery_and_query": int((both == 2).sum()),
            "zebras": int(df.zebra_id.nunique())}


if __name__ == "__main__":
    regions = build()
    print(regions.groupby(["region", "split"]).size().unstack().to_string())
    print(check_splits(regions))
    print(f"wrote {OUT} ({len(regions)} rows)")
