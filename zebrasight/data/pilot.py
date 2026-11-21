"""Pilot subset while masks are still running: ~50 zebras whose masks are done.

    python -m zebrasight.data.pilot [n_zebras]

Uses the finished records in data/interim/masks_index.partial.jsonl. A zebra qualifies when its
finished left views span >= 2 dates; only finished views are used. Writes data/regions_pilot.parquet
(same columns as regions.parquet; splits recomputed on the pilot views).
"""

from __future__ import annotations

import json
import sys

import pandas as pd

from zebrasight.config import REPO_ROOT
from zebrasight.data.gzgc import load_annotations, left_subset
from zebrasight.data.masks import INDEX, box_in_image
from zebrasight.data.regions import build, check_splits

OUT = REPO_ROOT / "data" / "regions_pilot.parquet"


def main(n: int = 50, seed: int = 0) -> pd.DataFrame:
    partial = INDEX.with_suffix(".partial.jsonl")
    recs = [json.loads(x) for x in partial.read_text(encoding="utf-8").splitlines() if x.strip()]
    idx = pd.DataFrame(recs).drop_duplicates("ann_id", keep="last")
    sub = left_subset(load_annotations())
    # same trust rule as the mask run: made after the clipping fix, or box inside the image
    inside = {int(r.ann_id) for r in sub.itertuples() if box_in_image(r, r.img_w, r.img_h)}
    clipped = idx["clipped_v2"].fillna(False).astype(bool) if "clipped_v2" in idx else False
    idx = idx[clipped | idx.ann_id.isin(inside)]
    sub = sub[sub.ann_id.isin(idx.ann_id)]
    n_dates = sub.groupby("zebra_id").date.nunique()
    eligible = n_dates[n_dates >= 2].index.to_series()
    pick = eligible.sample(min(n, len(eligible)), random_state=seed)
    sub = sub[sub.zebra_id.isin(pick)]
    reg = build(sub, idx[["ann_id", "mask_method", "mask_iou", "mask_score", "mask_frac", "mask_path",
                          "mask_bbox_xyxy"]], OUT)
    print(f"eligible zebras {len(eligible)}, picked {len(pick)}; views {len(sub)}; region rows {len(reg)}")
    print(reg.groupby(["region", "split"]).size().unstack().to_string())
    print(check_splits(reg))
    return reg


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 50)
