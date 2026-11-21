"""Full-run subset: at most 4 left views per zebra (2 gallery + 2 query from different sessions).

    python -m zebrasight.data.subset

Per zebra (from the left-view subset with masks):
- quality q = log(box area) + log(sharpness), sharpness = variance of the Laplacian of the grey
  crop resized to 256 px wide (scale-normalised); bbox_rule views (no Mask R-CNN match) are used
  only when nothing else is available;
- gallery: the session on the zebra's earliest date with the most usable views (ties: higher
  summed q), best 2 views by q;
- query: likewise among sessions on later dates (always a different session), best 2 views.
Writes data/interim/quality.parquet (cache), data/regions_all.parquet (all left views) and
data/regions.parquet (the selected views; splits as in zebrasight.data.gzgc.assign_splits).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from PIL import Image, ImageFilter

from zebrasight.config import REPO_ROOT
from zebrasight.data.gzgc import IMAGES, load_annotations, left_subset
from zebrasight.data.masks import INDEX
from zebrasight.data.regions import OUT as REGIONS, build, check_splits

QUALITY = REPO_ROOT / "data" / "interim" / "quality.parquet"
REGIONS_ALL = REPO_ROOT / "data" / "regions_all.parquet"
PER_SPLIT = 2


def sharpness(path, box) -> float:
    with Image.open(path) as im:
        full_w = im.width
        im.draft("L", (im.width // 4, im.height // 4))  # fast JPEG decode at reduced size
        scale = im.size[0] / full_w
        x0, y0, x1, y1 = (v * scale for v in box)
        crop = im.convert("L").crop((int(x0), int(y0), max(int(x0) + 2, int(x1)), max(int(y0) + 2, int(y1))))
    w = 256
    crop = crop.resize((w, max(2, round(crop.height * w / max(crop.width, 1)))), Image.BILINEAR)
    lap = np.asarray(crop.filter(ImageFilter.Kernel((3, 3), (0, 1, 0, 1, -4, 1, 0, 1, 0), 1, 128)), np.float32)
    return float(lap.var())


def quality_table(sub: pd.DataFrame) -> pd.DataFrame:
    if QUALITY.exists():
        q = pd.read_parquet(QUALITY)
        if set(sub.ann_id) <= set(q.ann_id):
            return q
    rows = []
    for r in sub.itertuples():
        s = sharpness(IMAGES / r.file_name, (r.bbox_x0, r.bbox_y0, r.bbox_x1, r.bbox_y1))
        rows.append({"ann_id": r.ann_id, "sharpness": s, "box_area": r.bbox_w * r.bbox_h})
    q = pd.DataFrame(rows)
    q["quality"] = np.log(q.box_area) + np.log(q.sharpness.clip(lower=1e-3))
    QUALITY.parent.mkdir(parents=True, exist_ok=True)
    q.to_parquet(QUALITY, index=False)
    return q


def _best_session(views: pd.DataFrame) -> pd.DataFrame:
    usable = views.assign(_ok=views.mask_method == "maskrcnn_rule")
    stats = usable.groupby("session_id").agg(n_ok=("_ok", "sum"), n=("ann_id", "size"),
                                             q=("quality", lambda s: s.nlargest(PER_SPLIT).sum()))
    stats["n_ok"] = stats.n_ok.clip(upper=PER_SPLIT)
    stats["n"] = stats.n.clip(upper=PER_SPLIT)
    best = stats.sort_values(["n_ok", "n", "q"], ascending=False).index[0]
    s = usable[usable.session_id == best]
    return s.sort_values(["_ok", "quality"], ascending=False).head(PER_SPLIT).drop(columns="_ok")


def select(sub: pd.DataFrame) -> pd.DataFrame:
    picks = []
    for _, v in sub.groupby("zebra_id"):
        first = v.date.min()
        picks.append(_best_session(v[v.date == first]))
        picks.append(_best_session(v[v.date > first]))
    return pd.concat(picks)


def main() -> None:
    idx = pd.read_parquet(INDEX)
    sub = left_subset(load_annotations()).merge(idx[["ann_id", "mask_method"]], on="ann_id")
    sub = sub.merge(quality_table(sub), on="ann_id")
    chosen = select(sub)
    keep_cols = [c for c in chosen.columns if c not in ("mask_method", "sharpness", "box_area", "quality")]
    if not REGIONS_ALL.exists():
        build(left_subset(load_annotations()), idx, REGIONS_ALL)
    reg = build(chosen[keep_cols], idx, REGIONS)
    per = chosen.groupby("zebra_id").size()
    n_gal = chosen.groupby("zebra_id").date.transform("min") == chosen.date
    print(f"zebras {chosen.zebra_id.nunique()}, views {len(chosen)} (gallery {int(n_gal.sum())}, "
          f"query {int((~n_gal).sum())}); views per zebra {per.value_counts().sort_index().to_dict()}")
    print(f"bbox_rule views kept: {int((chosen.mask_method == 'bbox_rule').sum())}")
    print(reg.groupby(["region", "split"]).size().unstack().to_string())
    print(check_splits(reg))
    gs = chosen[n_gal].groupby("zebra_id").session_id.nunique()
    qs = chosen[~n_gal].groupby("zebra_id").session_id.nunique()
    print(f"sessions per zebra in gallery {gs.max()}, query {qs.max()} (must be 1 each)")


if __name__ == "__main__":
    main()
