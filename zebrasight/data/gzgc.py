"""GZGC (Wild Me GZGC-IBEIS, COCO format) loader: one row per plains-zebra annotation.

IDs
- zebra_id   = annotation ``name`` (e.g. IBEIS_PZ_1561): the individual.
- image_id   = "gzgc_<COCO image id>".
- session_id = "<date>_<car>_<person>" parsed from the image ``photographer`` field
  ("NNP GZC Car '10WHITE', Person 'A', Image 0005"); "<date>_unknown" when it is empty.
  A session is one camera team on one day.
- encounter_id = "<zebra_id>@<session_id>": all views of one zebra by one team on one day.

Scale (grade C): cm_per_px = BODY_LENGTH_CM / bbox width, for left/right side views only,
uncertainty 0.20 (head-down grazing poses change the box width).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

from zebrasight.config import REPO_ROOT

DATASET = "gzgc"
ROOT = REPO_ROOT / "data" / "raw" / "gzgc" / "gzgc.coco"
ANNOTATIONS = ROOT / "annotations" / "instances_train2020.json"
IMAGES = ROOT / "images" / "train2020"
LICENCE = "CC BY-NC-ND 2.0 (Wild Me GZGC-IBEIS; organiser-provided)"

# Plains zebra head-body length 217-246 cm; midpoint used. Range as reported by GBIF and
# Wikipedia, attributed to Grubb (1981) Equus burchelli, Mammalian Species 157:1-9.
# Primary text not seen; DOI UNVERIFIED (10.2307/3504012 and 10.2307/3503962 both cited).
BODY_LENGTH_CM = (217.0 + 246.0) / 2.0
BODY_LENGTH_RANGE_CM = (217.0, 246.0)
CM_PER_PX_UNC = 0.20
SCALE_METHOD = "body_length_bbox_width"

_CAR = re.compile(r"Car '([^']+)'")
_PERSON = re.compile(r"Person '([^']+)'")


def _session(photographer: str, date: str) -> str:
    car = _CAR.search(photographer or "")
    person = _PERSON.search(photographer or "")
    if car and person:
        return f"{date}_{car.group(1)}_{person.group(1)}"
    return f"{date}_unknown"


def load_annotations(path: Path = ANNOTATIONS) -> pd.DataFrame:
    """All plains-zebra annotations with IDs, viewpoint, bbox, date and scale."""
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    cats = {c["id"]: c["name"] for c in d["categories"]}
    imgs = {i["id"]: i for i in d["images"]}
    rows = []
    for a in d["annotations"]:
        if cats[a["category_id"]] != "zebra_plains":
            continue
        im = imgs[a["image_id"]]
        date = (im.get("date_captured") or "NA")[:10]
        x, y, w, h = a["bbox"]
        side = a["viewpoint"] in ("left", "right")
        rows.append({
            "ann_id": a["id"],
            "ann_uuid": a["uuid"],
            "zebra_id": a["name"],
            "image_id": f"{DATASET}_{a['image_id']}",
            "coco_image_id": a["image_id"],
            "file_name": im["file_name"],
            "img_w": im["width"],
            "img_h": im["height"],
            "session_id": _session(im.get("photographer", ""), date),
            "date": date,
            "viewpoint": a["viewpoint"],
            "theta": float(a.get("theta", 0.0)),
            "bbox_x0": x, "bbox_y0": y, "bbox_x1": x + w, "bbox_y1": y + h,
            "bbox_w": w, "bbox_h": h,
            "cm_per_px": BODY_LENGTH_CM / w if side else float("nan"),
            "cm_per_px_unc": CM_PER_PX_UNC if side else float("nan"),
            "scale_method": SCALE_METHOD if side else "none",
            "gps_lat": float(im.get("gps_lat_captured") or "nan"),
            "gps_lon": float(im.get("gps_lon_captured") or "nan"),
        })
    df = pd.DataFrame(rows)
    df["encounter_id"] = df["zebra_id"] + "@" + df["session_id"]
    df["dataset"] = DATASET
    return df


def left_subset(df: pd.DataFrame) -> pd.DataFrame:
    """All LEFT views of every zebra that has >= 2 left views on different (known) dates."""
    left = df[(df["viewpoint"] == "left") & (df["date"] != "NA")]
    n_dates = left.groupby("zebra_id")["date"].nunique()
    keep = n_dates[n_dates >= 2].index
    return left[left["zebra_id"].isin(keep)].copy()


def assign_splits(sub: pd.DataFrame) -> pd.Series:
    """Gallery = the zebra's encounters on its earliest date; query = encounters on later dates.

    Splitting by date (sessions grouped by day) means no encounter, and no session of that
    zebra, is ever in both gallery and query.
    """
    first = sub.groupby("zebra_id")["date"].transform("min")
    return pd.Series(["gallery" if d == f else "query" for d, f in zip(sub["date"], first)], index=sub.index)
