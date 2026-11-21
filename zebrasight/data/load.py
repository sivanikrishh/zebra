"""Load one region row of data/regions.parquet as pipeline input."""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image

from zebrasight.config import REPO_ROOT


def load_region(row) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """-> img (1,3,h,w) sRGB in [0,1], mask (1,1,h,w) {0,1}, meta (IDs only).

    The image is cropped at the region's bbox_xyxy; the mask is (label PNG == mask_label)
    cropped from the PNG frame (mask_bbox_xyxy) to the same box.
    """
    x0, y0, x1, y1 = (int(v) for v in row["bbox_xyxy"])
    with Image.open(REPO_ROOT / row["image_path"]) as im:
        crop = np.asarray(im.convert("RGB").crop((x0, y0, x1, y1)), dtype=np.float32) / 255.0
    label = np.asarray(Image.open(REPO_ROOT / row["mask_path"]))
    mx0, my0 = int(row["mask_bbox_xyxy"][0]), int(row["mask_bbox_xyxy"][1])
    lab = label[y0 - my0 : y1 - my0, x0 - mx0 : x1 - mx0]
    mask = (lab == int(row["mask_label"])).astype(np.float32)
    h, w = min(crop.shape[0], mask.shape[0]), min(crop.shape[1], mask.shape[1])
    img = torch.from_numpy(np.ascontiguousarray(crop[:h, :w])).permute(2, 0, 1)[None]
    m = torch.from_numpy(np.ascontiguousarray(mask[:h, :w]))[None, None]
    meta = {k: row[k] for k in ("zebra_id", "image_id", "session_id", "encounter_id", "region")}
    return img, m, meta
