"""Body masks and face / flank / rump regions for GZGC left-side views.

    python -m zebrasight.data.masks            # all subset annotations -> data/masks/gzgc/*.png
    python -m zebrasight.data.masks --qa       # QA grid only (docs/examples/masks/)

Mask: torchvision Mask R-CNN (ResNet-50 FPN v2, COCO weights; COCO has a "zebra" class).
The image is downscaled to a longer side of 1333 px (the model's own working size), zebra
detections are matched to each annotation box by IoU (>= 0.5), and the matched mask is
cropped to the annotation box and resized to it. mask_method = "maskrcnn_rule". If there
is no match, or the mask covers < 20% of the box, the whole box is used: "bbox_rule".

Region rule (grade C), using facing direction (viewpoint "left" -> head on the image left):
over the mask's horizontal extent, measured from the head end,
    face  = front 20%,  flank = 30-70%,  rump = rear 25% (75-100%),
all restricted to the upper 65% of the mask's height (legs = lower 35% dropped).
Labels in the saved PNG (one per annotation, cropped to its box):
    0 background, 1 face, 2 flank, 3 rump, 4 other body (neck, legs, gaps).
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from zebrasight.config import REPO_ROOT

MASK_DIR = REPO_ROOT / "data" / "masks" / "gzgc"
INDEX = REPO_ROOT / "data" / "interim" / "masks_index.parquet"
QA_DIR = REPO_ROOT / "docs" / "examples" / "masks"
LABELS = {"face": 1, "flank": 2, "rump": 3}
OTHER = 4
RULE = {"face": (0.0, 0.20), "flank": (0.30, 0.70), "rump": (0.75, 1.00), "body_band": 0.65}
WORK_SIDE = 1333
IOU_MIN = 0.5
MIN_MASK_FRAC = 0.20
SCORE_MIN = 0.3


def head_on_left(viewpoint: str) -> bool:
    """Animal's left side visible -> it faces the viewer's left."""
    return "left" in viewpoint


def rule_regions(mask: np.ndarray, head_left: bool) -> np.ndarray:
    """(H,W) bool body mask -> uint8 label map (see module docstring)."""
    label = np.zeros(mask.shape, dtype=np.uint8)
    if not mask.any():
        return label
    label[mask] = OTHER
    cols = np.nonzero(mask.any(axis=0))[0]
    rows = np.nonzero(mask.any(axis=1))[0]
    x0, x1 = cols[0], cols[-1] + 1
    y0, y1 = rows[0], rows[-1] + 1
    band = np.zeros_like(mask)
    band[y0 : y0 + int(round(RULE["body_band"] * (y1 - y0))), :] = True
    length = x1 - x0
    xs = np.arange(mask.shape[1])
    frac = (xs - x0) / length if head_left else (x1 - 1 - xs) / length  # 0 at the head end
    for name, code in LABELS.items():
        lo, hi = RULE[name]
        colsel = (frac >= lo) & (frac < hi) if hi < 1.0 else (frac >= lo) & (frac <= 1.0)
        label[mask & band & colsel[None, :]] = code
    return label


def box_iou(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    from torchvision.ops import box_iou as _iou

    return _iou(a, b)


def load_model(device: torch.device):
    from torchvision.models.detection import MaskRCNN_ResNet50_FPN_V2_Weights, maskrcnn_resnet50_fpn_v2

    weights = MaskRCNN_ResNet50_FPN_V2_Weights.COCO_V1
    model = maskrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=SCORE_MIN)
    model.roi_heads.detections_per_img = 30
    zebra = weights.meta["categories"].index("zebra")
    return model.eval().to(device), zebra


def read_image(path: Path, device: torch.device) -> torch.Tensor:
    from torchvision.io import ImageReadMode, decode_jpeg, read_file

    data = read_file(str(path))
    try:
        return decode_jpeg(data, mode=ImageReadMode.RGB, device=device)
    except RuntimeError:  # fall back to CPU decode
        return decode_jpeg(data, mode=ImageReadMode.RGB).to(device)


@torch.no_grad()
def clip_box(x0: float, y0: float, x1: float, y1: float, W: int, H: int) -> tuple[int, int, int, int]:
    """Integer box (floor/ceil) clipped to the image. GZGC boxes can extend past the frame
    when the animal is cut off by the image edge."""
    bx0, by0 = max(0, int(math.floor(x0))), max(0, int(math.floor(y0)))
    bx1, by1 = min(W, int(math.ceil(x1))), min(H, int(math.ceil(y1)))
    return bx0, by0, max(bx0 + 1, bx1), max(by0 + 1, by1)


def box_in_image(r, W: int, H: int) -> bool:
    return r.bbox_x0 >= 0 and r.bbox_y0 >= 0 and r.bbox_x1 <= W and r.bbox_y1 <= H


def masks_for_image(model, zebra_label: int, img_u8: torch.Tensor, ann_boxes: np.ndarray):
    """Return, per annotation box (integer xyxy, clipped to the image): (mask HxW bool np or None, iou, score)."""
    _, H, W = img_u8.shape
    s = min(1.0, WORK_SIDE / max(H, W))
    img = img_u8.float().div(255.0)
    if s < 1.0:
        img = F.interpolate(img[None], scale_factor=s, mode="bilinear", antialias=True, align_corners=False)[0]
    out = model([img])[0]
    keep = out["labels"] == zebra_label
    dboxes, dmasks, dscores = out["boxes"][keep], out["masks"][keep], out["scores"][keep]
    results = []
    abox = torch.tensor(np.array(ann_boxes, dtype=np.float32), device=img.device) * s
    ious = box_iou(abox, dboxes) if len(dboxes) else torch.zeros(len(abox), 0, device=img.device)
    mh, mw = img.shape[-2:]
    for i, (x0, y0, x1, y1) in enumerate(ann_boxes):
        x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
        bw, bh = x1 - x0, y1 - y0
        if ious.shape[1] == 0 or float(ious[i].max()) < IOU_MIN:
            results.append((None, float(ious[i].max()) if ious.shape[1] else 0.0, 0.0))
            continue
        j = int(ious[i].argmax())
        sx0, sy0 = min(mw - 1, math.floor(x0 * s)), min(mh - 1, math.floor(y0 * s))
        sx1, sy1 = min(mw, max(sx0 + 1, math.ceil(x1 * s))), min(mh, max(sy0 + 1, math.ceil(y1 * s)))
        crop = dmasks[j, :, sy0:sy1, sx0:sx1]
        crop = F.interpolate(crop[None], size=(bh, bw), mode="bilinear", align_corners=False)[0, 0]
        m = (crop > 0.5).cpu().numpy()
        results.append((m, float(ious[i, j]), float(dscores[j])))
    return results


def run(sub: pd.DataFrame, images_dir: Path, device: torch.device | None = None, every: int = 50) -> pd.DataFrame:
    from PIL import Image

    import json

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MASK_DIR.mkdir(parents=True, exist_ok=True)
    INDEX.parent.mkdir(parents=True, exist_ok=True)
    partial = INDEX.with_suffix(".partial.jsonl")  # one line per finished annotation: resumable
    records = []
    if partial.exists():
        records = [json.loads(line) for line in partial.read_text(encoding="utf-8").splitlines() if line.strip()]
    # records made before boxes were clipped are only trusted when the box was inside the image
    inside = {int(r.ann_id) for r in sub.itertuples() if box_in_image(r, r.img_w, r.img_h)}
    records = [r for r in records if r["ann_id"] in inside or r.get("clipped_v2")]
    done = {r["ann_id"] for r in records}
    groups = [(f, g) for f, g in sub.groupby("file_name") if not set(g.ann_id) <= done]
    print(f"resuming: {len(done)} annotations already done, {len(groups)} images to go", flush=True)
    model, zebra = load_model(device)
    t0 = time.perf_counter()
    log = partial.open("a", encoding="utf-8")
    for k, (fname, g) in enumerate(groups, 1):
        img = read_image(images_dir / fname, device)
        H, W = img.shape[-2:]
        boxes = np.array([clip_box(r.bbox_x0, r.bbox_y0, r.bbox_x1, r.bbox_y1, W, H) for r in g.itertuples()])
        res = masks_for_image(model, zebra, img, boxes)
        for (_, r), box, (m, iou, score) in zip(g.iterrows(), boxes, res):
            bx0, by0, bx1, by1 = (int(v) for v in box)
            shape = (by1 - by0, bx1 - bx0)
            method = "maskrcnn_rule"
            if m is None or m.mean() < MIN_MASK_FRAC:
                m, method = np.ones(shape, dtype=bool), "bbox_rule"
            label = rule_regions(m, head_on_left(r.viewpoint))
            path = MASK_DIR / f"{r.ann_id}.png"
            Image.fromarray(label).save(path, optimize=True)
            rec = {"ann_id": int(r.ann_id), "mask_method": method, "mask_iou": iou, "mask_score": score,
                   "mask_frac": float(m.mean()), "mask_path": path.relative_to(REPO_ROOT).as_posix(),
                   "mask_bbox_xyxy": [bx0, by0, bx1, by1], "clipped_v2": True}
            records.append(rec)
            log.write(json.dumps(rec) + "\n")
        log.flush()
        if k % every == 0 or k == len(groups):
            el = time.perf_counter() - t0
            print(f"{k:5d}/{len(groups)} images  {el:5.0f} s  ~{el / k * (len(groups) - k):4.0f} s left", flush=True)
    log.close()
    idx = pd.DataFrame(records).drop_duplicates("ann_id", keep="last")
    idx.to_parquet(INDEX, index=False)
    return idx


def qa_grid(sub: pd.DataFrame, idx: pd.DataFrame, images_dir: Path, n: int = 12, seed: int = 0) -> Path:
    """12 zebras with regions drawn; watermarked internal-only (derived from CC BY-NC-ND images)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    df = sub.merge(idx, on="ann_id")
    fb = df[df.mask_method == "bbox_rule"]
    pick = pd.concat([df[df.mask_method == "maskrcnn_rule"].sample(n - min(2, len(fb)), random_state=seed),
                      fb.sample(min(2, len(fb)), random_state=seed)])
    colours = {1: (0.95, 0.35, 0.1), 2: (0.15, 0.5, 0.9), 3: (0.1, 0.7, 0.45), 4: (0.6, 0.6, 0.6)}
    fig, axes = plt.subplots(3, 4, figsize=(20, 12))
    for ax, (_, r) in zip(axes.ravel(), pick.iterrows()):
        img = np.asarray(Image.open(images_dir / r.file_name).convert("RGB"), dtype=np.float32) / 255.0
        x0, y0, x1, y1 = r.mask_bbox_xyxy
        crop = img[y0:y1, x0:x1].copy()
        label = np.asarray(Image.open(REPO_ROOT / r.mask_path))
        h = min(crop.shape[0], label.shape[0])
        w = min(crop.shape[1], label.shape[1])
        crop, label = crop[:h, :w], label[:h, :w]
        for code, col in colours.items():
            sel = label == code
            a = 0.45 if code != OTHER else 0.25
            crop[sel] = (1 - a) * crop[sel] + a * np.array(col)
        ax.imshow(crop)
        ax.set_title(f"{r.zebra_id}  {r.viewpoint}  {r.mask_method}\nIoU {r.mask_iou:.2f}  mask {r.mask_frac:.0%} of box",
                     fontsize=10)
        ax.axis("off")
    handles = [plt.Rectangle((0, 0), 1, 1, color=colours[c]) for c in (1, 2, 3, 4)]
    fig.legend(handles, ["face", "flank", "rump", "other body"], loc="lower center", ncol=4, fontsize=14, frameon=False)
    fig.suptitle("Region QA - INTERNAL, NOT FOR PUBLIC DISPLAY (derived from GZGC, CC BY-NC-ND 2.0)",
                 fontsize=16, color="#b00020")
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    QA_DIR.mkdir(parents=True, exist_ok=True)
    out = QA_DIR / "qa_grid.png"
    fig.savefig(out, dpi=80)
    plt.close(fig)
    (QA_DIR / "README.md").write_text(
        "# Mask QA (internal)\n\n`qa_grid.png` is derived from GZGC images (CC BY-NC-ND 2.0).\n"
        "**Internal - not for public display** until the licence for derived images is confirmed.\n\n"
        "Regenerate: `python -m zebrasight.data.masks --qa`.\n", encoding="utf-8")
    return out


def main() -> None:
    from zebrasight.data.gzgc import IMAGES, load_annotations, left_subset

    sub = left_subset(load_annotations())
    if "--qa" in sys.argv:
        idx = pd.read_parquet(INDEX)
    else:
        print(f"{len(sub)} annotations in {sub.file_name.nunique()} images", flush=True)
        idx = run(sub, IMAGES)
        print(idx.mask_method.value_counts().to_string(), flush=True)
    print("QA grid:", qa_grid(sub, idx, IMAGES), flush=True)


if __name__ == "__main__":
    main()
