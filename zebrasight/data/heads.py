"""Region rule v2: real faces (OWLv2 head detection), flank between shoulder and hip, rump without tail.

    python -m zebrasight.data.heads qa      # 24-zebra QA grid -> docs/examples/masks_v2/ (+ score table)
    python -m zebrasight.data.heads run     # full-run subset (+ env-demo zebras) -> data/masks/gzgc_v2/

Why: rule v1 took the face as the front 20% / upper part of the body mask, which is mostly NECK and
misses a lowered (grazing) head.

Face (region label 1):
1. OWLv2 (google/owlv2-base-patch16-ensemble, fp16 on the GPU) on the zebra's box crop + 10% padding,
   prompts "head of a zebra" and "zebra face". Keep the highest-scoring box with score >= SCORE_MIN
   that is head-sized (width <= 0.45 x body length, area <= 25% of the body box: OWLv2 often boxes
   the WHOLE zebra for these prompts) and whose centre lies in the front 35% of the body (image left
   for left-side views: the animal's left side faces the camera, so its head points to the viewer's
   left; verified on the QA grids). face = body mask AND head box. mask_method "owlv2".
2. Fallback (no valid detection, or the box misses the silhouette): the front-most extremity of the
   body silhouette along the body axis, 0.22 x body length, full column height (so a lowered head is
   included). mask_method "silhouette_tip".
3. Neither (no body mask, or face < MIN_FACE_FRAC of the body): no face row for that photo.

Flank (2) / rump (3) from the trunk: body mask minus the head box, thin parts (tail, lower legs)
removed by a morphological opening; trunk columns = columns whose mask height is >= 0.6 x the
tallest column (neck and head are thinner). Shoulder / hip = front / rear end of the trunk columns,
trunk length T. Flank = shoulder + 0.15 T .. hip - 0.25 T; rump = hip - 0.25 T .. hip; both within
the upper 65% of the trunk's vertical extent (legs dropped). Label 4 = other body.
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

from zebrasight.config import REPO_ROOT
from zebrasight.data.masks import head_on_left

MASK_DIR = REPO_ROOT / "data" / "masks" / "gzgc_v2"
INDEX = REPO_ROOT / "data" / "interim" / "masks_v2_index.parquet"
QA_DIR = REPO_ROOT / "docs" / "examples" / "masks_v2"
MODEL_ID = "google/owlv2-base-patch16-ensemble"
PROMPTS = ["head of a zebra", "zebra face"]
SCORE_MIN = 0.15  # tuned on the QA set (see docs/examples/masks_v2/README.md)
PAD = 0.10
TIP_FRAC = 0.22
MIN_FACE_FRAC = 0.02
MAX_HEAD_W = 0.45  # head box width <= 0.45 x body length (OWLv2 sometimes boxes the whole zebra)
MAX_HEAD_AREA = 0.25  # head box area <= 25% of the body's bounding box
FRONT_MAX = 0.35  # head box centre within the front 35% of the body length
FACE, FLANK, RUMP, OTHER = 1, 2, 3, 4


def load_owl(device):
    from transformers import Owlv2ForObjectDetection, Owlv2Processor

    proc = Owlv2Processor.from_pretrained(MODEL_ID)
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    model = Owlv2ForObjectDetection.from_pretrained(MODEL_ID, torch_dtype=dtype).to(device).eval()
    return proc, model, dtype


@torch.no_grad()
def detect_heads(proc, model, dtype, crop: Image.Image) -> list[tuple[float, list[float]]]:
    """[(score, [x0, y0, x1, y1] in crop px)] for all prompts, highest first."""
    inputs = proc(text=[PROMPTS], images=crop, return_tensors="pt").to(model.device)
    inputs["pixel_values"] = inputs["pixel_values"].to(dtype)
    out = model(**inputs)
    # OWLv2 pads the image to a square; boxes are relative to that square
    side = max(crop.size)
    res = proc.post_process_grounded_object_detection(out, threshold=0.01,
                                                      target_sizes=torch.tensor([[side, side]]))[0]
    dets = [(float(s), [float(v) for v in b]) for s, b in zip(res["scores"].float(), res["boxes"].float())]
    return sorted(dets, key=lambda d: -d[0])


def pick_head(dets, body: np.ndarray, head_left: bool, offset: tuple[int, int], score_min: float = SCORE_MIN):
    """Best detection at the front of the body. ``offset`` maps crop px -> body-mask px."""
    cols = np.nonzero(body.any(axis=0))[0]
    rows = np.nonzero(body.any(axis=1))[0]
    if cols.size == 0:
        return None
    x0b, x1b = cols[0], cols[-1] + 1
    L = x1b - x0b
    body_box_area = L * (rows[-1] + 1 - rows[0])
    for s, (x0, y0, x1, y1) in dets:
        if s < score_min:
            break
        bx0, by0, bx1, by1 = x0 - offset[0], y0 - offset[1], x1 - offset[0], y1 - offset[1]
        # OWLv2 often boxes the WHOLE zebra for "head of a zebra": reject body-sized boxes
        if (bx1 - bx0) > MAX_HEAD_W * L or (bx1 - bx0) * (by1 - by0) > MAX_HEAD_AREA * body_box_area:
            continue
        if bx1 <= x0b or bx0 >= x1b or by1 <= rows[0] or by0 >= rows[-1] + 1:
            continue  # the box must overlap the body
        cx = (bx0 + bx1) / 2
        front = (cx - x0b) / L if head_left else (x1b - cx) / L  # 0 at the front end of the body
        if front <= FRONT_MAX:
            return s, [bx0, by0, bx1, by1]
    return None


def _box_mask(shape, box) -> np.ndarray:
    m = np.zeros(shape, bool)
    h, w = shape
    # clamp BOTH ends into [0, size]: a negative end index would silently mean "from the far end"
    x0, x1 = (min(max(int(round(v)), 0), w) for v in (box[0], box[2]))
    y0, y1 = (min(max(int(round(v)), 0), h) for v in (box[1], box[3]))
    m[y0:y1, x0:x1] = True
    return m


def silhouette_tip(body: np.ndarray, head_left: bool) -> np.ndarray:
    cols = np.nonzero(body.any(axis=0))[0]
    x0, x1 = cols[0], cols[-1] + 1
    L = x1 - x0
    sel = np.zeros(body.shape[1], bool)
    if head_left:
        sel[x0 : x0 + max(1, int(round(TIP_FRAC * L)))] = True
    else:
        sel[x1 - max(1, int(round(TIP_FRAC * L))) : x1] = True
    return body & sel[None, :]


def _opening(m: np.ndarray, r: int) -> np.ndarray:
    if r < 1:
        return m
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # large masks: GPU is ~50x faster
    t = torch.from_numpy(m.astype(np.float32))[None, None].to(dev)
    k = 2 * r + 1

    def maxpool(x):  # square max filter as two 1-D passes (separable: identical result, ~k/2x cheaper)
        x = F.max_pool2d(x, (1, k), stride=1, padding=(0, r))
        return F.max_pool2d(x, (k, 1), stride=1, padding=(r, 0))

    eroded = -maxpool(-t)
    opened = maxpool(eroded)
    return opened[0, 0].cpu().numpy() > 0.5


def regions_v2(body: np.ndarray, head_left: bool, head_box) -> tuple[np.ndarray, str | None]:
    """Label map (0 bg, 1 face, 2 flank, 3 rump, 4 other) and the face method (None = no face)."""
    label = np.zeros(body.shape, np.uint8)
    if not body.any():
        return label, None
    label[body] = OTHER
    cols = np.nonzero(body.any(axis=0))[0]
    L = cols[-1] + 1 - cols[0]
    # face
    face, method = None, None
    if head_box is not None:
        face, method = body & _box_mask(body.shape, head_box), "owlv2"
        if face.sum() < MIN_FACE_FRAC * body.sum():  # box mostly off the silhouette: treat as no detection
            face, method, head_box = None, None, None
    if face is None:
        face, method = silhouette_tip(body, head_left), "silhouette_tip"
    if face.sum() < MIN_FACE_FRAC * body.sum():
        face, method = np.zeros_like(body), None
    # trunk: no head box, no thin parts (tail, lower legs), columns at least 0.6 x the tallest
    trunk = body & ~(_box_mask(body.shape, head_box) if head_box is not None else face)
    trunk = _opening(trunk, max(1, int(round(0.03 * L))))
    height = trunk.sum(axis=0)
    tcols = np.nonzero(height >= 0.6 * height.max())[0] if height.max() > 0 else np.array([], int)
    if tcols.size:
        shoulder, hip = (tcols[0], tcols[-1] + 1) if head_left else (tcols[-1] + 1, tcols[0])
        T = abs(hip - shoulder)
        rows = np.nonzero(trunk[:, tcols].any(axis=1))[0]
        top, bottom = rows[0], rows[-1] + 1
        band = np.zeros(body.shape[0], bool)
        band[top : top + int(round(0.65 * (bottom - top)))] = True
        xs = np.arange(body.shape[1])
        pos = (xs - shoulder) / T if head_left else (shoulder - xs) / T  # 0 at shoulder, 1 at hip
        flank_cols = (pos >= 0.15) & (pos < 0.75)
        rump_cols = (pos >= 0.75) & (pos <= 1.0)
        label[trunk & band[:, None] & flank_cols[None, :]] = FLANK
        label[trunk & band[:, None] & rump_cols[None, :]] = RUMP
    label[face] = FACE  # the face wins over anything else
    return label, method


def body_mask_for(row, v1_index: pd.DataFrame) -> tuple[np.ndarray | None, list[int], str]:
    """Body mask (label > 0) from the v1 label PNG, its frame, and the v1 mask method."""
    r = v1_index.loc[row.ann_id]
    label = np.asarray(Image.open(REPO_ROOT / r.mask_path))
    return label > 0, [int(v) for v in r.mask_bbox_xyxy], str(r.mask_method)


def process(anns: pd.DataFrame, v1_index: pd.DataFrame, images_dir: Path, device, score_min=SCORE_MIN,
            write=True, every=50) -> pd.DataFrame:
    proc, model, dtype = load_owl(device)
    MASK_DIR.mkdir(parents=True, exist_ok=True)
    recs = []
    t0 = time.perf_counter()
    for k, row in enumerate(anns.itertuples(), 1):
        body, frame, v1_method = body_mask_for(row, v1_index)
        head_left = head_on_left(row.viewpoint)
        rec = {"ann_id": int(row.ann_id), "mask_bbox_xyxy": frame, "body_method": v1_method,
               "head_score": None, "head_box": None}
        head_box = None
        if v1_method == "maskrcnn_rule":  # no silhouette for bbox fallbacks
            x0, y0, x1, y1 = frame
            pw, ph = int(PAD * (x1 - x0)), int(PAD * (y1 - y0))
            with Image.open(images_dir / row.file_name) as im:
                W, H = im.size
                cx0, cy0, cx1, cy1 = max(0, x0 - pw), max(0, y0 - ph), min(W, x1 + pw), min(H, y1 + ph)
                crop = im.convert("RGB").crop((cx0, cy0, cx1, cy1))
            dets = detect_heads(proc, model, dtype, crop)
            got = pick_head(dets, body, head_left, (x0 - cx0, y0 - cy0), score_min)
            rec["top_scores"] = [round(d[0], 3) for d in dets[:3]]
            label, face_method = regions_v2(body, head_left, got[1] if got else None)
            if got and face_method == "owlv2":
                rec["head_score"], rec["head_box"] = got[0], [round(v, 1) for v in got[1]]
        else:
            label = np.zeros(body.shape, np.uint8)
            face_method = None
            label[body] = OTHER
            v1 = np.asarray(Image.open(REPO_ROOT / v1_index.loc[row.ann_id].mask_path))
            label[v1 == FLANK], label[v1 == RUMP] = FLANK, RUMP  # bbox-rule flank/rump as before
        rec["face_method"] = face_method or "none"
        path = MASK_DIR / f"{row.ann_id}.png"
        if write:
            Image.fromarray(label).save(path)  # no optimize: several x faster, same content
        rec["mask_path"] = path.relative_to(REPO_ROOT).as_posix()
        rec["_label"] = None if write else label
        recs.append(rec)
        if k % every == 0 or k == len(anns):
            el = time.perf_counter() - t0
            print(f"{k:5d}/{len(anns)}  {el:5.0f} s  ~{el / k * (len(anns) - k):4.0f} s left", flush=True)
    return pd.DataFrame(recs)


def head_down_score(row, v1_index) -> float:
    """Share of the front 20% of the body silhouette lying in the lower half of the box (grazing)."""
    body, _, method = body_mask_for(row, v1_index)
    if method != "maskrcnn_rule" or not body.any():
        return 0.0
    cols = np.nonzero(body.any(axis=0))[0]
    L = cols[-1] + 1 - cols[0]
    front = body[:, cols[0] : cols[0] + max(1, int(0.2 * L))] if head_on_left(row.viewpoint) else \
        body[:, cols[-1] + 1 - max(1, int(0.2 * L)) : cols[-1] + 1]
    ys = np.nonzero(front)[0]
    return float((ys > body.shape[0] / 2).mean()) if ys.size else 0.0


def target_anns() -> pd.DataFrame:
    """Full-run subset photos + every env-demo photo (for the VR face textures)."""
    from zebrasight.data.gzgc import load_annotations

    ann = load_annotations().set_index("ann_id", drop=False)
    want = set(pd.read_parquet(REPO_ROOT / "data" / "regions.parquet").ann_id)
    demo = REPO_ROOT / "data" / "regions_demo5.parquet"
    if demo.exists():
        want |= set(pd.read_parquet(demo).ann_id)
    return ann.loc[sorted(want)]


def main() -> None:
    from zebrasight.data.gzgc import IMAGES
    from zebrasight.data.masks import INDEX as V1_INDEX

    cmd = sys.argv[1] if len(sys.argv) > 1 else "qa"
    v1 = pd.read_parquet(V1_INDEX).set_index("ann_id", drop=False)
    anns = target_anns()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if cmd == "qa":
        anns = anns[anns.ann_id.isin(v1.index[v1.mask_method == "maskrcnn_rule"])]
        hd = anns.apply(lambda r: head_down_score(r, v1), axis=1)
        grazing = anns[hd > 0.6].sample(min(12, int((hd > 0.6).sum())), random_state=0)
        rest = anns.drop(grazing.index).sample(24 - len(grazing), random_state=0)
        qa = pd.concat([grazing, rest])
        idx = process(qa, v1, IMAGES, device, score_min=float(sys.argv[2]) if len(sys.argv) > 2 else SCORE_MIN,
                      write=False)
        labels = dict(zip(idx.index, idx["_label"]))
        out = qa_grid(qa, idx.drop(columns="_label"), IMAGES, QA_DIR / "qa_grid_v2.png", labels)
        print(idx[["ann_id", "face_method", "head_score", "top_scores"]].to_string())
        print(f"grazing-heuristic photos: {len(grazing)}; QA grid: {out}")
    elif cmd == "run":
        idx = process(anns, v1, IMAGES, device).drop(columns="_label")
        INDEX.parent.mkdir(parents=True, exist_ok=True)
        idx.to_parquet(INDEX, index=False)
        print(idx.face_method.value_counts(normalize=True).round(3).to_dict())


def qa_grid(anns: pd.DataFrame, idx: pd.DataFrame, images_dir: Path, out: Path, labels=None) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colours = {FACE: (0.95, 0.35, 0.1), FLANK: (0.15, 0.5, 0.9), RUMP: (0.1, 0.7, 0.45), OTHER: (0.6, 0.6, 0.6)}
    n = len(anns)
    cols = 6
    fig, axes = plt.subplots(math.ceil(n / cols), cols, figsize=(4 * cols, 3.2 * math.ceil(n / cols)))
    for ax, row, rec in zip(axes.ravel(), anns.itertuples(), idx.itertuples()):
        img = np.asarray(Image.open(images_dir / row.file_name).convert("RGB"), dtype=np.float32) / 255
        x0, y0, x1, y1 = rec.mask_bbox_xyxy
        crop = img[y0:y1, x0:x1].copy()
        lab = labels[rec.Index] if labels is not None else np.asarray(Image.open(REPO_ROOT / rec.mask_path))
        h, w = min(crop.shape[0], lab.shape[0]), min(crop.shape[1], lab.shape[1])
        crop, lab = crop[:h, :w], lab[:h, :w]
        for code, col in colours.items():
            sel = lab == code
            a = 0.25 if code == OTHER else 0.5
            crop[sel] = (1 - a) * crop[sel] + a * np.array(col)
        ax.imshow(crop)
        if rec.head_box is not None:
            bx0, by0, bx1, by1 = rec.head_box
            ax.add_patch(plt.Rectangle((bx0, by0), bx1 - bx0, by1 - by0, fill=False, ec="yellow", lw=1.5))
        ax.set_title(f"{row.zebra_id} {rec.face_method}\nscore {rec.head_score if rec.head_score else '-'}"
                     f"  top {getattr(rec, 'top_scores', '')}", fontsize=8)
        ax.axis("off")
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    handles = [plt.Rectangle((0, 0), 1, 1, color=colours[c]) for c in (FACE, FLANK, RUMP, OTHER)]
    fig.legend(handles, ["face", "flank", "rump", "other body"], loc="lower center", ncol=4, fontsize=14, frameon=False)
    fig.suptitle("Region rule v2 QA (OWLv2 head + trunk rule); yellow = head box", fontsize=15)
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=70)
    plt.close(fig)
    return out


if __name__ == "__main__":
    main()
