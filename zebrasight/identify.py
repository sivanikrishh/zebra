"""Who is this zebra? Zero-shot individual identification from flank stripes (no training).

    python -m zebrasight.identify build      [--regions data/regions.parquet] [--region flank]
    python -m zebrasight.identify evaluate   [--region flank]
    python -m zebrasight.identify query IMAGE [--facing left|right] [--box X0 Y0 X1 Y1] [--top 5] [--gallery gallery|all]

build     embeds every photo region of the chosen kind (MegaDescriptor-L-384, the same preprocessing as
          zebrasight.metrics.megadesc: region mask, outside grey, square pad, 384 px) and saves
          outputs/identify/embeddings_<region>.parquet (IDs, split, date + 1536 embedding columns).
evaluate  closed-set identification: every "query" photo (later dates) is ranked against the known
          zebras from the "gallery" photos (earliest date). Reports top-1 / top-5 accuracy with 95%
          bootstrap CIs over zebras, and the chance level (1 / number of known zebras).
query     a new photo: Mask R-CNN finds the zebra (the largest one, or the one best matching --box),
          the v2 trunk rule cuts the flank (facing direction from --facing; GZGC side views are "left"),
          MegaDescriptor embeds it, and the closest known zebras are printed with their similarity.

Similarity = cosine of L2-normalised embeddings; a zebra's score = its best-matching gallery photo.
This is CLOSED-SET: it always names the nearest known zebra, even for an animal it has never seen.
CAVEAT: MegaDescriptor's training data includes GiraffeZebraID (= GZGC), so accuracy on GZGC photos
can be optimistic; a credible number needs a different zebra population or a model trained only on
our own training split.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from zebrasight.config import REPO_ROOT

OUT_DIR = REPO_ROOT / "outputs" / "identify"
REGIONS = REPO_ROOT / "data" / "regions.parquet"
META = ["key", "zebra_id", "image_id", "ann_id", "split", "date", "region"]


# ---- ranking (pure numpy, unit-tested) ---------------------------------------------------------------

def emb_matrix(df: pd.DataFrame) -> np.ndarray:
    cols = [c for c in df.columns if c.startswith("E_")]
    e = df[cols].to_numpy(np.float32)
    return e / np.linalg.norm(e, axis=1, keepdims=True)


def rank_zebras(q: np.ndarray, g: np.ndarray, g_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """q (n,d), g (m,d) unit vectors; g_ids (m,) zebra id per gallery photo.

    Returns (ids (u,), scores (n,u)): the u distinct zebra ids and, per query, each zebra's score =
    max cosine over that zebra's gallery photos. Sort a row of scores descending to rank zebras."""
    ids, inv = np.unique(g_ids, return_inverse=True)
    s = q @ g.T
    out = np.full((q.shape[0], len(ids)), -np.inf, np.float32)
    for j in range(len(ids)):
        out[:, j] = s[:, inv == j].max(axis=1)
    return ids, out


def topk_hits(scores: np.ndarray, ids: np.ndarray, truth: np.ndarray, k: int) -> np.ndarray:
    """(n,) bool: is the true zebra among the k best-scoring zebras?"""
    top = np.argsort(-scores, axis=1)[:, :k]
    return (ids[top] == truth[:, None]).any(axis=1)


def bootstrap_ci(hits: np.ndarray, groups: np.ndarray, n_boot: int = 1000, seed: int = 0) -> tuple[float, float]:
    """95% CI of the mean of hits, resampling groups (zebras) with replacement."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    per = [hits[groups == u] for u in uniq]
    vals = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        vals.append(np.concatenate([per[i] for i in pick]).mean())
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


# ---- build / evaluate ------------------------------------------------------------------------------

def emb_path(region: str) -> Path:
    return OUT_DIR / f"embeddings_{region}.parquet"


def build(regions_path: Path, region: str) -> Path:
    import torch

    from zebrasight.data.load import load_region
    from zebrasight.metrics import megadesc

    reg = pd.read_parquet(regions_path)
    reg = reg[reg.region == region].reset_index(drop=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = megadesc.load_model(device)

    def items():
        for _, r in reg.iterrows():
            img, mask, _ = load_region(r)
            rgb = img[0].permute(1, 2, 0).numpy()
            m = mask[0, 0].numpy() > 0.5
            meta = {"key": f"{r.ann_id}_{r.region}", "zebra_id": r.zebra_id, "image_id": r.image_id,
                    "ann_id": int(r.ann_id), "split": r.split, "date": str(r.date), "region": r.region}
            yield meta, (megadesc.to_input(rgb, m) if m.any() else None)

    df = megadesc.run(items(), model, device)
    df = df.dropna(subset=["E_0000"]).reset_index(drop=True)
    out = emb_path(region)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"wrote {out}: {len(df)} photos of {df.zebra_id.nunique()} zebras")
    return out


def evaluate(region: str) -> dict:
    df = pd.read_parquet(emb_path(region))
    gal, qry = df[df.split == "gallery"], df[df.split == "query"]
    qry = qry[qry.zebra_id.isin(gal.zebra_id)]  # closed set: only zebras the system knows
    ids, scores = rank_zebras(emb_matrix(qry), emb_matrix(gal), gal.zebra_id.to_numpy())
    truth = qry.zebra_id.to_numpy()
    res = {"region": region, "known_zebras": int(len(ids)), "gallery_photos": int(len(gal)),
           "query_photos": int(len(qry)), "query_zebras": int(qry.zebra_id.nunique()),
           "chance_top1": 1 / len(ids), "chance_top5": min(1.0, 5 / len(ids))}
    for k in (1, 5):
        h = topk_hits(scores, ids, truth, k)
        lo, hi = bootstrap_ci(h, truth)
        res[f"top{k}"], res[f"top{k}_ci"] = float(h.mean()), (round(lo, 3), round(hi, 3))
    best = scores.max(axis=1)
    hit1 = topk_hits(scores, ids, truth, 1)
    res["median_best_score_correct"] = float(np.median(best[hit1])) if hit1.any() else float("nan")
    res["median_best_score_wrong"] = float(np.median(best[~hit1])) if (~hit1).any() else float("nan")
    return res


# ---- query a new photo --------------------------------------------------------------------------------

def flank_from_photo(path: Path, facing: str, box=None):
    """-> (rgb crop (h,w,3) in [0,1], flank mask (h,w) bool, zebra box xyxy, detection score)."""
    import torch
    import torch.nn.functional as F
    from PIL import Image

    from zebrasight.data import heads, masks

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with Image.open(path) as im:
        rgb_full = np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0
    H, W = rgb_full.shape[:2]
    model, zebra = masks.load_model(device)
    s = min(1.0, masks.WORK_SIDE / max(H, W))
    img = torch.from_numpy(rgb_full).permute(2, 0, 1).to(device)
    if s < 1.0:
        img = F.interpolate(img[None], scale_factor=s, mode="bilinear", antialias=True, align_corners=False)[0]
    with torch.no_grad():
        out = model([img])[0]
    keep = out["labels"] == zebra
    boxes, mks, scores = out["boxes"][keep] / s, out["masks"][keep], out["scores"][keep]
    if len(boxes) == 0:
        raise SystemExit("no zebra found in the photo")
    if box is not None:
        iou = masks.box_iou(torch.tensor([box], dtype=torch.float32, device=device), boxes)[0]
        j = int(iou.argmax())
    else:
        area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        j = int(area.argmax())
    x0, y0, x1, y1 = masks.clip_box(*boxes[j].tolist(), W, H)
    m = F.interpolate(mks[j][None], size=(H, W), mode="bilinear", align_corners=False)[0, 0]
    body = (m > 0.5).cpu().numpy()[y0:y1, x0:x1]
    label, _ = heads.regions_v2(body, facing == "left", None)
    flank = label == heads.FLANK
    if not flank.any():
        raise SystemExit("found a zebra but could not cut a flank region (try --facing or --box)")
    ys, xs = np.nonzero(flank)
    fy0, fy1, fx0, fx1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    crop = rgb_full[y0:y1, x0:x1][fy0:fy1, fx0:fx1]
    return crop, flank[fy0:fy1, fx0:fx1], (x0, y0, x1, y1), float(scores[j])


def query(path: Path, facing: str, box, top: int, gallery: str, region: str = "flank") -> pd.DataFrame:
    import torch

    from zebrasight.metrics import megadesc

    df = pd.read_parquet(emb_path(region))
    gal = df if gallery == "all" else df[df.split == "gallery"]
    crop, m, zbox, det = flank_from_photo(path, facing, box)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = megadesc.load_model(device)
    q = megadesc.embed(model, [megadesc.to_input(crop, m)], device)
    ids, scores = rank_zebras(q, emb_matrix(gal), gal.zebra_id.to_numpy())
    order = np.argsort(-scores[0])[:top]
    res = pd.DataFrame({"rank": np.arange(1, len(order) + 1), "zebra_id": ids[order],
                        "similarity": np.round(scores[0][order], 3)})
    print(f"zebra found at {zbox} (detector score {det:.2f}); flank {m.shape[1]}x{m.shape[0]} px; "
          f"compared with {len(ids)} known zebras ({len(gal)} photos)")
    print(res.to_string(index=False))
    print("closed set: the best match is always a known zebra, even for a new animal")
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--regions", type=Path, default=REGIONS)
    b.add_argument("--region", default="flank")
    e = sub.add_parser("evaluate")
    e.add_argument("--region", default="flank")
    q = sub.add_parser("query")
    q.add_argument("image", type=Path)
    q.add_argument("--facing", choices=["left", "right"], default="left")
    q.add_argument("--box", type=float, nargs=4, default=None)
    q.add_argument("--top", type=int, default=5)
    q.add_argument("--gallery", choices=["gallery", "all"], default="gallery")
    a = ap.parse_args()
    t0 = time.perf_counter()
    if a.cmd == "build":
        build(a.regions, a.region)
    elif a.cmd == "evaluate":
        for k, v in evaluate(a.region).items():
            print(f"{k}: {v}")
    else:
        query(a.image, a.facing, a.box, a.top, a.gallery)
    print(f"done in {time.perf_counter() - t0:.0f} s")


if __name__ == "__main__":
    main()
