"""MegaDescriptor embeddings (learned observer, zero-shot, no training).

    python -m zebrasight.metrics.megadesc originals <regions.parquet> <out.parquet>
    python -m zebrasight.metrics.megadesc percepts  <store.zarr> <out.parquet> --conds c0,c8,c9

Model: hf-hub:BVRA/MegaDescriptor-L-384 (Swin-L, 384 px; licence CC BY-NC 4.0) via timm, fp16 on
the GPU. CAVEAT: MegaDescriptor was trained on re-ID datasets that include GiraffeZebraID (= GZGC),
so some of these photos may be in its training data; zero-shot results on GZGC can be optimistic.

Inputs: originals = the photo region (outside the mask grey), percepts = the zebra-eye view rendered
for a human exactly like the VR textures (dichromat display transform, grey at night), exposed to
the region's 99.5th percentile, sRGB, outside the mask grey. Both padded square (grey) and resized
to 384x384, ImageNet normalisation. No minimum-size rule (tiny percepts are upsampled).
Output columns: key, cond, d_index (originals: cond "orig", d_index -1), E_0000..E_1535.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

MODEL = "hf-hub:BVRA/MegaDescriptor-L-384"
SIZE = 384
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
GREY = 0.5


def load_model(device):
    import timm

    return timm.create_model(MODEL, num_classes=0, pretrained=True).half().to(device).eval()


def to_input(rgb: np.ndarray, mask: np.ndarray) -> torch.Tensor:
    """(h, w, 3) display RGB in [0,1] + mask -> (3, 384, 384) normalised tensor."""
    rgb = np.where(mask[..., None], rgb, GREY).astype(np.float32)
    h, w = mask.shape
    side = max(h, w)
    canvas = np.full((side, side, 3), GREY, np.float32)
    canvas[(side - h) // 2 : (side - h) // 2 + h, (side - w) // 2 : (side - w) // 2 + w] = rgb
    t = torch.from_numpy(canvas).permute(2, 0, 1)[None]
    t = F.interpolate(t, size=(SIZE, SIZE), mode="bilinear", align_corners=False, antialias=side > SIZE)
    return ((t - MEAN) / STD)[0]


def percept_rgb(p: dict, d_matrix: np.ndarray) -> np.ndarray:
    from zebrasight.vrpack.textures import display_linear, srgb_encode

    lin = display_linear(p, d_matrix)
    m = p["mask"]
    ref = float(np.percentile(lin[m], 99.5)) if m.any() else 1.0
    return srgb_encode(lin / max(ref, 1e-6))


@torch.no_grad()
def embed(model, batch: list[torch.Tensor], device) -> np.ndarray:
    x = torch.stack(batch).to(device).half()
    e = model(x).float()
    return F.normalize(e, dim=1).cpu().numpy().astype(np.float32)


def run(items, model, device, bs=16, every=200):
    """items: iterable of (meta dict, tensor or None). Returns DataFrame with E columns."""
    recs, embs, buf, bufmeta = [], [], [], []
    t0 = time.perf_counter()
    n = 0
    for meta, t in items:
        n += 1
        if t is None:
            recs.append(meta)
            embs.append(np.full(1536, np.nan, np.float32))
            continue
        buf.append(t)
        bufmeta.append(meta)
        if len(buf) == bs:
            e = embed(model, buf, device)
            recs += bufmeta
            embs += list(e)
            buf, bufmeta = [], []
        if n % every == 0:
            print(f"{n} embedded  {time.perf_counter() - t0:.0f} s", flush=True)
    if buf:
        recs += bufmeta
        embs += list(embed(model, buf, device))
    E = pd.DataFrame(np.vstack(embs), columns=[f"E_{i:04d}" for i in range(1536)])
    return pd.concat([pd.DataFrame(recs), E], axis=1)


def originals(regions_path, out, model, device):
    from zebrasight.data.load import load_region

    reg = pd.read_parquet(regions_path)
    reg = reg[reg.region.isin(["flank", "face"])]

    def items():
        for _, r in reg.iterrows():
            img, mask, _ = load_region(r)
            rgb = img[0].permute(1, 2, 0).numpy()
            m = mask[0, 0].numpy() > 0.5
            yield ({"key": f"{r.ann_id}_{r.region}", "cond": "orig", "d_index": -1}, to_input(rgb, m) if m.any() else None)

    df = run(items(), model, device)
    df.to_parquet(out, index=False)
    return df


def percepts(store, out, conds, model, device):
    from zebrasight.pipeline.read_store import PerceptStore
    from zebrasight.vrpack.params import matrices

    st = PerceptStore(store)
    rows = st.rows()
    rows = rows[rows.region.isin(["flank", "face"])]
    if "valid" in rows:
        rows = rows[rows.valid != False]  # noqa: E712
    d = matrices()[4]

    def items():
        for cond in conds:
            for key in rows.key:
                for di in range(len(st.distances)):
                    p = st.get(key, cond, di)
                    m = p["mask"]
                    yield ({"key": key, "cond": cond, "d_index": di}, to_input(percept_rgb(p, d), m) if m.any() else None)

    df = run(items(), model, device)
    df.to_parquet(out, index=False)
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["originals", "percepts"])
    ap.add_argument("src")
    ap.add_argument("out")
    ap.add_argument("--conds", default="c0")
    a = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.perf_counter()
    model = load_model(device)
    print(f"model loaded in {time.perf_counter() - t0:.0f} s", flush=True)
    if a.mode == "originals":
        df = originals(a.src, a.out, model, device)
    else:
        df = percepts(a.src, a.out, a.conds.split(","), model, device)
    print(f"wrote {a.out}: {len(df)} embeddings in {time.perf_counter() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
