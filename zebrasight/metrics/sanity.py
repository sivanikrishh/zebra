"""Sanity check on the ORIGINAL crops (no eye model): python -m zebrasight.metrics.sanity

Embeds the original face/flank crops with MegaDescriptor (skipped if outputs/features/megadesc_orig.parquet
exists), then prints AUC (200-draw zebra bootstrap CI) and top-1 at N=5/20 per region, for MegaDescriptor
and, if outputs/features/hotspotter_orig.parquet exists, the HotSpotter matcher. Writes
outputs/metrics/originals.csv (read by the figures and the app export). Exit code 2 if flank
MegaDescriptor AUC < 0.65 (crop/split problem: stop).
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from zebrasight.config import REPO_ROOT
from zebrasight.metrics.core import auc, boot_auc, ci, herd_top1, pairs, similarity, similarity_from_pairs, split

FEAT = REPO_ROOT / "outputs" / "features"
REGIONS = REPO_ROOT / "data" / "regions_50.parquet"


def main() -> int:
    emb = FEAT / "megadesc_orig.parquet"
    if not emb.exists():
        import torch

        from zebrasight.metrics.megadesc import load_model, originals

        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        originals(REGIONS, emb, load_model(dev), dev)
    reg = pd.read_parquet(REGIONS)
    reg["key"] = reg.ann_id.astype(str) + "_" + reg.region
    meta = reg[["key", "region", "zebra_id", "split", "mask_method"]]
    E = pd.read_parquet(emb).drop(columns=["cond", "d_index"])
    df = meta.merge(E, on="key")
    hs_path = FEAT / "hotspotter_orig.parquet"
    hs = pd.read_parquet(hs_path) if hs_path.exists() else None
    recs = []
    for region in ("flank", "face", "face_owlv2"):
        sub = df[df.mask_method == "owlv2"] if region == "face_owlv2" else df
        sub = sub[sub.region == region.split("_")[0]]
        sp = split(sub)
        zeb = np.intersect1d(np.unique(sp.gz), np.unique(sp.qz))
        obs = {"megadesc": similarity(sp, "megadesc", {"F": []})}
        if hs is not None:
            obs["hotspotter"] = similarity_from_pairs(sp, hs[hs.region == region.split("_")[0]])
        for name, S in obs.items():
            s, l, qz = pairs(sp, S, np.random.default_rng(0))
            lo, hi = ci(boot_auc(s, l, qz, zeb, 200, np.random.default_rng(1)))
            rec = {"region": region, "observer": name, "auc": auc(s, l), "auc_ci_low": lo, "auc_ci_high": hi}
            for n in (5, 20):
                acc = herd_top1(sp, S, n, 200, np.random.default_rng(3))
                rec[f"top1_n{n}"], (rec[f"top1_n{n}_ci_low"], rec[f"top1_n{n}_ci_high"]) = float(acc.mean()), ci(acc)
                rec[f"chance_n{n}"] = 1.0 / min(n, len(zeb))
            recs.append(rec)
            print(f"{region:10s} {name:10s} AUC {rec['auc']:.3f} ({lo:.3f}-{hi:.3f})  top-1 N=20 {rec['top1_n20']:.2f} "
                  f"(chance {rec['chance_n20']:.2f})  N=5 {rec['top1_n5']:.2f}", flush=True)
    out = REPO_ROOT / "outputs" / "metrics"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(recs).to_csv(out / "originals.csv", index=False)
    flank = next(r for r in recs if r["region"] == "flank" and r["observer"] == "megadesc")
    if flank["auc"] < 0.65:
        print("STOP: MegaDescriptor on original flank crops is near chance -> crop/split problem", flush=True)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
