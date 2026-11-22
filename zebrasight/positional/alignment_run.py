"""B2 batch alignment and B4 feasibility report.

    python -m zebrasight.positional.alignment_run all  [--regions body,flank,rump] [--n-jobs N]   # DGX step 1
    python -m zebrasight.positional.alignment_run smoke                                       # laptop: 5 zebras

all:   every gallery x query pair (same and different zebras) of the 50, per region, on the sharp original
       crops -> outputs/positional/align_<region>.parquet (resumable: pairs already stored are skipped).
smoke: the 5 smoke-test zebras' same-zebra pairs + smoke.n_diff_pairs different-zebra pairs per region
       -> outputs/positional/smoke/align_<region>.parquet, the feasibility report b4_report.md and a QA
       image of 12 aligned same-zebra pairs (internal: derived from GZGC photos).
Go / no-go (configs/positional.yaml align.go_no_go_share): at least that share of SAME-zebra pairs must
reach min_inliers_affine inliers, else a better fallback is needed before the observers are trusted.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from zebrasight.positional import data_io as dio
from zebrasight.positional.align import align_records, crop_record


def pair_table(reg: pd.DataFrame, region: str, zebras=None, n_diff: int | None = None, seed: int = 0) -> pd.DataFrame:
    r = reg[reg.region == region]
    if zebras is not None:
        r = r[r.zebra_id.isin(zebras)]
    g, q = r[r.split == "gallery"], r[r.split == "query"]
    p = g[["key", "zebra_id"]].merge(q[["key", "zebra_id"]], how="cross", suffixes=("_g", "_q"))
    p = p.rename(columns={"key_g": "gkey", "key_q": "qkey", "zebra_id_g": "g_zebra", "zebra_id_q": "q_zebra"})
    p["same"] = p.g_zebra == p.q_zebra
    if n_diff is not None:
        diff = p[~p.same]
        diff = diff.sample(min(n_diff, len(diff)), random_state=seed)
        p = pd.concat([p[p.same], diff])
    p.insert(0, "region", region)
    return p.reset_index(drop=True)


def _records(reg: pd.DataFrame, keys, frames) -> dict:
    out = {}
    for row in reg[reg.key.isin(keys)].to_dict("records"):
        rgb, mask = dio.load_crop(row)
        out[row["key"]] = crop_record(rgb, mask, row["bbox_xyxy"], frames[int(row["ann_id"])])
    return out


def _job(reg, frames, qkey, gkeys):
    recs = _records(reg, [qkey, *gkeys], frames)
    rows = []
    for gk in gkeys:
        t0 = time.perf_counter()
        pa = align_records(recs[qkey], recs[gk])
        rows.append({"qkey": qkey, "gkey": gk, **pa.to_row(), "seconds": time.perf_counter() - t0})
    return rows


def run(pairs: pd.DataFrame, out: Path, n_jobs: int = 1) -> pd.DataFrame:
    """Align all pairs (resumable: pairs already in `out` are skipped); returns the full table."""
    from joblib import Parallel, delayed

    reg = dio.regions()
    frames = dio.frames(reg)
    done = pd.read_parquet(out) if out.exists() else pd.DataFrame(columns=["qkey", "gkey"])
    todo = pairs.merge(done[["qkey", "gkey"]], how="left", indicator=True)
    todo = todo[todo._merge == "left_only"].drop(columns="_merge")
    if len(todo):
        t0 = time.perf_counter()
        groups = list(todo.groupby("qkey").gkey.apply(list).items())
        res = Parallel(n_jobs=n_jobs)(delayed(_job)(reg, frames, qk, gks) for qk, gks in groups)
        new = pd.DataFrame([r for rows in res for r in rows])
        done = pd.concat([done, new], ignore_index=True) if len(done) else new
        out.parent.mkdir(parents=True, exist_ok=True)
        done.to_parquet(out, index=False)
        print(f"  aligned {len(new)} pairs in {time.perf_counter() - t0:.0f} s -> {out}", flush=True)
    return pairs.merge(done, on=["qkey", "gkey"], how="left")


def summary(t: pd.DataFrame, min_aff: int, min_tps: int) -> dict:
    out = {}
    for lab, s in (("same", t[t.same]), ("different", t[~t.same])):
        if not len(s):
            continue
        out[lab] = {"pairs": len(s), "median_inliers": float(s.inliers.median()),
                    "share_ge_affine": float((s.inliers >= min_aff).mean()),
                    "share_ge_tps": float((s.inliers >= min_tps).mean()),
                    "share_sift_used": float(s.kind.isin(["affine", "tps"]).mean()),
                    "rejected": s.rejected.replace("", None).dropna().value_counts().to_dict() if "rejected" in s else {},
                    "median_overlap": float(s.overlap.median()), "share_aligned_ok": float(s.aligned_ok.mean()),
                    "kinds": s.kind.value_counts().to_dict()}
    return out


def qa_image(t: pd.DataFrame, out: Path, n: int = 12) -> Path:
    """Gallery crop | query warped onto it | red/cyan overlay, for n same-zebra pairs (internal only)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from zebrasight.positional.align import PairAlign, warp_backward

    reg = dio.regions().set_index("key")
    s = t[t.same].sort_values("inliers", ascending=False)
    s = pd.concat([s.iloc[::max(1, len(s) // n)]]).head(n)
    fig, axes = plt.subplots(len(s), 3, figsize=(9, 2.2 * len(s)))
    axes = np.atleast_2d(axes)
    for i, r in enumerate(s.to_dict("records")):
        g_rgb, g_m = dio.load_crop(reg.loc[r["gkey"]].to_dict())
        q_rgb, _ = dio.load_crop(reg.loc[r["qkey"]].to_dict())
        pa = PairAlign.from_row(r)
        gy, qy = g_rgb.mean(axis=2), q_rgb.mean(axis=2)
        w = warp_backward(qy, pa, gy.shape)
        ov = np.dstack([gy, w, w]) * g_m[..., None]
        for ax, im, ttl in zip(axes[i], (gy * g_m, w * g_m, ov), ("gallery", f"query aligned ({r['kind']}, {r['inliers']} inl)", "overlay")):
            ax.imshow(np.clip(im, 0, 1), cmap="gray")
            ax.set_title(ttl, fontsize=8)
            ax.axis("off")
    fig.suptitle("B4 alignment QA: same-zebra pairs   INTERNAL - NOT FOR PUBLIC DISPLAY (GZGC, CC BY-NC-ND 2.0)",
                  fontsize=9, color="#b00020")
    fig.tight_layout()
    fig.savefig(out, dpi=80)
    plt.close(fig)
    return out


def main() -> None:
    c = dio.cfg()
    ap = argparse.ArgumentParser(description="Positional B2 alignment / B4 feasibility")
    ap.add_argument("mode", choices=["all", "smoke"])
    ap.add_argument("--regions", default=",".join(c.data.regions))
    ap.add_argument("--n-jobs", type=int, default=int(c.run.n_jobs))
    a = ap.parse_args()
    reg = dio.regions()
    zebras = dio.smoke_zebras(reg) if a.mode == "smoke" else None
    base = dio.out_dir() / ("smoke" if a.mode == "smoke" else "")
    lines = ["# B4 alignment feasibility" + (" (5 smoke-test zebras)" if zebras else ""), ""]
    if zebras:
        lines += [f"Zebras: {', '.join(zebras)}", ""]
    tabs = []
    for region in a.regions.split(","):
        pairs = pair_table(reg, region, zebras, c.smoke.n_diff_pairs if zebras else None, seed=0)
        t = run(pairs, base / f"align_{region}.parquet", a.n_jobs)
        tabs.append(t)
        s = summary(t, c.align.min_inliers_affine, c.align.min_inliers_tps)
        lines.append(f"## {region}")
        for lab, d in s.items():
            lines.append(f"- {lab}-zebra pairs: {d['pairs']}; median inliers {d['median_inliers']:.0f}; "
                         f">= {c.align.min_inliers_affine} inliers {d['share_ge_affine']:.0%}; "
                         f">= {c.align.min_inliers_tps} inliers {d['share_ge_tps']:.0%}; median overlap "
                         f"{d['median_overlap']:.2f}; aligned_ok {d['share_aligned_ok']:.0%}; kinds {d['kinds']}; "
                         f"SIFT map used (plausible) {d['share_sift_used']:.0%}; rejected as degenerate {d['rejected']}")
        ok = s.get("same", {}).get("share_ge_affine", 0) >= c.align.go_no_go_share
        lines += [f"- go/no-go (>= {c.align.go_no_go_share:.0%} of same-zebra pairs with >= "
                  f"{c.align.min_inliers_affine} inliers): **{'GO' if ok else 'NO-GO'}**",
                  f"- stricter check: same-zebra pairs with a PLAUSIBLE SIFT map {s.get('same', {}).get('share_sift_used', 0):.0%}; "
                  f"aligned_ok {s.get('same', {}).get('share_aligned_ok', 0):.0%} (the rest use the canonical-frame fallback)", ""]
    allt = pd.concat(tabs)
    lines.append(f"Median time per pair (alignment only): {allt.seconds.median() * 1000:.0f} ms")
    (base / "b4_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    if zebras:
        print("QA image:", qa_image(allt, base / "b4_qa_aligned_pairs.png"))


if __name__ == "__main__":
    main()
