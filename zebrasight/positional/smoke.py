"""Part F: smoke test on real data (5 zebras, body/flank/rump, c0, 3 distances). NOT the experiment.

    python -m zebrasight.positional.smoke [--device auto|cpu|cuda]

Needs the B4 smoke alignments (python -m zebrasight.positional.alignment_run smoke). Steps:
  1. c0 percepts of the 5 smoke zebras' body/flank/rump rows at smoke.distance_indices (2, 13, 32 m),
     cached in outputs/positional/smoke/percepts.npz (the 50-zebra store is NOT touched).
  2. For every stored pair (same-zebra pairs + n_diff_pairs different-zebra pairs per region) and distance:
     the query percept is oracle-warped onto the gallery percept grid (B3) and every observer scores it.
  3. Sanity only: finite share, score range, same-zebra mean vs different-zebra mean at 2 m for the three
     positional observers named in the brief; time per pair and per percept; projected full-run time.
Outputs: outputs/positional/smoke/f_scores.parquet, f_report.md. No statistics, no claims.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from zebrasight.config import load_config
from zebrasight.positional import data_io as dio
from zebrasight.positional.align import PairAlign, oracle_warp
from zebrasight.positional.frame import percept_to_canonical, percept_to_crop
from zebrasight.positional.observers import OBSERVERS
from zebrasight.positional.observers.base import View

CHECK_AT_2M = ("gabor_phase", "orientation_field", "aligned_minutiae")
RANGES = {"gabor_phase": (0, 1), "gabor_magnitude": (-1, 1), "gabor_pooled": (-1, 1), "orientation_field": (-1, 1),
          "orientation_pooled": (-1, 1), "aligned_minutiae": (0, 1), "registered_layout_oracle": (-1, 1),
          "hotspotter_self": (0, np.inf)}


def make_view(p: dict, row: dict, frame) -> View:
    hw = p["y"].shape
    crop_hw = dio.crop_hw(row)
    M = percept_to_canonical(frame, row["bbox_xyxy"], crop_hw, hw)
    sx = percept_to_crop(crop_hw, hw)[0, 0]                  # crop px per percept px (isotropic resampling)
    return View(y=p["y"].astype(np.float32), mask=p["mask"], bl_px=frame.length_px / sx, px2can=M,
                deg_per_px=p["deg_per_px"], distance_m=p["distance_m"], region=row["region"], cond=p["cond"])


def run(device: str | None = None) -> tuple[pd.DataFrame, dict]:
    c = dio.cfg()
    reg = dio.regions()
    frames = dio.frames(reg)
    zebras = dio.smoke_zebras(reg)
    rows = reg[reg.zebra_id.isin(zebras) & reg.region.isin(list(c.data.regions))]
    base = dio.out_dir() / "smoke"
    di = list(c.smoke.distance_indices)
    cache = base / "percepts.npz"
    fresh = not cache.exists()
    t0 = time.perf_counter()
    P = dio.make_percepts(rows, di, c.data.condition, cache)
    t_percept = (time.perf_counter() - t0) / (len(rows) * len(di)) if fresh else float("nan")
    byk = rows.set_index("key")
    out = []
    for region in c.data.regions:
        al = pd.read_parquet(base / f"align_{region}.parquet")
        pairs = al[al.qkey.isin(byk.index) & al.gkey.isin(byk.index)]
        for r in pairs.to_dict("records"):
            gr, qr = byk.loc[r["gkey"]].to_dict(), byk.loc[r["qkey"]].to_dict()
            gr["key"], qr["key"] = r["gkey"], r["qkey"]
            pa = PairAlign.from_row(r)
            same = gr["zebra_id"] == qr["zebra_id"]
            for i in di:
                pg, pq = P[(r["gkey"], i)], P[(r["qkey"], i)]
                g = make_view(pg, gr, frames[int(gr["ann_id"])])
                q = make_view(pq, qr, frames[int(qr["ann_id"])])
                wy = oracle_warp(pq["y"], pa, dio.crop_hw(qr), dio.crop_hw(gr), pg["y"].shape)
                wm = oracle_warp(pq["mask"].astype(np.float32), pa, dio.crop_hw(qr), dio.crop_hw(gr), pg["y"].shape,
                                 nearest=True) > 0.5
                qa = g.like(wy, wm)
                for name, obs in OBSERVERS.items():
                    t1 = time.perf_counter()
                    try:
                        s, err = float(obs.score(g, qa if obs.needs_alignment else q, device=device)["score"]), ""
                    except Exception as e:  # sanity run: record, don't stop
                        s, err = float("nan"), f"{type(e).__name__}: {e}"
                    out.append({"region": region, "gkey": r["gkey"], "qkey": r["qkey"], "same": same,
                                "align_kind": pa.kind, "aligned_ok": pa.aligned_ok, "d_index": i,
                                "distance_m": pg["distance_m"], "observer": name, "score": s,
                                "seconds": time.perf_counter() - t1, "error": err})
    t = pd.DataFrame(out)
    t.to_parquet(base / "f_scores.parquet", index=False)
    return t, {"zebras": zebras, "n_rows": len(rows), "s_per_percept": t_percept}


def projection(t: pd.DataFrame, s_per_percept: float) -> list[str]:
    """Full experiment = every gallery x query pair of the 50 per region x 16 distances x all observers,
    plus percept generation for body/rump (flank exists). Scaled from the measured laptop times."""
    c = dio.cfg()
    pr = c.projection
    reg = dio.regions()
    n_pairs = sum(int((reg[(reg.region == r)].split == "gallery").sum()) * int((reg[(reg.region == r)].split == "query").sum())
                  for r in c.data.regions)
    n_dist = len(load_config("grid").distances_m["values"])
    per_obs = t.groupby("observer").seconds.median()
    s_pass = float(per_obs.sum()) * n_pairs * n_dist                      # one observer pass, single worker
    n_new_rows = int(reg.region.isin(["body", "rump"]).sum())
    s_perc = s_per_percept * n_new_rows * n_dist if np.isfinite(s_per_percept) else float("nan")
    lap = s_pass / pr.laptop_workers
    dgx = s_pass / pr.dgx_workers / pr.dgx_cpu_speedup
    L = ["## Projected full-run time (rough; assumptions in configs/positional.yaml projection)", "",
         f"- pairs (all gallery x query of the 50, body+flank+rump): {n_pairs}; distances: {n_dist}",
         "- median time per pair per observer (ms): "
         + ", ".join(f"{k} {v * 1000:.0f}" for k, v in per_obs.sort_values().items()),
         f"- one ladder pass (all observers, all pairs, 16 distances): single worker {s_pass / 3600:.1f} h -> "
         f"laptop ({pr.laptop_workers} workers) **{lap / 3600:.1f} h**; DGX ({pr.dgx_workers} cores) **{dgx / 3600:.1f} h**"]
    if np.isfinite(s_perc):
        L.append(f"- A2 body/rump c0 percepts ({n_new_rows} rows x {n_dist} distances, {s_per_percept:.2f} s per percept "
                 f"on the laptop GPU): laptop **{s_perc / 3600:.1f} h**; DGX GPU **{s_perc / pr.dgx_gpu_speedup / 3600:.1f} h**")
    L += [f"- generator steps (bands, shuffle, scramble, morph, transplant, occlusion) each cost about one ladder pass "
          f"per generator setting x seed; with the settings in configs/positional.yaml that is ~{pr.generator_passes} passes "
          f"-> DGX ~{pr.generator_passes * dgx / 3600:.0f} h (subset distances/observers to cut it).", ""]
    return L


def report(t: pd.DataFrame, info: dict) -> str:
    c = dio.cfg()
    L = ["# Part F smoke test (5 zebras, c0, 2 / 13 / 32 m) - sanity only, not the experiment", "",
         f"Zebras: {', '.join(info['zebras'])}; region rows: {info['n_rows']}; scored rows: {len(t)}", ""]
    errs = t[t.error != ""]
    L.append(f"Exceptions: {len(errs)}" + (f" (first: {errs.error.iloc[0]})" if len(errs) else ""))
    L += ["", "## Per observer: finite share, range, same vs different mean (all regions, per distance)", "",
          "| observer | finite | min | max | in range | 2 m same / diff | 13 m same / diff | 32 m same / diff |",
          "|---|---|---|---|---|---|---|---|"]
    ok_all = True
    for name, s in t.groupby("observer"):
        f = s[np.isfinite(s.score)]
        lo, hi = RANGES.get(name, (-np.inf, np.inf))
        inr = bool(((f.score >= lo - 1e-6) & (f.score <= hi + 1e-6)).all())
        ok_all &= inr
        cells = []
        for i in c.smoke.distance_indices:
            d = f[f.d_index == i]
            cells.append(f"{d[d.same].score.mean():.3f} / {d[~d.same].score.mean():.3f}")
        L.append(f"| {name} | {np.isfinite(s.score).mean():.0%} | {f.score.min():.3f} | {f.score.max():.3f} | "
                 f"{'yes' if inr else 'NO'} | " + " | ".join(cells) + " |")
    L += ["", "## Check: same-zebra mean > different-zebra mean at 2 m (per region)", ""]
    near = t[(t.d_index == c.smoke.distance_indices[0]) & np.isfinite(t.score)]
    for name in CHECK_AT_2M:
        for region in c.data.regions:
            d = near[(near.observer == name) & (near.region == region)]
            a, b = d[d.same].score.mean(), d[~d.same].score.mean()
            L.append(f"- {name} / {region}: same {a:.3f} vs different {b:.3f} -> {'ok' if a > b else 'NOT ok'}")
    L += ["", f"All finite scores in their expected range: {'yes' if ok_all else 'NO'}", ""]
    L += projection(t, info["s_per_percept"])
    return "\n".join(L) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description="Positional Part F smoke test (5 zebras)")
    ap.add_argument("--device", default=str(dio.cfg().run.device))
    a = ap.parse_args()
    dev = None if a.device == "auto" else a.device
    t, info = run(dev)
    txt = report(t, info)
    (dio.out_dir() / "smoke" / "f_report.md").write_text(txt, encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    main()
