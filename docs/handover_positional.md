# Handover: positional-pattern study (branch `positional`), 2026-11-23

Laptop ("light mode") part is finished: code + tests + 5-zebra smoke runs. No full data runs were done.
Everything below is for whoever runs the full experiment on the DGX.

## 1. What is done (committed)

| Part | What | Where |
|---|---|---|
| A1 | body-quality check: 136 of 165 photos usable for FULL BODY (cut / occluded / no mask excluded from body only, kept for flank + rump) | `zebrasight/positional/preflight.py`, `data/regions_50_positional.parquet`, metrics.yaml amendment A5 |
| A3 | held-out set FROZEN (seed 20261123): 271 zebras, 879 photos. Nothing computed on them. | `data/heldout_zebras.csv`, amendment A5 |
| B1-B3 | canonical body frame, pairwise alignment (SIFT + RANSAC affine, TPS if >= 12 inliers, canonical fallback), oracle warp to any distance | `frame.py`, `align.py` |
| B4 | batch alignment runner + feasibility report; run on the 5 smoke zebras only | `alignment_run.py` |
| C | observers: gabor_phase / magnitude / pooled, orientation_field / pooled, aligned_minutiae, + registered_layout_oracle, hotspotter_self | `zebrasight/positional/observers/` |
| D | generators: band filter, patch shuffle, phase scramble, silhouette, occlusion, morph (positional / style), transplant | `generators.py` |
| E | tests: `pytest tests/positional` -> 24 passed | `tests/positional/` |
| F | smoke test on real data (5 zebras, c0, 2 / 13 / 32 m) + run-time projection | `smoke.py` |

All parameters: `configs/positional.yaml` (value + one-line justification each).

## 2. Smoke-test findings you must know

**B4 alignment (5 zebras):** go/no-go rule (>= 60% of same-zebra pairs with >= 4 inliers) = GO in body,
flank and rump. BUT the QA image showed many 4-11-inlier affines collapse to a line (degenerate). A
plausibility gate was added (reject reflection, anisotropy > 2x, scale > 1.5x off the canonical map ->
canonical fallback). After the gate, same-zebra pairs with a usable SIFT map: **body 82%**, flank 29%,
rump 14%. So FULL BODY is where the oracle alignment is most trustworthy; flank/rump mostly run on the
canonical-frame fallback. Read both lines of `b4_report.md` after the full run, not just GO/NO-GO.

**F sanity (5 zebras, not a result):** 0 exceptions, no NaN explosions, every observer in its range.
Same-zebra > different-zebra at 2 m in 6 of 9 checks; NOT in: aligned_minutiae / **body**,
aligned_minutiae / rump, orientation_field / flank. Tiny sample, but look at aligned_minutiae on body.
orientation_field gives NaN on ~21% of pairs (NaN = counted as a miss).

## 3. Before you run anything: rules

1. **Never compare raw scores across distances or light.** Blur inflates similarity. Compare AUC/ranks
   within one distance only (`docs/todo.md`).
2. **Commit the predictions (G1) BEFORE the full run** as a dated amendment A6 in `configs/metrics.yaml`:
   near range gabor_phase / hotspotter > orientation_field > gabor_magnitude > pooled ~ chance; with
   distance orientation_field decays slower than gabor_phase / hotspotter; phase scramble -> chance;
   morphing: positional breaks identity, style doesn't; transplant: key region transfers, controls don't;
   held-out confirmation of the top region. State in it that the alignment plausibility gate was added
   after looking at the 5-zebra B4 QA image (no identity results were used).
3. **Do not change any threshold after seeing results.** If you must, write a dated amendment first.
4. **Held-out zebras (`data/heldout_zebras.csv`) are touched only in the final confirmation step.**
5. **Photo-derived images are internal only** (GZGC, CC BY-NC-ND 2.0): QA images never go on slides.
6. Linux: all entry points are `python -m ...`; use `--n-jobs 32`. Keep GPU batches within memory.

## 4. Files to copy (not in git: `cache/` and `outputs/` are gitignored)

- the repo at branch `positional` (`git pull`)
- `cache/percepts_50.zarr`
- `data/regions_50_positional.parquet`, `data/regions_50.parquet`, `data/regions.parquet`, `data/heldout_zebras.csv`
- `data/masks/` (v2 label PNGs incl. body masks), `data/interim/masks_v2_index.parquet`
- the original photos for the 50 + held-out zebras (paths as in the parquet `image_path` column)
- optional: `outputs/positional/` (frames cache, A1 table, smoke outputs) to compare against

Environment: Linux, Python 3.11, torch matching the DGX CUDA, then `pip install -r requirements-lock.txt`
and `pip install -e .[dev]`. Check: `pytest tests/positional` -> 24 passed.

## 5. Run order on the DGX

| Step | Command | Expected time | Status |
|---|---|---|---|
| 0 | `pytest tests/positional` | < 1 min | ready |
| 1 | A2: c0 percepts for **body** and rump, 16 distances, into `cache/percepts_50.zarr` (flank c0 already there). **The store has NO full-body percepts yet: every full-body step fails without this.** Use `zebrasight.pipeline.store.run(regions=data/regions_50_positional.parquet, out=cache/percepts_50.zarr, conds=["c0"], regions=("body","rump"))`, as `scripts/run_percepts_50.py` does for its passes. | ~0.1 h DGX (0.3 h laptop) | command to write + check on 2 rows first; not run |
| 2 | B2 full alignment: `python -m zebrasight.positional.alignment_run all --n-jobs 32` -> `outputs/positional/align_{body,flank,rump}.parquet`, resumable | minutes (~30 ms/pair, 18,176 pairs) | ready |
| 3 | Go/no-go: >= 60% of same-zebra pairs with >= 4 inliers (report also prints the stricter plausible-SIFT share). If NO-GO: build the better fallback (mask-contour registration in the canonical frame) before step 4. | - | rule fixed |
| 4 | G1 predictions amendment committed | - | to do (text only) |
| 5 | Ladder: every observer x every pair x 16 distances x body/flank/rump. Generalise `zebrasight/positional/smoke.py` `run()` to read percepts from the zarr store and all pairs. | ~1.7 h DGX (32 cores) | runner to build |
| 6+ | scale/arrangement (D1-D3), identity maps (C8 per-cell outputs -> Fisher / H_S), morphing (D6), transplant (D7), held-out confirmation, sensitivity | ~60 ladder passes if run in full = **~105 h**: subset (e.g. full body, a few distances) | runners to build |

Projection details: `outputs/positional/smoke/f_report.md`; assumptions in `configs/positional.yaml` `projection`.

## 6. What to send back

`outputs/positional/` (alignment parquets, b4_report.md, observer score tables, reports), `logs/`, and the
commit hash you ran from.
