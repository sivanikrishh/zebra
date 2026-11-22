# Handover: positional-pattern study (branch `positional`), 2026-11-23

**Question.** Which POSITIONAL stripe patterns (exact stripe position, stripe flow, junctions, where the
energy is) make two zebras distinguishable, and how does that change with distance (2-200 m, clear day
c0), as seen through the (horse-proxy) zebra eye? Regions: **FULL BODY (most important)**, flank, rump.

The laptop ("light mode") part is finished: all code built and tested, plus 5-zebra smoke runs. **No full
data runs have been done.** The paste-ready prompt for the DGX session is `docs/dgx_prompt.md`.

---------------------------------------------------------------------------------------------------------

## 1. Status per part

| Part | What | State |
|---|---|---|
| A1 | Body-quality check (frame edge, mask fill, other zebras' boxes, real mask) | **done** |
| A2 | c0 percepts for body + rump, 16 distances | **command ready + tested on 2 rows; NOT run** (DGX step 1) |
| A3 | Held-out set frozen | **done** |
| B1 | Canonical body frame (PCA of body mask, head left, 100 x 60 grid, body-part rules) | **done** |
| B2 | Pairwise alignment on sharp originals (SIFT + ratio + RANSAC affine, TPS >= 12 inliers, canonical fallback < 4, plausibility gate) | **done**; full run NOT done (DGX step 2) |
| B3 | Oracle alignment (stored map applied to any distance) + self alignment | **done** |
| B4 | Alignment feasibility | **done on 5 zebras**; full report comes from DGX step 2 |
| C1-C8 | Observers (torch Gabor bank on GPU, phase, magnitude, pooled, orientation field/pooled, aligned minutiae, per-cell outputs) | **done** |
| D1-D7 | Generators (band, shuffle, phase scramble, silhouette, occlusion, morph positional/style, transplant) | **done** (built + tested, not run at scale) |
| E | Tests | **done: full suite 154 passed** (positional: 24) |
| F | Smoke test on real data + time projection | **done** |
| G1 | Predictions amendment A6 in configs/metrics.yaml | **TO DO before any full result** (text in section 6) |
| G2 | One resumable command per experiment step | **partly**: A2 + alignment exist; experiment-step runners **TO BUILD** |
| G3 | DGX runbook | this document + `docs/dgx_prompt.md` |
| G4 | Final report | **TO DO after the DGX run** |

"torch / Gabor" is done: C1 `observers/gabor_bank.py` runs the complex Gabor bank with torch FFTs on cuda
if available, else cpu (`--device`). One deliberate deviation from the brief: the FFTs run in float32, not
float16 (cuFFT half precision needs power-of-two sizes and loses accuracy on the large low-frequency
kernels); `half=True` returns float16 for storage/transfer.

## 2. Results so far

### A1 / A3 (amendment A5 in configs/metrics.yaml)
- Full body usable on **136 of 165** photos (cut by frame / no body mask / mask fill < 0.40 / another
  zebra's box covering > 10% -> excluded from BODY only, kept for flank and rump). Table:
  `outputs/positional/a1_body_quality.csv`.
- Rows per region x split (`data/regions_50_positional.parquet`): body 74 gallery / 62 query;
  flank 86 / 79; rump 86 / 79. All gallery x query pairs: body 4,588; flank 6,794; rump 6,794 = **18,176**.
- Held-out: **271 zebras, 879 photos**, seed 20261123 (`data/heldout_zebras.csv`). Nothing computed on them.

### Percept store (`cache/percepts_50.zarr`, 0.82 GB)
- flank c0 (and other conditions): all 165 rows present. **body: 0 rows. rump: 0 rows.** -> A2 must run.
- A2 tested into a scratch store: 1.7 s per row for 16 distances on the laptop GPU -> 301 rows ~ 9 min.

### B4 alignment, 5 smoke zebras (IBEIS_PZ_1293, 1350, 1648, 1671, 1909); same-zebra pairs + 20 different
| region | same pairs | >= 4 inliers | >= 12 | plausible SIFT map (same) | plausible SIFT map (different) | rule (>= 60% with >= 4) |
|---|---|---|---|---|---|---|
| **body** | 11 | 100% | 45% | **82%** | 5% | GO |
| flank | 14 | 93% | 14% | 29% | 5% | GO |
| rump | 14 | 71% | 0% | 14% | 0% | GO |

The QA image showed many 4-11-inlier affines collapsing to a line, so a plausibility gate was added
(reject reflection, singular-value ratio > 2, scale > 1.5x off the canonical map -> canonical-frame
fallback; `align.max_anisotropy`, `align.max_scale_ratio`). An overlap-fraction bug (could exceed 1) was
fixed. Meaning: on **full body** the oracle alignment is mostly real SIFT/TPS; on flank and rump it is
mostly the coarser canonical-frame fallback. The gate is not identity-informed (it uses geometry only),
but it was added after looking at the smoke QA image: this must be stated in amendment A6.
Median alignment time 30 ms per pair. Files: `outputs/positional/smoke/b4_report.md`, `b4_qa_aligned_pairs.png`.

### F smoke test (same 5 zebras, c0, 2 / 12.6 / 31.7 m; all observers; 2,376 scores) - sanity only
- 0 exceptions, no NaN explosions, every observer's scores inside its valid range.
- Finite share: gabor_phase 91%, gabor_magnitude 91%, orientation_field **79%**, aligned_minutiae 93%,
  registered_layout_oracle 93%, pooled controls / hotspotter_self 100%. NaN = a miss (minimum similarity).
- Same-zebra mean > different-zebra mean at 2 m: gabor_phase body/flank/rump ok; orientation_field body
  ok, rump ok, **flank not**; aligned_minutiae flank ok, **body not, rump not**. 5 zebras: not a result,
  but check aligned_minutiae on body in the full run.
- Median ms per pair per observer: hotspotter_self 24, gabor_magnitude 65, registered_layout_oracle 67,
  gabor_pooled 89, aligned_minutiae 92, orientation_pooled 96, gabor_phase 127, orientation_field 130.
- Report: `outputs/positional/smoke/f_report.md`; raw: `f_scores.parquet`.

### Projected DGX time (assumptions in configs/positional.yaml `projection`)
| step | DGX estimate |
|---|---|
| A2 body + rump percepts | ~5-10 min |
| B2 full alignment, 18,176 pairs | ~10 min single core, less with --n-jobs 32 |
| one ladder pass: 8 observers x 18,176 pairs x 16 distances | **~1.7 h on 32 cores** (55.7 core-hours) |
| every generator setting x seed in full (~60 passes) | ~105 h -> **must subset** (see section 5) |

## 3. Rules (do not break)
1. **Never compare raw similarity across distances or light.** Blur inflates similarity. Compare AUC/ranks
   within one (region, distance). (docs/todo.md)
2. **Commit amendment A6 (predictions) BEFORE looking at any full result.** Thresholds and parameters are
   frozen; any change needs a new dated amendment first, saying what was seen.
3. **Held-out zebras only in the final confirmation step**, once, for the top region.
4. Metrics as pre-registered (configs/metrics.yaml): ROC AUC, 95% CI by bootstrap over ZEBRAS, n_boot 1000,
   d*_AUC = farthest distance with lower CI >= 0.80; NaN score = miss (never dropped). `zebrasight.metrics.core`
   has `auc`, `boot_auc`, `eer_dprime_tar`, `herd_top1`, `ci`.
5. Pairs are cross-date: gallery = earliest date, query = later date (column `split`). Never pair two
   photos of the same date.
6. Positional observers get the query ORACLE-warped onto the gallery grid (stored B2 map, `oracle_warp`);
   pooled controls and hotspotter_self get the unwarped query (`needs_alignment` flag). Follow `smoke.py`.
7. Photo-derived images are internal only (GZGC, CC BY-NC-ND 2.0): never on public slides.
8. Every new parameter goes in configs/positional.yaml with a one-line justification.
9. Linux: entry points `python -m ...`, pathlib only; joblib under `if __name__ == "__main__":`.
   On the DGX set `run.low_priority: false` and use `--n-jobs 32`.

## 4. What to give the colleague

1. **Git** (code, configs, docs, tests, `data/heldout_zebras.csv`): branch `positional` on GitHub.
2. **`zebrasight_positional_handover.zip` (~3.1 GB)**, made by `python scripts/pack_handover.py`. Unzip at
   the repo root. It holds everything NOT in git that is needed:
   - `cache/percepts_50.zarr` (0.82 GB)
   - `data/regions_50_positional.parquet`, `data/regions_50.parquet`, `data/regions.parquet` (**the region
     tables are not in git**: `data/` is gitignored), `data/heldout_zebras.csv`
   - `data/interim/` (mask index incl. head boxes), `data/masks/` (v2 label PNGs + body PNGs)
   - `data/raw/gzgc/gzgc.coco/annotations/` (COCO json; needed for a body check on held-out photos)
   - original photos of the 50 + held-out zebras only (985 photos, 2.2 GB; not the whole 20 GB data/raw)
   - `outputs/positional/` (frames cache, A1 table, smoke results to compare against)
3. Nothing else: no new cache is needed beyond the zarr in the zip. A2 adds body/rump to that same zarr.

Environment (Linux): Python 3.11;
`pip install -r requirements-lock.txt --extra-index-url https://download.pytorch.org/whl/cu126`
(the lock pins torch 2.14.1+cu126; needs an NVIDIA driver for CUDA >= 12.6), then `pip install -e .[dev]`.
Check: `python -c "import torch; print(torch.cuda.is_available())"` -> True, `pytest` -> 154 passed.

## 5. What is left, in order (overnight plan)

| # | Step | Command / what to build | Est. DGX time |
|---|---|---|---|
| 0 | Setup + tests | section 4; `pytest` -> 154 passed | 20 min |
| 1 | A2 percepts body + rump | `python -m zebrasight.positional.preflight percepts` (resumable). Check after: body 136 + rump 165 rows in the store | 10 min |
| 2 | B2 full alignment | `python -m zebrasight.positional.alignment_run all --n-jobs 32` -> `outputs/positional/align_{body,flank,rump}.parquet` + `b4_report.md` | 10 min |
| 3 | Go/no-go | >= 60% of same-zebra pairs with >= 4 inliers per region. Also read the plausible-SIFT line. NO-GO in a region -> build mask-contour registration in the canonical frame for it before step 5 | - |
| 4 | **G1 amendment A6** | add the text of section 6 to `configs/metrics.yaml`, commit, push | 5 min |
| 5 | **Ladder runner (BUILD)** + run | `zebrasight/positional/ladder.py`: like `smoke.run()` but percepts from `PerceptStore(cache/percepts_50.zarr).get(key, "c0", d_index)`, ALL pairs, 16 distances, joblib over pairs, resumable per (region, distance) parquet, `--n-jobs --device --regions`. Save per-cell outputs (C8) for gabor_phase / orientation_field / aligned_minutiae | build 1 h, run 1.7 h |
| 6 | Metrics | per observer x region x distance: AUC + 95% CI (bootstrap over zebras, 1000), d*_AUC; plot AUC vs distance per observer, body first | 30 min |
| 7 | Scale / arrangement (BUILD) | band keep/remove (D1), patch shuffle (D2), phase scramble (D3), silhouette (D4) applied to the QUERY percept inside its mask, then the ladder observers. **Body first**, distances 2, 6.8, 12.6, 31.7, 58.6 m (d_index 0, 4, 6, 9, 11), 1 seed first | ~3-5 h |
| 8 | Identity maps (BUILD) | from step-5 per-cell outputs: per canonical cell Fisher ratio and Beecher H_S (same vs different pairs); maps on the 12 x 20 / 40 x 24 grid with body-part outlines | 30 min |
| 9 | Morphing (BUILD) | D6 positional amplitudes vs style (width, contrast) on gallery-query same pairs; does identity break? Body, 2 and 12.6 m | ~2 h |
| 10 | Transplant (BUILD) | D7 per body part (neck, shoulder, mid_flank, hip, rump) donor -> recipient + "own region re-pasted" control; which part carries identity | ~2 h |
| 11 | Held-out confirmation (BUILD) | top region only. If body: build held-out body rows with `preflight.body_quality` + `body_rows` on the held-out photos (masks in data/masks/gzgc_v2), percepts c0, alignment, then the ladder for the top observers. Flank/rump held-out rows already exist in data/regions.parquet | ~1-2 h |
| 12 | Sensitivity | sweep the grade-C parameters (noise floor, shift search, align thresholds) on body, 2 distances | if time |
| 13 | G4 report | `outputs/positional/final_report.md`: what was run, AUC tables/plots, verdict per prediction, held-out result, limits. Commit + push | 30 min |

If the night is short: steps 0-6 on **body** first, then 7 (phase scramble + band filter only), 8, 9,
then flank/rump, then the rest. Partial but correct beats complete but rushed.

## 6. Text for amendment A6 (commit before step 5 results)

```yaml
  - id: A6
    when: "2026-11-2x (positional study, before any full positional result)"
    what: "Pre-registered predictions for the positional study (c0; body, flank, rump; AUC per distance):
      (1) near range: gabor_phase ~ hotspotter_self > orientation_field > gabor_magnitude > pooled
      controls (gabor_pooled, orientation_pooled) ~ chance; (2) with distance, orientation_field decays
      slower than gabor_phase and hotspotter_self; (3) phase scramble of the query -> chance; (4) morphing:
      positional displacement breaks identity, style changes (width, contrast) do not; (5) transplant: the
      key region carries identity to the recipient, controls do not; (6) the top region is confirmed on
      the frozen held-out set (A5). Alignment: B2 SIFT affines are rejected as degenerate (reflection,
      singular-value ratio > 2, scale > 1.5x the canonical map) and replaced by the canonical-frame map;
      this gate was added after viewing the 5-zebra B4 QA image (geometry only, no identity scores used)."
    why: "fixes the predictions and the alignment rule before results, so the study cannot be tuned to them."
```

## 7. What to send back
`outputs/positional/` (alignment parquets, b4_report.md, ladder score tables, AUC tables, plots, maps,
final_report.md), `logs/`, and the commit hash that was run. Push code changes to branch `positional`.
