# Prompt for the DGX session (paste everything below the line into Claude Code, at the repo root)

---

# FINISH the positional-pattern study on the DGX (branch "positional"), by tomorrow morning

You are continuing a study that is already built and tested. Read FIRST, in this order:
CLAUDE.md, docs/handover_positional.md (status, results so far, rules, run order: it is the source of
truth), configs/positional.yaml, configs/metrics.yaml (amendments A1-A5), docs/contracts.md,
zebrasight/positional/smoke.py (the reference for how observers are called on oracle-aligned pairs).

Question: which POSITIONAL stripe patterns make two zebras distinguishable as distance grows (c0 clear
day, 16 distances 2-200 m, zebra-eye percepts). Regions: FULL BODY (priority), flank, rump. 50 zebras,
left side, gallery = earliest date, query = later date.

Hard rules (also in the handover, section 3):
- Do not change eye/, env/, data/ logic or any existing threshold. New parameters go in
  configs/positional.yaml with a one-line justification. If blocked, stop and ask.
- Commit amendment A6 (text in handover section 6) to configs/metrics.yaml BEFORE looking at any full
  result. Any later change = new dated amendment saying what was seen.
- Never compare raw similarity across distances; AUC per (region, distance) with 95% CI by bootstrap
  over zebras (n_boot 1000), NaN = miss. Use zebrasight.metrics.core (auc, boot_auc, ci, herd_top1).
- Held-out zebras (data/heldout_zebras.csv) only in the final confirmation of the top region.
- Photo-derived images internal only. Linux + Windows portable (pathlib, python -m entry points,
  joblib under __main__). Set run.low_priority false; use --n-jobs 32 and --device cuda.
- Every runner: resumable (skip finished outputs), logs to logs/, flags --n-jobs --device --regions.
- Commit after each step; push branch positional. Stop and report in plain words at each CHECKPOINT.

## Step 0: setup (handover section 4)
Unzip zebrasight_positional_handover.zip at the repo root. Python 3.11 venv;
pip install -r requirements-lock.txt --extra-index-url https://download.pytorch.org/whl/cu126 ;
pip install -e .[dev]. Check torch.cuda.is_available() and `pytest` -> 154 passed. Report GPU/CPU count.

## Step 1: A2 percepts
python -m zebrasight.positional.preflight percepts   (body + rump c0, 16 distances, into
cache/percepts_50.zarr; resumable). Verify with zebrasight.pipeline.read_store.PerceptStore(...).rows():
body 136 rows and rump 165 rows have c0 complete; flank 165 already there.

## Step 2-3: full alignment + go/no-go
python -m zebrasight.positional.alignment_run all --n-jobs 32. Read outputs/positional/b4_report.md.
Rule: >= 60% of same-zebra pairs with >= 4 inliers per region (also report the plausible-SIFT share).
Compare with the 5-zebra smoke (body 82% plausible, flank 29%, rump 14%). If a region is NO-GO, build
mask-contour registration in the canonical frame as its fallback before step 5.
CHECKPOINT 1: percept counts, alignment table, verdict per region.

## Step 4: commit amendment A6 (handover section 6), push.

## Step 5: ladder runner (build zebrasight/positional/ladder.py, test on the 5 smoke zebras first and
check it reproduces outputs/positional/smoke/f_scores.parquet at d_index 0/6/9 closely: the store keeps float16 percepts, so expect small differences, not exact equality)
Same logic as smoke.run(), but percepts from PerceptStore(cache/percepts_50.zarr).get(key, "c0", d_index),
all gallery x query pairs from align_<region>.parquet, all 16 distances, all 8 observers in
positional.observers.OBSERVERS, joblib over query rows, one parquet per (region, distance) so it resumes.
Also save the C8 per-cell outputs ("cells") of gabor_phase, orientation_field, aligned_minutiae
(float16 arrays, one file per region/distance). Body first, then flank, rump. ~1.7 h on 32 cores.

## Step 6: metrics
Per observer x region x distance: AUC + 95% CI (bootstrap over zebras), d*_AUC (lower CI >= 0.80),
top-1 at herd N = 20. Tables + AUC-vs-distance plots (body first). Check each A6 prediction: supported /
not supported / unclear, with the numbers.
CHECKPOINT 2: the ladder result in plain words.

## Steps 7-12 (handover section 5 has the details and the order if time runs short)
7 scale/arrangement (D1 band keep/remove, D2 patch shuffle, D3 phase scramble, D4 silhouette) on the
  QUERY percept inside its mask, body first, d_index 0, 4, 6, 9, 11, seed 0 first;
8 identity maps from the per-cell outputs: Fisher ratio and Beecher H_S per canonical cell, with
  body-part outlines (frame.part_mask);
9 morphing (D6): positional amplitudes vs style (width, contrast), body, 2 and 12.6 m;
10 transplant (D7) per body part + "own region re-pasted" control;
11 held-out confirmation of the top region (if body: make held-out body rows with
  preflight.body_quality + body_rows, then percepts, alignment, ladder for the top observers);
12 sensitivity of grade-C parameters, if time.
Priority if the night is short: 0-6 on body, then 7 (phase scramble + band filter), 8, 9, then
flank/rump, then 10-12. Partial but correct beats complete but rushed.

## Step 13: final report
outputs/positional/final_report.md: what was run (commit hash, times), alignment verdict, AUC tables and
plots, each prediction's verdict, identity maps, generator results, held-out result, limits (horse-proxy
eye, grade-C parameters, 50 zebras, canonical fallback on flank/rump). Commit + push. Tell me what to
send back (handover section 7).
