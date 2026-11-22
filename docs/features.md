# Stripe features (person B)

Code: `zebrasight/features/` (registry, `common.prepare`, `stripes.py`, runner `run.py`, QC `qc.py`).
Thresholds: `configs/features.yaml`. Units: coat cm (`cm_per_px = 100·d·tan(deg_per_px)`),
densities per 100 cm², frequencies in cycles/cm. NaN = "not recognisable", never dropped.

Families: F1 width · F2 black fraction · F3 curvature · F4 orientation · F5 junctions ·
F6 endings · F7 spacing/frequency · F8 shadow stripes · F9 edge contrast · F10 persistence ·
L layout (black fraction on 16×24, PCA on gallery later) · M minutiae map (4×6) ·
R raw pixels (16×24 Weber-normalised percept for the raw-pixel baseline; **no** minimum-size rule).

## QC-driven deviations (decided at Checkpoint 1, 2026-11-22 ~08:30 IST, before any recognition result)

Only feature QC had been seen (montage, NaN rates); no AUC or other recognition number existed.

- **D1, lighting correction.** The prescribed single Otsu on y put the shaded underside of the
  flank into the black class (QC montage `outputs/qc/qc_montage_c0.png`), inflating F2 and L and
  adding fake skeleton branches. Before Otsu (all features, L and M), y is divided by a large-scale
  illumination estimate: Gaussian low-pass inside the mask by normalised convolution (so the mask
  edge isn't darkened), σ = 3 × the region's median stripe period at 2 m in coat cm (flank 7.61 cm,
  σ = 22.8 cm; face 3.38 cm, σ = 10.1 cm; from the first c0 run), converted to px at each distance,
  floored at 3 px. Sensitivity check: the original single-Otsu versions are kept as `F2_rawotsu_*`
  and `L_rawotsu_*`.
- **D2, spur pruning.** Prune length = max(1 acuity cycle, 3 px, 0.5 × median stripe width in px).
  At close range one acuity cycle is < 1 px, so fur texture survived as junctions (59–114 per crop
  at 2 m).

**Reliability note:** F5 / F6 / M (junctions, endings, minutiae map) are trustworthy only from ~10 m;
closer, they still partly count texture. F8 shadow stripes are absent (zero) in 60–90% of percepts.
