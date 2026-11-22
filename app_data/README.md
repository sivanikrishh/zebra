# ZebraSight app data

The web app only **plays back** these files. It computes nothing.

| Path | What it is |
|---|---|
| `results.json` | All recognition numbers (schema below) |
| `figures/` | Slide-ready PNG charts (statistics only) |
| `textures/` | What a zebra sees, per zebra × region × distance × light; `textures/textures.json` indexes them |
| `params/` | `dichromat_matrix.json`, `haze.json`, `light.json`, for any live effect in the app |

`results.json` has a top-level `"status"` field. `"DUMMY ..."` means the numbers are made up and only the shape is real. `"final"` means the numbers are real. **Never show a DUMMY file in the demo.**

## The study in one paragraph

We take photos of 50 plains zebras and simulate how another zebra's eye would see each coat region (face or flank) at 16 distances from 2 to 200 m, in three lights: clear day, dusk and night. Then we ask: **can the same individual still be recognised?** Each photo of a zebra from one day (the "query") is compared with photos of the same zebra and of other zebras from a different day (the "gallery").

## Words used

- **AUC** (0.5 to 1): how often a same-zebra pair looks more alike than a different-zebra pair. 0.5 = chance, 1 = perfect.
- **CI**: 95% confidence interval from 200 resamples of the zebras (`n_bootstrap`). It is shown as `[low, high]`. 200 resamples is final for this demo; say so on the slides.
- **Usable**: the *lower* CI of the AUC is ≥ 0.80. **d\*** is the farthest distance that is still usable.
- **Top-1 at N = 5 / 20**: put N random zebras in a herd; how often is the best match the right one? `chance_n5` = 0.2, `chance_n20` = 0.05.
- **Observer**: who or what does the comparing.
  - `megadesc` (**main**) is a learned animal re-identification model that compares stripe *positions*.
  - `spatial_nn` and `interpretable_nn` use only *summary statistics* of the stripes (width, spacing, junctions…, plus a coarse layout). They form the "statistics floor".
  - `raw_pixel` is a dumb baseline.
- **Photo-limited**: at ≤ 9.3 m (`markers.photo_limited_max_m`) the simulated eye sees more detail than the photo contains. Shade that zone: the curve there is about the photo, not the eye.
- **Size rule**: very far away, the region becomes only a few pixels on the retina and the statistics can no longer be computed. `markers.size_rule_median_m` marks where this happens for the median photo; per-point `size_rule_share` gives the fraction of photos affected.

## `results.json` schema (version 1.0)

```
schema_version  "1.0"
status          "final" | "DUMMY ..."
generated       timestamp
n_zebras        50
n_bootstrap     200
bootstrap_note  text to show under charts
distances_m     [16 numbers]           index 0..15 = d_index
conditions      [{id, label}]          c0 Clear day, c8 Dusk, c9 Night (moonlight)
regions         [{id, label}]          face, flank
observers       [{id, label, kind, main}]   kind: positional | summary_statistics | baseline
families        [{id, label}]          clue families F1..F10, L, M
thresholds      {usable_auc_lower_ci: 0.80, reliable_auc_lower_ci: 0.95, chance_auc: 0.5}
markers         {photo_limited_max_m, size_rule_median_m: {face, flank}}

curves[]        one per region x condition x observer
  region, condition, observer
  points[16]:   d_index, distance_m,
                auc, auc_ci [lo, hi],
                top1_n5, top1_n5_ci, chance_n5,
                top1_n20, top1_n20_ci, chance_n20,
                usable (bool), photo_limited (bool), size_rule_share (0..1)

dstar[]         region, condition, observer, dstar_auc_m (null = never usable),
                dstar_auc_ci_m [lo, hi], never_usable (bool)

distance_loss_ratio[]   region, observer, condition (c8 | c9), vs "c0", ratio, ratio_ci
                        = d* in that light / d* in clear day (null if d* undefined)

leading_features[]      region, condition, distance_m,
                        families: [{family, label, auc}] sorted best first
                        (AUC when the summary-statistics observer uses that ONE clue family)

original_crops[]        region, observer, auc, auc_ci, top1_n20, chance_n20
                        = the same comparison on the ORIGINAL photos (no eye model);
                          draw it as a marker at the left edge of the AUC chart

statistics_vs_positions {headline,
                         at_2m_c0: [{region, observer, auc}],
                         same_day_pairs_auc: [{region, observer, auc}]}
```

Any number may be `null`, meaning undefined (for example, d\* when the curve is never usable). Show "–" for those, never 0.

## Licence

The textures come from GZGC photos (Wildbook / Great Zebra and Giraffe Count, research use). MegaDescriptor is licensed CC BY-NC 4.0. Both are for non-commercial demo use only.
