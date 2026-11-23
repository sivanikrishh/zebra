# ZebraSight app features: research prototype → 3D application (handover)

Branch `app-prototype`. The 2D prototype validates the science; the 3D app should reproduce its **behaviour and
numbers**, not its look. Open the prototype: unzip `zebrasight_app_prototype.zip` and open `index.html`
(offline, no server). Rebuild it from the repo: `python scripts/build_app_demo.py` (~5 min, needs the data below).

## 1. The three features

| Tab | Question it answers | What it shows |
|---|---|---|
| **Recognition horizon** | How far away can a zebra still tell this individual from others? | Zebra-eye image at 16 distances (2–200 m) × 3 lights; a distance line with a small walking zebra; ONE number: identification accuracy (top-1 in a herd of 20, 95% CI, chance 5%) |
| **Stripe identity explorer** | What makes one zebra distinguishable? | Two original photos side by side. Mode A: matching stripe features (lines). Mode B: the study's stripe map + measured width / spacing / junctions / black fraction against all 165 photos. Study result: arrangement AUC 0.84 vs characteristics 0.52 (flank, 2 m, clear day) |
| **Pattern Lab** | Can you change a zebra's identity? | Exp 1 "Move away": zebra recedes in a scene; real rank of the true zebra among 50 at each distance + study result. Exp 2 "Change the stripes": thickness / contrast / arrangement / spacing sliders, each step a real modified photo with real model inference. Exp 3 "Fool the observer": needs horse photos (not in the dataset) |

Body regions: **Flank** and **Face** have validated results. **Full body is a visible "Results pending"
option**: show no values and never substitute flank/face numbers (full-body results are being computed on the DGX).

## 2. Scientific rules the app must keep (non-negotiable)

1. **Only stored or real-inference numbers.** Never interpolate between the 16 measured distances, never invent values.
2. **Name the metric correctly.** *Identification accuracy* = % of trials where the right zebra is ranked first
   (herd of 20). *AUC* = separability of same vs different zebras (0.5 = chance), **not** a percentage correct.
   *Similarity / matching features* = a score for one pair, comparable only within one distance and light.
3. **Like with like.** Distance results compare two photos BOTH seen through the zebra eye at the same distance
   and light. Never compare a blurred image with a sharp reference (biased).
4. **Label sources.** "Study result (all 50 zebras)" vs "this photo / this pair / experimental (real inference)".
5. **Computer observers, not zebras.** MegaDescriptor and HotSpotter are computational stand-ins; never claim
   a real zebra recognises (or is fooled) at these rates. Results are a lower bound.
6. **Full body = results pending** until the validated results are added.
7. **Licence.** All photo-derived images are GZGC CC BY-NC-ND 2.0: internal / non-commercial only, never public.
8. Images: every distance layer covers the same patch of coat; stretch it over one frame (bilinear) — fewer
   pixels at distance IS the effect (`vr/export/README.md`).

## 3. Code to reuse

| File | Role |
|---|---|
| `zebrasight/patternlab.py` | **All Pattern Lab science, no UI.** `modify(rgb, mask, bl_px, control, value)`; `Observers(region).identify(rgb, mask, true_zebra)` (MegaDescriptor + HotSpotter with the study's exact preprocessing, rank among the known zebras, ties); `self_similarity`; `check_against_study`; `distance_measurements(...)` |
| `zebrasight/positional/generators.py` | Stripe edits: `morph_style` (thickness, contrast), `morph_positional` (arrangement), `coat_rescale` (spacing; new, tested) |
| `configs/positional.yaml` → `pattern_lab` | Slider steps with justifications |
| `scripts/build_app_demo.py` | Builds `data.js` (horizon) and `explorer.js` (explorer); calls the lab build |
| `scripts/build_pattern_lab.py` | Builds `lab.js` (real inference for every slider step) |
| `scripts/app_demo.html` | The 2D prototype (reference behaviour, not production UI) |

Live inference in the 3D app (any slider combination): call `patternlab.Observers(...).identify(...)` from a
small Python service; MegaDescriptor ~0.1–0.3 s per image on a GPU. The prototype precomputes one control at a time.

## 4. Data contract (the three JSON files in the prototype)

**`data.js` → `window.DEMO` (Recognition horizon)**
`distances_m[16]`, `conditions[{id,label}]` (c0 clear day, c8 dusk, c9 night), `regions`, `observers`,
`curves[{region, condition, observer, points[16]{distance_m, top1_n20, top1_n20_ci, chance_n20}}]`,
`zebras`, `textures[zebra][region][cond][16]` = image paths.

**`explorer.js` → `window.EX` (Stripe identity explorer)**
`photos[{key, zebra, split (query = later day, gallery = earliest day), date}]`, `pool_q`, `pool_g`,
`pairs.orig["<qkey>|<gkey>"] = {md (MegaDescriptor cosine), hs (HotSpotter inliers), q_pts, g_pts (matched points, px of img/<key>_orig.png)}`,
`maps.c0[0][key] = {valid, junctions[[row,col]], endings[[row,col]], F5, F6}` + image `map/<key>_c0_d00.png`
(palette: 1 black stripe, 2 centre-line) over `img/<key>_c0_d00.png`,
`features.c0[0] = {width, spacing, junctions, black}` arrays aligned with `feat_keys` / `feat_zebra` (165 flank photos),
`stats_vs_pos` (study headline: `at_2m_c0`, `same_day_pairs_auc`).

**`lab.js` → `window.LAB` (Pattern Lab)**
`zebras`, `regions` (available), `pending_regions` (["body"]), `controls{control: [steps]}`, `study[...]`,
`checks` (validation per zebra/region),
`zebra[z].body = {height_m, img[cond][16]}` (scene visual only),
`zebra[z].exp1[region] = {query, ref, query_date, ref_date, measure[cond][16] = {megadesc|hotspotter: {ref_score, rank, tied_with, n_known} | null}, img[cond]{query, ref}[16]}`,
`zebra[z].exp2[region] = {original, reference, steps[control][i] = {value, neutral, img, self{megadesc, hotspotter}, megadesc{true_score, rank, tied_with, n_known, top}, hotspotter{...}}}`.
Display rule: `tied_with >= n_known - 1` → "No information" (every zebra scores the same).

## 5. Validation already done (keep these passing)

- HotSpotter match points: 0 differences from the official inlier tables.
- Stripe maps: junction density identical to the stored F5 values.
- Pattern Lab: unmodified photos reproduce the stored MegaDescriptor embeddings (cosine 1.0) and HotSpotter
  inliers (0 mismatches), all 4 zebras × flank and face.
- `pytest`: 155 passed (incl. `coat_rescale`).

## 6. Data needed to rebuild (not in git)

`cache/percepts_50.zarr`, `data/` (region tables, masks, photos) and `outputs/features/*.parquet`
(MegaDescriptor embeddings, HotSpotter tables, F1–F10 features). `app_data/` is in git.
The finished prototype (images + JSON) is shared separately as `zebrasight_app_prototype.zip` (internal only).

## 7. Known limitations

Fixed slider steps, one change at a time; 4 zebras in the Pattern Lab (those with a later-day full-body photo);
spacing denser than 0.85× would need coat from beyond the outline; MegaDescriptor's training data include GZGC
(scores may be optimistic); identification is closed-set (rank among 50), no validated "same zebra" threshold;
Experiment 3 needs licensed horse photos; full-body results pending.
