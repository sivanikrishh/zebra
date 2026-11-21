# Data contracts (A → B, V)

What each artefact contains and guarantees. Code that produces them is named in brackets.
All derived images are **internal – not for public display** (GZGC is CC BY-NC-ND 2.0).

## 1. `data/regions.parquet`: one row per (photo × region) [`zebrasight.data.subset`]

Full-run subset: every zebra with ≥ 2 left-side views on different dates (321 zebras), at most
4 views each (2 gallery from one session, 2 query from another), 1,044 views × {face, flank,
rump}. **Region rule v2** (`zebrasight.data.heads`, masks in `data/masks/gzgc_v2/`): face = OWLv2
head box ∩ body mask (fallback: front-most tip of the silhouette, 0.22 × body length; photos with
neither have **no face row**); flank = middle of the trunk between shoulder and hip, excluding neck
and head; rump = rear 25% of the trunk, excluding the tail. The rule-v1 table (face was mostly
neck) is kept as `data/regions_v1.parquet`; `data/regions_all.parquet` (all 2,315 left views) and
`data/regions_pilot.parquet` are still rule v1.

| Column | Type | Meaning |
|---|---|---|
| zebra_id | str | individual (`IBEIS_PZ_xxxx`); links face and flank of the same animal |
| image_id | str | `gzgc_<COCO image id>` |
| session_id | str | `<date>_<car>_<person>` (one camera team on one day) |
| encounter_id | str | `<zebra_id>@<session_id>` |
| dataset | str | `gzgc` |
| region | str | face / flank / rump |
| split | str | gallery (zebra's earliest date) / query (later dates); no encounter is in both |
| bbox_xyxy | int[4] | region's tight box, image px |
| mask_path | str | repo-relative POSIX path of the label PNG (whole annotation box) |
| mask_bbox_xyxy | int[4] | the PNG's frame in image px |
| mask_label | int | 1 face, 2 flank, 3 rump (PNG also has 4 = other body) |
| cm_per_px, cm_per_px_unc | float | scale (231.5 cm / box width), ±0.20 |
| scale_method | str | `body_length_bbox_width` |
| mask_method | str | face: `owlv2` or `silhouette_tip`; flank / rump: `maskrcnn_trunk`, or `bbox_rule` (no body mask; 7 views) |
| region_rule | str | `v2` (rule-v1 tables lack this column) |
| viewpoint, date | str | always `left`; ISO date |
| ann_id, image_path, theta, mask_iou, region_px | — | provenance |

Load a region with `zebrasight.data.load.load_region(row)` → `(img sRGB (1,3,h,w), mask (1,1,h,w), meta)`.

## 2. Conditions [`configs/grid.yaml`]

16 distances (2–200 m, log-spaced) × conditions:

| id | meaning |
|---|---|
| c0 | clear midday, side-on (reference) |
| c1 / c2 / c3 | hazy / dusty / dust storm (β₅₅₀ 0.002 / 0.01 / 0.03 per m) |
| c4 / c5 / c6 | view azimuth 30 / 60 / 75° (cos foreshortening) |
| c7 | dawn: **alias of c8** (same luminance regime), never computed separately |
| c8 | dusk (0.16 cd/m²) |
| c9 | moonless night (rod only, no S channel) |
| s25, s50 | 25 / 50% of the region in cast shadow (extra) |
| o20, o40, o60 | 20 / 40 / 60% occluded by grass / neighbours; mask updated (extra) |
| m2, m5 | moving 2 / 5 m/s (extra) |

`core_conditions` = c0–c9; `extra_conditions` = the rest.

## 3. Percept store (Zarr v3, layout 2) [`zebrasight.pipeline.store`]

```
/                         attrs: layout_version=2, distances_m[16], eye_config, env_config,
                                 aliases {c7: c8}, regions_file, created, licence
/rows/<ann_id>_<region>/  attrs: zebra_id, image_id, session_id, encounter_id, region, split,
                                 ann_id, cm_per_px, cm_per_px_unc, scale_method, mask_method, date
    <cond_id>/            one group per condition, ADDITIVE (adding conditions never rewrites others)
        y, rho            float16 (16, H, W)   luminance (L cone; rod at night) / black fraction
        s                 float16 (16, H/2, W/2)  S cone at half resolution; absent at night
        mask              uint8   (16, H, W)   region mask (occlusion removes hidden coat)
        shape, s_shape    int32   (16, 2)      valid (h, w) of each distance's arrays
        deg_per_px        float32 (16)         pixel size of y/rho/mask (s pixels are 2x)
        display_limited   bool    (16)         a size cap made the grid coarser than 2.5 px/acuity cycle
        source_limited    bool    (16)         the photo is coarser than the eye could resolve
        attrs: complete=True
    c7/                   attrs: alias_of=c8 (no arrays)
```

Arrays are zero-padded to the largest distance's size; always crop to `shape`. Compression:
blosc-zstd level 5, bit-shuffle; one shard file per array.

Read with `zebrasight.pipeline.read_store.PerceptStore`:

```python
st = PerceptStore("cache/percepts_full.zarr")
rows = st.rows()                      # DataFrame with the row attrs + available conds
p = st.get(rows.key[0], "c7", 4)      # aliases resolved; distance index 0..15
p["y"], p["s"], p["rho"], p["mask"], p["deg_per_px"], p["distance_m"], p["display_limited"]
```

The pilot store `cache/percepts_pilot.zarr` is still **layout 1** (conditions c0, c7, c8, c9 as the
first array axis; produced before the speed fixes; storage cap 1024, full-resolution S). See
`docs/handoff_B.md` for how to read or migrate it.

**Invalid face rows.** Face percepts made with region rule v1 are mostly neck. Both existing stores
carry `attrs.invalid_regions = {"face": "face_v1 = neck (invalid) …"}` at the root and
`valid: false` on every v1 face row: **skip rows with `valid == False`**. In the env-demo store the
faces of the 5 demo zebras were regenerated with rule v2 (`region_rule: "v2"`, `valid: true`).

## 4. Guarantees

- Deterministic: every percept is regenerable from `regions.parquet` + mask PNGs + configs.
  Noise seeds = hash(image_id, region, distance, cond_id, base seed); scene patterns = hash(image_id,
  region, cond_id, base seed) (same shadow / grass at every distance).
- Units: angles in degrees, distances in metres, luminance in cd/m², linear light in arrays.
- `rho` = local black fraction (Otsu on y, box over one acuity cycle), never acuity.
- c0 is untouched by the environment stages (scene and atmosphere skip clear, side-on, no-shadow,
  no-occlusion conditions).
