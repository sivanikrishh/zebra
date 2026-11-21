# ZebraSight architecture

## Data flow

```
photo crop (sRGB, cm_per_px, region mask, IDs)
   |  Pipeline.run(img, cm_per_px, mask, cond, meta)   sRGB decoded to linear RGB here
   v
ViewState --geometry--> --scene--> --atmosphere--> --eye--> make_percept --> Percept
           (deg/px,      (linear     (linear RGB:    (RGB -> [L,S] or rod,   (<= 256 px,
            az/el maps,   RGB: grass,  haze,           motion, streak,         rho = black
            resample)     shadows)     airlight)       CSF, noise)             fraction)
```

The eye stage runs its own ordered sub-stages, set in `configs/eye.yaml`:
`dichromacy -> motion -> streak -> csf -> noise`. Motion blur and the streak act at the
receptors, so they come before neural filtering (csf); noise is added last. Each sub-stage
can be switched off with `eye_stages.<name>: false` (dichromacy is required).

## DESIGN RULES

1. **One pipeline, ordered stages:** geometry -> scene -> atmosphere -> eye. Each stage is
   a `torch.nn.Module` with `forward(state) -> state`, registered by name and switchable in
   config. The pipeline refuses stage lists that break this order. Phase 1 baseline =
   geometry + eye only; scene and atmosphere exist as no-op stubs now.
2. **Colour is preserved until the eye.** Scene and atmosphere work in linear RGB
   (3 channels), because coloured airlight (haze), shadows and grass backgrounds come
   later. The eye converts RGB -> zebra cone signals [L, S] (rod channel at night).
3. **Spatial:** the state carries per-pixel visual-field coordinate maps (`az_map`,
   `el_map`, degrees; elevation positive up), so the visual-streak acuity can vary across
   the image (needed later for full scenes and VR). For a single crop: centre position +
   pixel offsets.
4. **The region mask** (face/flank/rump) travels inside the state, resampled with the image.
   Binary masks: area average then threshold at 0.5. Label masks: nearest-exact.
5. **IDs** (`zebra_id`, `image_id`, `session_id`, `region`) travel as metadata in
   `state.meta`; maths never uses them. `zebra_id` links face and flank of the same animal.
6. **Provenance:** every stage appends the parameter values it used to `state.log`.
7. **Tensors:** batched `(B, C, H, W)`, cuda if available else cpu, float32 compute,
   float16 storage.

## Types (`zebrasight/types.py`)

| Type | Fields |
|---|---|
| `ViewCond` | `distance_m, cond_id, light (day/dawn/dusk/night), luminance_cd_m2, field_az_deg=0, field_el_deg=0, azimuth_view_deg=0, beta_haze=0, speed_mps=0, motion_dir_deg=0, seed` |
| `ViewState` | `rgb (B,3,H,W) linear or None, cones (B,2,H,W) or None, rod (B,1,H,W) or None, mask (B,1,H,W), deg_per_px, az_map, el_map, acuity_map, cm_per_px, cond, meta, log, source_limited` |
| `Percept` | `y = L luminance (B,1,h,w) (rod at night), s = S cone or None at night, rho, mask, deg_per_px, meta, log, source_limited` |

- `acuity_map` (cycles/deg, (B,1,H,W)): local resolution limit, set by the streak stage and
  read by the csf stage.
- `rho` (Percept only): local **black fraction**, feature F2 ("rho-field"). It is the
  fraction of pixels of y below the per-image Otsu threshold (computed over the mask),
  box-averaged over a window of one acuity cycle (1 / regime acuity, in output pixels).
- `source_limited`: the source pixel subtends more than the 0.01 deg/px working grid
  (close range), so the photo's own resolution limits what can be resolved.

## Stages

| Name | Module | What it does |
|---|---|---|
| `geometry` | `eye/geometry.py` | deg/px = atan(cm/100/d); area-downsample rgb + mask to 0.01 deg/px if finer, else keep the grid and set source_limited; az/el maps; cos foreshortening hook (off) |
| `scene` | `env/scene.py` | no-op stub |
| `atmosphere` | `env/atmosphere.py` | no-op stub |
| `eye` | `eye/__init__.py` | runs the enabled sub-stages in `eye_order` |
| `dichromacy` | `eye/dichromacy.py`, `eye/spectral.py` | 2x3 matrix from Govardovskii templates x CRT primaries; rod-only below 0.02 cd/m2 |
| `motion` | `eye/motion.py` | line kernel of length speed x t_int / d along `motion_dir_deg` |
| `streak` | `eye/streak.py` | acuity_map; per-pixel blend of a Gaussian stack by log2(A / acuity_map) |
| `csf` | `eye/csf.py` | FFT log-parabola CSF on luminance (S(acuity) = 1, contrast threshold); chroma low-pass at 0.5 x A |
| `noise` | `eye/noise.py` | inverse-CDF Poisson + Gaussian, CPU generator seeded by `cond.seed` |
| (output) | `eye/percept.py` | area-downsample to max side 256, rho |

Adding a stage: subclass `Stage`, decorate with `@register_stage("name")`, read parameters
with `self.param("block.name")`, call `self.record(state, **params)` in `forward`, and add
the name to the config.

## VR

`python -m zebrasight.eye.export_vr` writes `vr/shaders/dichromat_matrix.json`: the
linear-RGB -> [L, S] matrix, the [L, S] -> linear sRGB display matrix (Vienot, Brettel &
Mollon 1999 adapted to S430/L545: blue and yellow primaries map to themselves), their product
(3x3 simulation matrix) and the rod row. A test checks that the file matches the model.

## Config

- `configs/grid.yaml`: distances, conditions c0-c9, regions, `phase1`.
- `configs/eye.yaml`: every eye parameter as `{value, sweep, unit, grade, cite}`, plus
  `eye_order` and `eye_stages`.
- `configs/melin_table2.yaml`: Melin et al. 2016 Table 2, used as a calibration check.

Configs load with OmegaConf (`zebrasight.config.load_config`). Tests override values with
`OmegaConf.update(cfg, "csf.b_peak_fraction.value", 0.12)`.

## Performance

`python -m zebrasight.eye.bench` (256x256, RTX 3050 Laptop / i7):

| | batch 1 | batch 16 |
|---|---|---|
| CPU, day (c0) | 14.6 ms | 8.5 ms |
| CPU, night (c9) | 34.6 ms | 24.4 ms |
| GPU, day (c0) | 13.4 ms | 7.3 ms |
| GPU, night (c9) | 21.3 ms | 7.8 ms |
| GPU, noise off | 4.8 ms | 1.1 ms |

The noise stage runs on the CPU, so CPU and CUDA give identical noise. It dominates GPU time.
