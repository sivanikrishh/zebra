# ZebraSight architecture

## Data flow

```
photo crop (sRGB, cm_per_px, region mask, IDs)
   |  Pipeline.run(img, cm_per_px, mask, cond, meta)
   v
ViewState --geometry--> --scene--> --atmosphere--> --eye--> Percept
           (deg/px,      (linear     (linear RGB:    (RGB -> [L,S] or rod,
            az/el maps,   RGB: grass,  haze,           streak acuity map,
            resample)     shadows)     airlight)       CSF, motion, noise)
```

The eye stage runs its own ordered sub-stages:
`dichromacy -> streak -> csf -> motion -> noise` (order set in `configs/eye.yaml`).

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
5. **IDs** (`zebra_id`, `image_id`, `session_id`, `region`) travel as metadata in
   `state.meta`; maths never uses them. `zebra_id` links face and flank of the same animal.
6. **Provenance:** every stage appends the parameter values it used to `state.log`.
7. **Tensors:** batched `(B, C, H, W)`, cuda if available else cpu, float32 compute,
   float16 storage.

## Types (`zebrasight/types.py`)

| Type | Fields |
|---|---|
| `ViewCond` | `distance_m, cond_id, light (day/dawn/dusk/night), luminance_cd_m2, field_az_deg=0, field_el_deg=0, azimuth_view_deg=0, beta_haze=0, speed_mps=0, seed` |
| `ViewState` | `rgb (B,3,H,W) linear or None, cones (B,2,H,W) or None, rod (B,1,H,W) or None, mask (B,1,H,W), deg_per_px, az_map, el_map, cond, meta, log, source_limited, rho` |
| `Percept` | `y = L luminance (B,1,h,w), s = S cone or None at night, rho, mask, deg_per_px, meta, log` |

`rho` is the per-pixel resolution limit (cycles/deg) set by the streak stage and used by
the CSF stage. (Interpretation chosen in task A; change it here if the team means something
else.)

`source_limited` is True when the photo's own pixel sampling, seen from that distance, is
coarser than the zebra eye's resolution limit (Nyquist of the source < acuity). In that
case the result reflects the camera, not the eye, and should be flagged in metrics.

## Stages and where they live

| Name | Module | Phase 1 | Status |
|---|---|---|---|
| `geometry` | `zebrasight/eye/geometry.py` | on | stub: sets deg/px, az/el maps, source_limited; no resampling yet |
| `scene` | `zebrasight/env/scene.py` | off | no-op stub |
| `atmosphere` | `zebrasight/env/atmosphere.py` | off | no-op stub |
| `eye` | `zebrasight/eye/__init__.py` | on | runs the sub-stages below |
| `dichromacy` | `zebrasight/eye/dichromacy.py` | | placeholder RGB -> [L,S]/rod |
| `streak` | `zebrasight/eye/streak.py` | | stub: uniform rho = photopic acuity |
| `csf` | `zebrasight/eye/csf.py` | | pass-through |
| `motion` | `zebrasight/eye/motion.py` | | pass-through |
| `noise` | `zebrasight/eye/noise.py` | | pass-through |

Adding a stage: subclass `Stage`, decorate with `@register_stage("name")`, call
`self.record(state, **params)` in `forward`, and add the name to the config.

## Config

- `configs/grid.yaml`: distances, conditions c0-c9, regions, `phase1`.
- `configs/eye.yaml`: every eye parameter as `{value, sweep, unit, grade, cite}`.
- `configs/melin_table2.yaml`: Melin et al. 2016 Table 2, used as a regression check.

Configs load with OmegaConf (`zebrasight.config.load_config`), so they can also be composed
with Hydra later.
