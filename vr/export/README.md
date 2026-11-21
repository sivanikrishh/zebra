# ZebraSight VR export pack

This pack lets the VR show exactly what the Python zebra-eye model computes. You don't need to
read any Python: every formula the shaders need is written out below, and every number lives in
the JSON files in `params/`. Those JSONs are generated from the model's configs by
`python -m zebrasight.vrpack`, and a test fails if they ever drift from the model.

Source photos: Wild Me GZGC-IBEIS dataset, licence CC BY-NC-ND 2.0 (note for later, if any of
this is shown publicly).

## What's in the pack

| Path | What it is |
|---|---|
| `textures/<zebra>/<region>/<condition>/dNN_<metres>m.png` | 16 texture layers per zebra × region × condition |
| `textures/textures.json` | Index: distance bins, layer files, sizes, deg/px, exposure, condition ids |
| `params/dichromat_matrix.json` | Colour matrices (same numbers as `vr/shaders/dichromat_matrix.json`) |
| `params/haze.json` | Haze and dust: extinction per channel, airlight colour, visual ranges |
| `params/light.json` | Day / dusk / night: luminance, acuity, colour on/off, noise level |
| `params/field_of_view.json` | Azimuth convention, acuity map across the visual field, FOV zones |
| `params/loupe.json` | Magnification for the headset |
| `vr_lut.json` | PLACEHOLDER identification table (dummy values) for building the HUD |

**Zebras:** IBEIS_PZ_0334, 0395, 1267, 1648, 1725, using the best photo of each (flank and face).

**Conditions:** c0 clear day, c8 dusk, c9 moonless night, c1 hazy, c2 dusty, c3 dust storm.
c7 (dawn) is the same as c8, so use the c8 textures for dawn.

**Distance bins (16, log-spaced):** 2, 2.719, 3.696, 5.024, 6.829, 9.283, 12.619, 17.154, 23.318,
31.698, 43.089, 58.573, 79.621, 108.234, 147.128, 200 m (also in `textures.json`).

## Units and conventions

- Distances in metres, angles in degrees, luminance in cd/m².
- **Azimuth:** 0 = straight ahead of the zebra's head, positive to the right. **Elevation:** 0 = horizon,
  positive up.
- "Linear" means linear light. PNGs are sRGB-encoded, so decode them before doing any maths and
  re-encode at the end.

```
srgb_decode(v) = v / 12.92                    if v <= 0.04045
               = ((v + 0.055) / 1.055)^2.4    otherwise
srgb_encode(x) = 12.92 * x                    if x <= 0.0031308
               = 1.055 * x^(1/2.4) - 0.055    otherwise
```

## 1. Textures

Each PNG is the zebra-eye percept at one distance, already rendered for a human viewer:
- **Colour (day, dusk, haze):** `rgb_lin = LS_to_rgb · [L, S]`. L and S are the zebra's two cone
  signals, and `LS_to_rgb` is a 3×2 matrix in `dichromat_matrix.json`. Blue and yellow survive;
  red and green collapse together, as for the zebra.
- **Night (c9):** only rods work, so the texture is grey: `rgb_lin = (rod, rod, rod)`.
- **Exposure:** one number per zebra × region, the same for all its conditions and distances
  (`exposure` in `textures.json`). It is 1 / the brightest value in that set, so nothing clips: hazy
  layers turn pale, not white. Blending between layers never jumps in brightness.
- **Stored value:** `png.rgb = srgb_encode(clip(exposure · rgb_lin, 0, 1))`.
- **Alpha:** the region mask (flank or face). Render alpha 0 as transparent or as the coat around it.
- **Recover the model's linear values:** `rgb_lin = srgb_decode(png.rgb) / exposure`.

**Size on screen.** Every layer covers the same region of the coat, so map it to the same UV rect.
Layers have fewer pixels at larger distances because the zebra sees less detail there: that *is*
the effect, so stretch the small textures over the full rect. Use bilinear filtering, or nearest
filtering if you want to show the pixel grid. Physical size of the region: `region_size_cm`
(width, height) in `textures.json`. Its angular width at distance d is
`2·atan(width_m / (2d))` degrees.

## 2. Picking and blending layers by live distance

The textures exist at 16 distances. Between them, blend the two neighbouring layers in
**log-distance**, because the bins are log-spaced and blur grows roughly with log distance.

```
function sample_percept(uv, d_live):                  # d_live in metres
    d = clamp(d_live, bins[0], bins[15])              # outside 2..200 m: hold the end layer
    i = largest index with bins[i] <= d               # 0..14 (use 14 when d == bins[15])
    lo = log(bins[i]);  hi = log(bins[i+1])
    w = (log(d) - lo) / (hi - lo)                     # 0 at bins[i], 1 at bins[i+1]
    a = srgb_decode(texture[i].sample(uv))            # decode BEFORE blending
    b = srgb_decode(texture[i+1].sample(uv))
    rgb_lin = (1 - w) * a.rgb + w * b.rgb
    alpha   = (1 - w) * a.a   + w * b.a
    return rgb_lin, alpha                             # still divided by nothing: exposure is shared
```

Blend in linear light (decode first), never on the encoded PNG values.

## 3. Live haze: c0 textures only

**Use either the baked haze sets (c1, c2, c3) or live haze on the clear set (c0). Never both:
that would apply haze twice.**

Note: the c0 textures contain **no haze at all**. The model treats c0 as the unhazed reference
(`applied_in_model_textures: false` in `haze.json`). So live haze on c0 is the full effect, not an
extra on top of a clear-air baseline. For c1–c3 the haze is already in the pixels: show them as they are.

Haze model (Koschmieder), per channel, in linear light:

```
I(d) = I0 · t + A · (1 - t),      t = exp(-beta · d)
beta_c = beta_550 · (lambda_c / 550 nm)^(-alpha)       (Ångström law; lambda_R,G,B = 611, 549, 464 nm)
```

`beta_550`, `alpha` and the airlight colour `A` per condition are in `haze.json`. Because the
textures are already in the zebra's two-cone space, apply haze to L and S (exact for this display),
not to the display RGB:

```
rgb_lin = sample_percept(uv, d) / exposure                # model units
if colour regime:
    [L, S] = rgb_to_LS_display_inverse · rgb_lin           # 2×3 matrix, dichromat_matrix.json
    tL = exp(-beta_LS.L · d);   tS = exp(-beta_LS.S · d)    # haze.json, per condition
    L = L · tL + airlight_LS[0] · (1 - tL)
    S = S · tS + airlight_LS[1] · (1 - tS)
    rgb_lin = LS_to_rgb · [L, S]
else (night):
    t = exp(-beta_rod · d)
    rod = rgb_lin.r · t + airlight_rod · (1 - t);  rgb_lin = (rod, rod, rod)
out = srgb_encode(clip(exposure · rgb_lin, 0, 1))
```

`beta_LS` and `airlight_LS` are pre-computed in `haze.json` (the cone-weighted channel values).
Live haze is applied after the eye, whereas the model applies it before the eye's contrast filter.
That makes it a close approximation, not an exact match; use the baked c1–c3 sets when you need
the model's exact output.

**Visual range for the HUD:** Koschmieder `V = 3.912 / beta_550` (2% contrast), WMO meteorological
optical range `MOR = 2.996 / beta_550` (5%). Both are in `haze.json`.

**Airlight and light level.** The textures are relative (a sunlit white = 1), and so is `A`. The
haze-to-scene ratio is therefore the same by day and at dusk. Only the absolute airlight (cd/m²,
`airlight_by_regime`) is dimmer at dusk.

## 4. Light regimes (`light.json`)

| Regime | Luminance | Acuity | Channels |
|---|---|---|---|
| day | 16 cd/m² | 23.3 cpd | L + S colour |
| dusk (= dawn) | 0.16 cd/m² | 12.46 cpd | L + S colour |
| night | 1.6e-5 cd/m² | 1.61 cpd | rod only, grey |

Colour is on when luminance ≥ 0.02 cd/m². Photon and nerve noise are already baked into the
textures; `light.json` gives their levels if you want to show them on the HUD.

## 5. Field of view (`field_of_view.json`)

Acuity is highest in a horizontal band at eye level and falls off to the sides:

```
A(az, el) = A_regime · max(floor, exp(-el² / (2·sigma_el²)) · max(0, 1 - (1 - falloff_90) · |az| / 90))
sigma_el = 5°, falloff_90 = 0.5, floor = 0.16
```

The textures are computed for a zebra seen straight ahead at eye level (az = el = 0). For a zebra
elsewhere in the view, the model would blur it more. A cheap VR approximation is to pick the texture
layer for an "effective distance" `d_eff = d · A_regime / A(az, el)`, because blur scales with
distance ÷ acuity. Total FOV (~350°), binocular zone (65–80°, pointed down the nose) and the two
blind zones are **approximate (grade C)**: the panoramic model isn't implemented yet.

## 6. Loupe (`loupe.json`)

A Quest 3 shows about 25 pixels per degree, which is at most ~12.5 cycles/deg. The zebra resolves
23.3 cycles/deg by day, so without help the headset would throw away detail the zebra can see.
The loupe magnifies the zebra's view by `M = 23.3 / 12.5 ≈ 1.9` so the finest zebra-visible stripes
land on separate headset pixels. **The HUD always shows the true distance**: the loupe only
enlarges the picture, it doesn't move the zebra closer.

## 7. `vr_lut.json` (PLACEHOLDER)

Dummy values so the HUD can be built now. B will replace them with real feature values. Proposed
schema (not yet agreed):

```
distance_bins_m: [16 numbers]
features: ["F1", ..., "F10"]
zebras: [ { zebra_id, region, condition,
            features: { "F1": [16 values], ... },     # one value per distance bin
            id_probability: [16 values in 0..1] } ]
```

Look values up with the same two-bin log-distance blend as the textures (section 2).

## Regenerate

```
python -m zebrasight.vrpack            # params + textures + vr_lut placeholder + vr_export.zip
python -m zebrasight.vrpack --params-only
pytest tests/test_vrpack.py            # fails if params drift from the model
```
