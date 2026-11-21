# Model parameters: zebra eye and environment

Machine-readable sources: `configs/eye.yaml` (eye) and `configs/env.yaml` (environment; see the
"Environment" section). Keep both in sync with this page.

**All values are horse proxies.** No zebra spatial-resolving-power estimate exists yet: the
Wits University *Project Plains Zebra* retina sub-project plans to measure it from
ganglion-cell density, and its stripes page expects horse-like acuity (< 20 cpd)
([Wits retina sub-project](https://www.wits.ac.za/admintest/project-plains-zebra/sub-projects/retina/)).

Evidence grades:
- **A**: measured in horse.
- **B**: derived in a published model (mostly Melin et al. 2016).
- **C**: our assumption. Must be swept; goes on the limits slide.

## Parameter table

| Parameter | Value | Sweep | Unit | Grade | Source |
|---|---|---|---|---|---|
| Acuity, photopic | 23.30 | 16.5, 23.3, 25 | cpd | A | Timney & Keil 1992; used for zebra by Melin 2016 (who cites Veilleux & Kirk 2014). 16.5 = Harman 1999 streak value |
| Acuity, mesopic | 12.46 | 8 – 15 | cpd | B | Melin 2016: midway between photopic and scotopic, (23.30 + 1.61) / 2 |
| Acuity, scotopic | 1.61 | 1 – 2.5 | cpd | B | Melin 2016: scaled from cat by eye size |
| Individual variation | — | — | — | A | Evans & McGreevy 2007: streak ganglion density varies with breed and nasal length (n = 30) |
| Luminance, day | 16 | — | cd/m² | B | Pasternak & Merigan 1981 cat CSF levels, as used by Melin 2016 |
| Luminance, dusk | 0.16 | — | cd/m² | B | same |
| Luminance, dawn | 0.16 | — (fixed) | cd/m² | C | assumed equal to dusk |
| Luminance, night (moonless) | 1.6e-5 | — | cd/m² | B | same as day |
| S cone peak | 430 | — | nm | B | Melin 2016 |
| L cone peak | 545 | 539, 545 | nm | B | Melin 2016; 545 in vitro (Yokoyama & Radlwimmer 1999); 539 horse ERG with S 428 (Carroll et al. 2001, grade A) |
| Pigment template | Govardovskii A1 | — | — | B | Govardovskii et al. 2000 (alpha + beta band; no lens/media filtering, C) |
| Display primaries | Typical CRT Brainard 1997 | — | — | C | colour-science spectra standing in for sRGB primaries, scaled to sRGB luminance weights; von Kries to display white. Gives neutral point 484.8 nm (L545) / 483.3 nm (L539) |
| Rod peak | 500 | — (fixed) | nm | C | generic mammalian rhodopsin |
| Colour threshold | 0.02 | — | cd/m² | A | Roth, Balkenius & Kelber 2008. Below it the model is rod-only with no S channel. Dusk (0.16) is above it, so colour stays on at dusk |
| Neutral point | 480 | test only | nm | A | Geisbauer et al. 2004; Roth et al. 2007 |
| Chromatic acuity | 0.5 | 0.3 – 1.0 | × luminance acuity | C | no horse data |
| CSF form | log-parabola | — | — | B | Melin 2016 method (cat CSF of Pasternak & Merigan 1981, peak shifted by acuity, slope constant) |
| CSF constraint | S(acuity) = 1 | — | — | B | Melin 2016 |
| CSF peak frequency b | 0.10 | 0.08 – 0.15 | × acuity | C | Melin did not report a, b, c |
| CSF peak sensitivity c | 50 | 20 – 100 | 1/contrast | C | same |
| Cat reference acuity | 5.81 | — | cpd | B | Melin 2016 |
| CSF mode | peli | peli / global | — | C | Peli 1990 local band-limited contrast threshold; global = per-FFT-coefficient (comparison only). See the note below |
| CSF band spacing | 0.5 | 0.5 – 1.0 | octaves | C | raised-cosine log-frequency bands (Peli used 1 octave) |
| Soft threshold half-width | 0.05 | 0.02 – 0.1 | log10 visibility | C | kept fraction ramps 0 → 1 between visibility 10^-0.05 and 10^+0.05 |
| CSF cut-off taper | 0.02 | 0.01 – 0.05 | × acuity | C | raised-cosine roll-off from A to 1.02 A |
| Weber fraction (brightness) | 0.42 – 0.45 | upper bound for noise | — | A | Geisbauer et al. 2004 (large-field task, 2 horses) |
| Luminance discrimination limit | 0.2 | upper bound for noise | log units | A | Macuda & Timney 1999: near chance below 0.2 log units (large field) |
| Streak elevation | 0 | — | deg | A | Timney & Keil 1992 (acuity limited by streak ganglion density); Guo & Sugita 2000 |
| Streak peak density | 4000 | — | cells/mm² | A | Guo & Sugita 2000 |
| Periphery floor | 0.16 | — | × streak acuity | A | Harman et al. 1999: 2.7 / 16.5 cpd |
| Streak vertical σ | 5 | 3 – 10 | deg | C | — |
| Azimuth falloff at 90° | 0.50 | 0.30 – 0.70 | × peak | C | linear from az 0 |
| Integration time, day | 30 | 15 – 60 | ms | C | no horse flicker-fusion value found; human cones ~60 Hz |
| Integration time, night | 100 | — (fixed) | ms | C | human rods ~15 Hz |
| Noise model | Poisson + Gaussian | — | — | C | seeded, device-independent (CPU generator, inverse-CDF Poisson) |
| Photon scale q0 | 1e7 | 1e6 – 1e8 | photons/(cd/m²·s·px) | C | white-level count ≈ 4.8e6 day, 4.8e4 dusk, 16 night |
| Neural noise σ | 0.01 | 0 – 0.05 | × mean signal | C | bounded above by the 0.42 Weber fraction |
| Working grid | 0.01 | — (fixed) | deg/px | C | Nyquist 50 cpd > max acuity 25 |
| Working grid samples per cycle | 5 | — (fixed) | px per acuity cycle | C | working grid = max(0.01, 1/(5 × regime acuity)): day unchanged, dusk ~1.6×, night ~12× coarser (speed) |
| Working max side | 512 | — (fixed) | px | C | close-range crops are coarsened in ONE resampling step; `display_limited` set, effective cpd logged |
| Foreshortening | off | — | — | C | hook: width × cos(view azimuth); phase 2 |
| Storage samples per cycle | 2.5 | — (fixed) | px per acuity cycle | C | storage grid = max(0.01, 1 / (2.5 × regime acuity)) deg/px; Gaussian pre-filter (σ = 0.4 × factor) + bilinear |
| Storage max side | 512 | — (fixed) | px | C | if the cap coarsens the grid, `Percept.display_limited = True` and the log records the effective cpd |
| S channel storage | half resolution | — | — | C | chroma acuity is 0.5 × luminance acuity (`percept.s_half_res`) |
| Noise seed | hash(image_id, region, distance_m, cond_id, base seed) | — | — | — | blake2b; different per image, reproducible |
| Photo colour | uncalibrated sRGB | — | — | C | Melin used a GretagMacbeth chart; we have none, so cone catches are approximate |

## Notes and caveats

### CSF: Melin's peak shift, two readings
The PLOS text gives the shift as `b_species = b_cat + (VA_species − VA_cat)` and says the
parabola is "in log scale", but it does not say whether b and VA are in log units. There
are two readings:

1. **Log units:** `log b_species = log b_cat + (log VA_species − log VA_cat)`, so
   `b_species = b_cat × VA_species / VA_cat`. The peak scales with the acuity ratio. Our
   `b = 0.10 × acuity` is this form.
2. **Linear units (cpd):** `b_species = b_cat + (VA_species − VA_cat)`. The peak moves by a fixed
   number of cycles/deg. For zebra photopic vision (23.3 vs cat 5.81) it would move by +17.5 cpd.

**The calibration does not depend on this choice.** We solve the parabola width so that
`S(acuity) = 1`, so the cut-off frequency (where the filter output reaches zero) is the
acuity under either reading. The reading changes only where the peak b sits below the cut-off.

**We use the log reading; the linear reading is not swept.** Under the linear reading,
`b_zebra = b_cat + 17.5 ≥ 17.5 cpd` for any `b_cat ≥ 0`, which places the zebra CSF peak at
0.75–0.9 × acuity. Measured mammal CSFs peak far below their acuity. For example, Brown-Norway
rats peak at 0.17 cpd against a limit of 1.6 cpd, a ratio of about 0.11 (Johnson,
Gregorich & Passaglia 2020, *Neuroscience* 449:63–73, doi 10.1016/j.neuroscience.2020.09.056,
PMC7642022; abstract and Table 1, while the Results text gives 0.15 / 1.7 cpd). So we use the
log reading, with b = 0.08–0.15 × acuity (sweep), which matches that ratio.

### Melin Table 2 is a calibration check, not an independent validation
Melin's Table 2 distances are computed from the acuities with `d = 2·width / tan(1/acuity)`.
`tests/test_configs.py` reproduces them to within 1 cm. That shows our filter's cut-off
matches the published resolution limits, by construction. It is **not** independent
validation of the model: both sides come from the same acuity numbers.

### CSF contrast threshold: `csf.mode` peli (default) vs global
`global` zeroes an FFT component when `m(f) × S(f) < 1`, with `m = 2|F(f)|/|F(0)|`. That is
per-coefficient contrast, so it depends on how spread the pattern's spectrum is and on crop
size, not only on stripe contrast. Wavy stripes spread their energy over many coefficients
and are wiped out. `peli` (Peli 1990, doi 10.1364/JOSAA.7.002032) thresholds the local
band-limited contrast per pixel and half-octave band instead. Details are in
`zebrasight/eye/csf.py`, including three refinements: a flat-top analysis band, S weighted at
each frequency, and the monogenic envelope. Without them, a full-contrast grating at 0.9 ×
acuity fell between two bands and was erased.

RMS contrast after the CSF: "none" is the CSF transfer with no threshold. Stripes are
0.1 → 0.9 sRGB at a 5 cm period and 0.1 cm/px, straight vs wavy (x + 2.5 cm × sin(y/6 cm)):

| Pattern | Mode | c0 20 m | c8 20 m | c0 40 m |
|---|---|---|---|---|
| Straight | none / global / peli | 0.355 / 0.355 / 0.358 | 0.119 / 0.115 / 0.120 | 0.098 / 0.095 / 0.098 |
| Wavy | none / global / peli | 0.321 / 0.300 / 0.321 | 0.089 / **0.061** / 0.089 | 0.071 / **0.029** / 0.070 |

With peli, the wavy coat matches its unthresholded value; the global rule loses up to 60%.
Wavy stays below straight even without any threshold. That is physical: tilted stripes have
a higher local frequency (up to 8% here), which the CSF attenuates more near the acuity.
`tests/eye/test_fixes.py` checks that peli is independent of crop size (512 vs 1024 crop,
central contrast within 5%).

### Other notes
- **Pasternak & Merigan 1981 peak values not found.** The paper is paywalled and has no
  abstract online, so the cat CSF peak frequency and sensitivity could not be read. b and c
  stay grade C. Replace them if someone gets the PDF (doi below).
- **Weber fraction and the 0.2 log-unit limit** come from large-field brightness tasks, not
  gratings. Use them only as an upper bound for the noise sweep.
- **Horse critical flicker fusion:** no published value found (clinical ERGs use a 30 Hz
  flicker as a stimulus, which is not a threshold). Integration times stay grade C.
- **Yokoyama & Radlwimmer 545 nm:** the DOI is verified, but the 545 nm horse value was seen
  only in a secondary table (Yokoyama et al. 2005 citing it), not in the paper's own text.
- **Geisbauer 2004 DOI** (10.1139/z04-026) came from a citation aggregator. Its metadata
  matches, but it was not confirmed on the publisher's site.
- **Melin Table 2** (`configs/melin_table2.yaml`) is an image on the PLOS page. The values
  come from the brief. Every row reproduces `d = 2·width / tan(1/acuity)` to within 1 cm
  with the acuities above (`tests/test_configs.py`), and the rounded values in the Results
  text agree.

## Environment (configs/env.yaml, configs/grid.yaml)

Physical order, enforced by the pipeline: **geometry** (distance, viewing angle) → **scene**
(background, cast shadows, occlusion) → **atmosphere** (haze / dust along the path) → **eye**.
Scene and atmosphere work in linear RGB. Conditions c0–c9 plus extras s25, s50, o20, o40, o60,
m2, m5; c7 (dawn) is stored as an alias of c8 (dusk).

| Parameter | Value | Sweep | Unit | Grade | Source / note |
|---|---|---|---|---|---|
| Viewing angle φ | c4/c5/c6 = 30/60/75 | — | deg | C | coat compressed along the body axis by cos φ before resampling (planar flank); validation stub `scripts/validate_oblique.py` (real GZGC frontleft/backleft) |
| β₅₅₀ clear (c0) | 0.0002 | — | 1/m | B | V = 3.912/β ≈ 20 km; ~consistent with Nairobi AERONET AOD₄₄₀ 0.19 (Boiyo et al. 2019) over a 1–2 km mixing layer. Atmosphere is **not applied** to `aerosol: clear`, so c0 is unchanged |
| β₅₅₀ hazy (c1) | 0.002 | — | 1/m | C | V ≈ 1.96 km, MOR ≈ 1.5 km; a standard meteorological visibility class, **not a savanna measurement** |
| β₅₅₀ dusty (c2) | 0.01 | — | 1/m | C | V ≈ 390 m, MOR ≈ 300 m |
| β₅₅₀ dust storm (c3) | 0.03 | — | 1/m | C | V ≈ 130 m, MOR ≈ 100 m |
| Koschmieder threshold | 0.02 | — | contrast | B | Koschmieder 1924: V = ln(50)/β = 3.912/β |
| WMO MOR threshold | 0.05 | — | transmittance | B | WMO CIMO guide: MOR = ln(20)/β = 2.996/β |
| Channel wavelengths | 611 / 549 / 464 | — | nm | B | dominant wavelengths of the sRGB R/G/B primaries vs D65 (colour-science, CIE 1931) |
| Ångström α, haze | 1.3 | 1.0 – 1.6 | — | C | Ångström 1929 "average" (UNVERIFIED value): β_B/β_R = 1.43, haze is bluish |
| Ångström α, dust | 0.2 | 0.0 – 0.5 | — | B | coarse mineral dust; desert AERONET sites ~0.24–0.6 (Dubovik et al. 2002; Eck et al. 1999): β_B/β_R = 1.06, near-neutral |
| Airlight, haze | (0.80, 0.88, 1.00) at Y 0.90 | Y 0.6 – 1.2 | linear RGB × sunlit white | C | bluish-white horizon |
| Airlight, dust | (1.00, 0.80, 0.55) at Y 0.70 | Y 0.4 – 1.0 | linear RGB × sunlit white | C | yellow-brown horizon |
| Airlight vs light regime | relative | — | — | C | the image and its airlight are both lit by the sky, so the haze/scene ratio is regime-independent; absolute airlight (Y × regime luminance, cd/m²) is logged and is 100× dimmer at dusk |
| Shadow darkening | 0.55 | 0.40 – 0.70 | luminance removed | C | skylight-only shadow (s25, s50 = 25 / 50% of the region) |
| Shadow tint | (0.90, 1.00, 1.15) | — | relative, unit luminance | C | bluish skylight |
| Shadow penumbra σ | 0.04 | 0.02 – 0.08 | × region width | C | Gaussian |
| Occlusion | o20 / o40 / o60 | — | share of region | C | neighbour silhouettes (~40% of the cover) + grass blades from the lower edge; occluded pixels → dry-grass colour; region mask updated |
| Grass blade width | 0.012 | 0.006 – 0.025 | × region width | C | — |
| Motion speed | m2 / m5 = 2 / 5 | — | m/s | C | horizontal, existing motion stage (blur = speed × 30 ms / distance) |
| Background plates | short green / tall dry / bare dust | — | linear RGB mean + texture | C | default off for crops; `make_plate` reused for VR |

**Limits slide:** β values for c1–c3 and all airlight colours are standard meteorological ranges
and assumptions, not savanna measurements. I found a Nairobi AERONET climatology (annual mean
AOD₄₄₀ 0.19 ± 0.04; Boiyo et al. 2019) but no surface visibility or dust-event extinction data
for the GZGC area. The haze α = 1.3 value is Ångström's classic average; that value was not confirmed
in this session (UNVERIFIED).

## References

| Key | Reference | DOI | Verified |
|---|---|---|---|
| timney1992 | Timney B, Keil K (1992) Visual acuity in the horse. *Vision Res* 32(12):2289–2293 | [10.1016/0042-6989(92)90092-W](https://doi.org/10.1016/0042-6989(92)90092-W) | yes |
| melin2016 | Melin AD, Kline DW, Hiramatsu C, Caro T (2016) Zebra stripes through the eyes of their predators, zebras, and humans. *PLOS ONE* 11(1):e0145679 | [10.1371/journal.pone.0145679](https://doi.org/10.1371/journal.pone.0145679) | yes |
| veilleux2014 | Veilleux CC, Kirk EC (2014) Visual acuity in mammals: effects of eye size and ecology. *Brain Behav Evol* 83(1):43–53 | [10.1159/000357830](https://doi.org/10.1159/000357830) | yes |
| evans2007 | Evans KE, McGreevy PD (2007) The distribution of ganglion cells in the equine retina and its relationship to skull morphology. *Anat Histol Embryol* 36(2):151–156 | [10.1111/j.1439-0264.2006.00749.x](https://doi.org/10.1111/j.1439-0264.2006.00749.x) | yes |
| pasternak1981 | Pasternak T, Merigan WH (1981) The luminance dependence of spatial vision in the cat. *Vision Res* 21(9):1333–1339 | [10.1016/0042-6989(81)90240-6](https://doi.org/10.1016/0042-6989(81)90240-6) | yes |
| carroll2001 | Carroll J, Murphy CJ, Neitz M, Ver Hoeve JN, Neitz J (2001) Photopigment basis for dichromatic color vision in the horse. *J Vis* 1(2):80–87 | [10.1167/1.2.2](https://doi.org/10.1167/1.2.2) | yes |
| yokoyama1999 | Yokoyama S, Radlwimmer FB (1999) The molecular genetics of red and green color vision in mammals. *Genetics* 153(2):919–932 | [10.1093/genetics/153.2.919](https://doi.org/10.1093/genetics/153.2.919) | yes (545 nm value: secondary source) |
| govardovskii2000 | Govardovskii VI, Fyhrquist N, Reuter T, Kuzmin DG, Donner K (2000) In search of the visual pigment template. *Vis Neurosci* 17(4):509–528 | [10.1017/S0952523800174036](https://doi.org/10.1017/S0952523800174036) | yes |
| roth2008 | Roth LSV, Balkenius A, Kelber A (2008) The absolute threshold of colour vision in the horse. *PLoS ONE* 3(11):e3711 | [10.1371/journal.pone.0003711](https://doi.org/10.1371/journal.pone.0003711) | yes |
| roth2007 | Roth LSV, Balkenius A, Kelber A (2007) Colour perception in a dichromat. *J Exp Biol* 210(16):2795–2800 | [10.1242/jeb.007377](https://doi.org/10.1242/jeb.007377) | yes |
| geisbauer2004 | Geisbauer G, Griebel U, Schmid A, Timney B (2004) Brightness discrimination and neutral point testing in the horse. *Can J Zool* 82(4):660–670 | [10.1139/z04-026](https://doi.org/10.1139/z04-026) | yes (aggregator only) |
| macuda1999 | Macuda T, Timney B (1999) Luminance and chromatic discrimination in the horse (*Equus caballus*). *Behav Processes* 44(3):301–307 | [10.1016/S0376-6357(98)00039-4](https://doi.org/10.1016/S0376-6357(98)00039-4) | yes |
| harman1999 | Harman AM, Moore S, Hoskins R, Keller P (1999) Horse vision and an explanation for the visual behaviour originally explained by the 'ramp retina'. *Equine Vet J* 31(5):384–390 | [10.1111/j.2042-3306.1999.tb03837.x](https://doi.org/10.1111/j.2042-3306.1999.tb03837.x) | yes |
| guo2000 | Guo X, Sugita S (2000) Topography of ganglion cells in the retina of the horse. *J Vet Med Sci* 62(11):1145–1150 | [10.1292/jvms.62.1145](https://doi.org/10.1292/jvms.62.1145) | yes |
| peli1990 | Peli E (1990) Contrast in complex images. *J Opt Soc Am A* 7(10):2032–2040 | [10.1364/JOSAA.7.002032](https://doi.org/10.1364/JOSAA.7.002032) | yes |
| caves2018 | Caves EM, Johnsen S (2018) AcuityView: an R package for portraying the effects of visual acuity on scenes observed by an animal. *Methods Ecol Evol* 9(3):793–797 | [10.1111/2041-210X.12911](https://doi.org/10.1111/2041-210X.12911) | yes |
| johnson2020 | Johnson NP, Gregorich SM, Passaglia CL (2020) Spatiotemporal contrast sensitivity of Brown-Norway rats under scotopic and photopic illumination. *Neuroscience* 449:63–73 | [10.1016/j.neuroscience.2020.09.056](https://doi.org/10.1016/j.neuroscience.2020.09.056) | yes |
| koschmieder1924 | Koschmieder H (1924) Theorie der horizontalen Sichtweite. *Beitr Phys freien Atmos* 12:33–53, 171–181 | — | yes (0.02 / 3.912 via WMO guide and secondary sources; page range varies) |
| wmo_cimo | WMO (2018) Guide to Instruments and Methods of Observation (WMO-No. 8), Vol. I, Ch. 9 Visibility | [PDF](https://old.wmo.int/extranet/pages/prog/www/IMOP/publications/CIMO-Guide/Prelim_2018_ed/8_I_9_en_MR_tc.pdf) | yes (2018 preliminary edition) |
| angstrom1929 | Ångström A (1929) On the atmospheric transmission of sun radiation and on dust in the air. *Geogr Ann* 11:156–166 | — | **UNVERIFIED** (origin of the exponent confirmed; the 1.3 value and bibliographic details not confirmed) |
| dubovik2002 | Dubovik O et al. (2002) Variability of absorption and optical properties of key aerosol types observed in worldwide locations. *J Atmos Sci* 59:590–608 | — | yes (DOI not confirmed) |
| eck1999 | Eck TF et al. (1999) Wavelength dependence of the optical depth of biomass burning, urban, and desert dust aerosols. *J Geophys Res* 104(D24):31333–31349 | [10.1029/1999JD900923](https://doi.org/10.1029/1999JD900923) | yes |
| boiyo2019 | Boiyo R, Kumar KR, Zhao T, Guo J (2019) A 10-year record of aerosol optical properties and radiative forcing over three environmentally distinct AERONET sites in Kenya, East Africa. *J Geophys Res Atmos* | [10.1029/2018JD029461](https://doi.org/10.1029/2018JD029461) | yes |
| wits_zebra | Wits University, Project Plains Zebra, retina sub-project | [web page](https://www.wits.ac.za/admintest/project-plains-zebra/sub-projects/retina/) | yes |
