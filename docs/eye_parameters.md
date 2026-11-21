# Zebra-eye parameters

Machine-readable source: `configs/eye.yaml` (keep both in sync).

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
| Pigment template | Govardovskii A1 | — | — | B | Govardovskii et al. 2000 |
| Rod peak | 500 | — (fixed) | nm | C | generic mammalian rhodopsin |
| Colour threshold | 0.02 | — | cd/m² | A | Roth, Balkenius & Kelber 2008. Below it the model is rod-only with no S channel. Dusk (0.16) is above it, so colour stays on at dusk |
| Neutral point | 480 | test only | nm | A | Geisbauer et al. 2004; Roth et al. 2007 |
| Chromatic acuity | 0.5 | 0.3 – 1.0 | × luminance acuity | C | no horse data |
| CSF form | log-parabola | — | — | B | Melin 2016 method (cat CSF of Pasternak & Merigan 1981, peak shifted by acuity, slope constant) |
| CSF constraint | S(acuity) = 1 | — | — | B | Melin 2016 |
| CSF peak frequency b | 0.10 | 0.08 – 0.15 | × acuity | C | Melin did not report a, b, c |
| CSF peak sensitivity c | 50 | 20 – 100 | 1/contrast | C | same |
| Cat reference acuity | 5.81 | — | cpd | B | Melin 2016 |
| Weber fraction (brightness) | 0.42 – 0.45 | upper bound for noise | — | A | Geisbauer et al. 2004 (large-field task, 2 horses) |
| Luminance discrimination limit | 0.2 | upper bound for noise | log units | A | Macuda & Timney 1999: near chance below 0.2 log units (large field) |
| Streak elevation | 0 | — | deg | A | Timney & Keil 1992 (acuity limited by streak ganglion density); Guo & Sugita 2000 |
| Streak peak density | 4000 | — | cells/mm² | A | Guo & Sugita 2000 |
| Periphery floor | 0.16 | — | × streak acuity | A | Harman et al. 1999: 2.7 / 16.5 cpd |
| Streak vertical σ | 5 | 3 – 10 | deg | C | — |
| Azimuth falloff at 90° | 0.50 | 0.30 – 0.70 | × peak | C | linear from az 0 |
| Integration time, day | 30 | 15 – 60 | ms | C | no horse flicker-fusion value found; human cones ~60 Hz |
| Integration time, night | 100 | — (fixed) | ms | C | human rods ~15 Hz |
| Noise model | Poisson + Gaussian | sigma TBD | — | C | — |
| Working grid | 0.01 | — (fixed) | deg/px | C | Nyquist 50 cpd > max acuity 25 |
| Photo colour | uncalibrated sRGB | — | — | C | Melin used a GretagMacbeth chart; we have none, so cone catches are approximate |

## Notes and caveats

- **Melin CSF shift.** The PLOS text gives the shift as `b_species = b_cat + (VA_species − VA_cat)`
  and says the parabola is "in log scale", but it never states whether b and VA are in log units.
  Our b = 0.10 × acuity is equivalent to shifting log b by log(acuity ratio). That is the brief's
  reading, not a quote from the paper.
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
| wits_zebra | Wits University, Project Plains Zebra, retina sub-project | [web page](https://www.wits.ac.za/admintest/project-plains-zebra/sub-projects/retina/) | yes |
