# To do (not forgotten, deliberately deferred)

- **Night condition c9: label and credibility.** configs/grid.yaml and docs/contracts.md define c9 as a
  moonless night (1.6e-5 cd/m2), but app_data/results.json, the findings document and the slide deck
  call it "Night (moonlight)". Fix the label everywhere AND run credibility checks on the night results
  (photon-noise level, rod-only rendering, whether the 0.62 AUC at 2 m is plausible). Deferred on
  2026-11-23 by the team; do it in one pass.

## Rules to keep (not tasks)

- **Never compare raw similarity across distances or light levels.** Blurred percepts all look alike,
  so similarity INFLATES with distance (identify.py test photo: true match 0.62 at 2 m, but unrelated
  zebras scored 0.72-0.79 at 30 m). Compare ranks or AUC within one distance and light. Any "unknown
  zebra" threshold must be calibrated separately per distance and light, on gallery pairs only.
