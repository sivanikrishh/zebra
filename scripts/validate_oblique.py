"""STUB (not run yet): validate synthetic foreshortening against real GZGC oblique views.

    python scripts/validate_oblique.py [--phi 60]

Question: does compressing a LEFT side view by cos(phi) (geometry stage, c4-c6) reproduce the
stripe-width distribution of real oblique photos (GZGC viewpoints "frontleft" / "backleft")?

Planned method:
1. Pairs: zebras that have both a "left" view and a "frontleft"/"backleft" view
   (zebrasight.data.gzgc.load_annotations; ~400 / ~430 oblique annotations).
2. Masks: run zebrasight.data.masks on the oblique views (body mask; regions not needed).
3. Stripe widths: on the flank band of each body mask, binarise y (Otsu) and measure run lengths
   of black and white along rows, in cm via cm_per_px (oblique boxes are shorter, so the scale
   must come from the paired left view of the same animal, e.g. matched body height).
4. Compare the run-length histograms of (a) left views compressed by cos(phi) for phi in
   30..75 deg and (b) the real obliques: fit the phi that minimises a histogram distance
   (Earth mover's / KS), report it with a bootstrap CI over zebras.
5. Expected caveats: real obliques also show the body curving away (non-planar flank) and
   perspective; the planar cos(phi) model should underestimate compression near the rump.
Output: docs/figures/oblique_validation.png and a table in docs/model_parameters.md.
"""

from __future__ import annotations


def main() -> None:
    raise SystemExit("validate_oblique.py is a stub: see the docstring for the planned method")


if __name__ == "__main__":
    main()
