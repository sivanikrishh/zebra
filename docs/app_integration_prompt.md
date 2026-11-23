# Prompt for the 3D application's Claude session (paste everything below the line)

---

# Integrate three validated ZebraSight research features into our 3D application

You are adding three features to our existing 3D ZebraSight application. They were designed and scientifically
validated in a 2D research prototype. Reproduce their **behaviour and numbers**, using our app's own 3D design.
Do not change the science.

## Read first
1. In the zebrasight repo, branch `app-prototype`: `docs/app_integration.md` (features, scientific rules, code
   to reuse, data contract, validation, limitations). It is the source of truth.
2. Open the prototype: unzip `zebrasight_app_prototype.zip` and open `index.html` in a browser (offline). Try
   every tab and control before you design anything.
3. Then inspect our 3D application: its architecture, rendering, how it loads data, and where these features fit.
   Report the integration plan to me before coding.

## The three features
1. **Recognition horizon:** the zebra walks away from 2 m to 200 m. Show the zebra-eye texture for that distance
   and light, stretched over the same coat patch (fewer pixels at distance IS the effect). Show one number:
   identification accuracy (top-1 in a herd of 20) with its 95% CI and chance 5%. Lights: clear day, dusk,
   night. Regions: Face, Flank, and **Full body = "Results pending"** (no values).
2. **Stripe identity explorer:** two original photos (a later-day and an earliest-day photo) side by side, with
   "same zebra, other day" and "different zebra". Mode A: lines between matching stripe features. Mode B: the
   study's stripe map (black stripes, centre-lines, junctions, endings) plus the four measured characteristics
   (width, spacing, junctions, black fraction) against all 165 photos. The study result: arrangement (AUC 0.84)
   vs characteristics (AUC 0.52), flank, 2 m, clear day.
3. **Pattern Lab:**
   - **Exp 1, move away:** the zebra recedes along the depth axis (use our 3D scene). At each of the 16 measured
     distances show the real rank of the true zebra among the 50 known zebras and its score against its
     earliest-day reference, both seen through the zebra eye at that distance (like with like), plus the study
     result, labelled separately.
   - **Exp 2, change the stripes:** thickness, contrast, arrangement, spacing. Every step is a real modified photo
     with real MegaDescriptor and HotSpotter inference (precomputed in `lab.js`). Show rank among 50, similarity
     to the unmodified photo, and the rank across all steps.
   - **Exp 3, fool the observer:** show "needs horse photos". Do not fake it.

## Data
Load the prototype's JSON files (`data.js` → `DEMO`, `explorer.js` → `EX`, `lab.js` → `LAB`) and their image
folders; the schemas are in `docs/app_integration.md` §4. If our app needs another format, convert them with a
script; do not retype or edit numbers. For live inference on any slider combination, call
`zebrasight.patternlab` from a small Python service (`Observers(region).identify`, `modify`) instead of the
precomputed steps. Keep it optional.

## Non-negotiable rules
- Only stored or real-inference numbers. No interpolation between distances, no invented or "illustrative" values.
- Never call AUC a percentage correct. Keep identification accuracy, AUC and pair similarity clearly distinct.
- Label "study result (all 50 zebras)" separately from "this pair / experimental".
- These are computer observers, not zebra brains. No claims about real zebra behaviour.
- Full body shows "Results pending" until real full-body results are supplied; never substitute flank or face.
- When every zebra scores the same (`tied_with >= n_known - 1`), show "No information", not "tied for #1".
- Photo-derived images are GZGC CC BY-NC-ND 2.0: internal / non-commercial only. Never publish them.

## Done means
- All three features work in the 3D app with the same numbers as the prototype. Spot-check at least 10 values
  per feature against the prototype and list them.
- Full body pending everywhere it appears; Exp 3 honest placeholder.
- A short report: files changed, how data is loaded, what was tested, limitations.
- Do not commit or push until I have reviewed it.
