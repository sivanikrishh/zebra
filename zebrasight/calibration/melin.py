"""Calibrate the zebra eye against Melin et al. 2016 Table 2: ``python -m zebrasight.calibration.melin``.

Writes docs/calibration.md, docs/calibration_results.json and docs/figures/calibration.png.
"""

from __future__ import annotations

import json
import math
import time

import numpy as np
import torch

from zebrasight.calibration import acuityview
from zebrasight.calibration.gratings import (Grating, blank_floor, blank_grating, grating, make_pipe, stripe_contrast,
                                             stripe_contrasts)
from zebrasight.config import REPO_ROOT, load_config

DOCS = REPO_ROOT / "docs"
DETECT = 1e-3  # output Michelson contrast at the grating frequency
LIGHTS = ("day", "dusk", "night")


def detection_distance(pipe, g: Grating, light: str, d_ref: float, k_hi: int = 15, k_lo: int = -40,
                       noise: bool = False, image_id: str = "cal", confirm: int = 3) -> dict:
    """Largest distance d_ref * 1.01^k (k from k_hi down to k_lo) at which the fundamental survives.

    Scans from far to near and stops at the first distance detected ``confirm`` times in a row
    (one detection is enough without noise). With noise the threshold is max(1e-3, 3 x floor),
    the floor being the same measurement on a uniform field at the grating's mean level.
    """
    need = confirm if noise else 1
    run, trace = 0, []
    for k in range(k_hi, k_lo - 1, -1):
        d = d_ref * 1.01**k
        c, info = stripe_contrast(pipe, g, light, d, image_id)
        floor = blank_floor(pipe, g, light, d, image_id + "-blank") if noise else 0.0
        thr = max(DETECT, 3.0 * floor)
        hit = c > thr
        trace.append((round(d, 3), c, floor))
        run = run + 1 if hit else 0
        if run >= need:
            d_det = d_ref * 1.01 ** (k + need - 1)
            return {"distance_m": d_det, "k": k + need - 1, "contrast": c, "threshold": thr,
                    "display_limited": info["display_limited"], "trace_tail": trace[-5:]}
    return {"distance_m": float("nan"), "k": None, "trace_tail": trace[-5:]}


def detection_distance_bisect(pipe, g: Grating, light: str, d_ref: float, noise: bool = False, n_seeds: int = 5,
                              image_id: str = "cal", tol: float = 0.01) -> dict:
    """Largest detected distance by bisection on log distance, to ``tol`` (1%).

    Detected at d: median contrast over ``n_seeds`` noise seeds (one batched run; each image has
    its own noise seed) > max(1e-3, 3 x median of the same measurement on a uniform field at the
    grating's mean level, also over n_seeds). Without noise one seed and threshold 1e-3.
    Bracket: hi starts at 1.15 x d_ref (raised if still detected), lo at 0.3 x d_ref (halved
    until detected). Assumes detection is monotonic in distance.
    """
    n = n_seeds if noise else 1
    calls = []

    def detected(d: float) -> bool:
        cs, info = stripe_contrasts(pipe, g, light, d, image_id, n)
        thr = DETECT
        floor = 0.0
        if noise:
            fl, _ = stripe_contrasts(pipe, blank_grating(g), light, d, image_id + "-blank", n)
            floor = float(np.median(fl))
            thr = max(DETECT, 3.0 * floor)
        c = float(np.median(cs))
        calls.append({"d": d, "median_contrast": c, "floor": floor, "hit": c > thr})
        return c > thr

    hi = 1.15 * d_ref
    for _ in range(4):
        if not detected(hi):
            break
        hi *= 1.5
    else:
        return {"distance_m": float("nan"), "calls": calls, "note": "still detected at upper bracket"}
    lo = 0.3 * d_ref
    while not detected(lo):
        hi, lo = lo, lo / 2
        if lo < 0.01 * d_ref:
            return {"distance_m": float("nan"), "calls": calls, "note": "not detected down to 1% of Melin"}
    while hi / lo > 1.0 + tol:
        mid = math.sqrt(lo * hi)
        if detected(mid):
            lo = mid
        else:
            hi = mid
    return {"distance_m": lo, "bracket_hi_m": hi, "n_runs": len(calls) * (2 if noise else 1), "calls": calls}


def table1(results_rows):
    lines = ["| Species | Stripe | Width (cm) | Light | Ours (m) | Melin (m) | Error |", "|---|---|---|---|---|---|---|"]
    for r in results_rows:
        lines.append(f"| {r['species']} | {r['stripe']} | {r['width_cm']:.2f} | {r['light']} | {r['ours_m']:.2f} | "
                     f"{r['melin_m']:.2f} | {r['err_pct']:+.1f}% |")
    return "\n".join(lines)


def calibrate(rows, pipe, contrast=1.0, wavy=False, noise=False, k_lo=-40, tag="cal", search="scan", n_seeds=5):
    """search="scan": 1% steps from 1.15 x Melin down (round 1). "bisect": bisection, n_seeds with noise."""
    out = []
    for row in rows:
        g = grating(row.width_cm, contrast=contrast, wavy=wavy)
        for light in LIGHTS:
            melin = float(row[f"{light}_m"])
            image_id = f"{tag}-{row.species}-{row.stripe}-{light}"
            if search == "bisect":
                res = detection_distance_bisect(pipe, g, light, melin, noise=noise, n_seeds=n_seeds, image_id=image_id)
            else:
                res = detection_distance(pipe, g, light, melin, k_lo=k_lo, noise=noise, image_id=image_id)
            out.append({"species": row.species, "stripe": row.stripe, "width_cm": float(row.width_cm), "light": light,
                        "ours_m": res["distance_m"], "melin_m": melin,
                        "err_pct": 100.0 * (res["distance_m"] / melin - 1.0),
                        "display_limited": res.get("display_limited")})
    return out


def acuity_check(pipe, distance_m: float = 50.0) -> dict:
    """Day: full-contrast gratings at 23.3 cpd (visible) and 26 cpd (gone)."""
    out = {}
    for cpd in (23.3, 26.0):
        period_cm = 100.0 * distance_m * math.tan(math.radians(1.0 / cpd))
        c, _ = stripe_contrast(pipe, grating(period_cm / 2), "day", distance_m, f"acuity-{cpd}")
        out[cpd] = c
    return out


def contrast_curves(pipe, width_cm: float = 5.23, n: int = 90):
    d = np.geomspace(2.0, 300.0, n)
    g = grating(width_cm)
    return d, {light: [stripe_contrast(pipe, g, light, float(x), "curve")[0] for x in d] for light in LIGHTS}


def figure(d, curves, melin_row, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colours = {"day": "#2a78d6", "dusk": "#eb6834", "night": "#1baf7a"}  # validated categorical slots 1-3
    styles = {"day": "-", "dusk": "--", "night": "-."}
    floor = 1e-5
    plt.rcParams.update({"font.size": 18, "axes.titlesize": 22, "axes.labelsize": 20, "legend.fontsize": 16})
    fig, ax = plt.subplots(figsize=(12, 7))
    for light in LIGHTS:
        y = np.asarray(curves[light], dtype=float)
        below = np.nonzero(y < floor)[0]
        if below.size:  # draw the drop to the axis floor, then stop (no flat line of zeros)
            first = below[0]
            y = y.copy()
            y[first] = floor
            y[first + 1 :] = np.nan
        ax.plot(d, y, styles[light], color=colours[light], lw=2.5, label=f"ZebraEye, {light}")
        dm = float(melin_row[f"{light}_m"])
        ax.axvline(dm, color=colours[light], lw=1.5, ls=":", alpha=0.9)
        ax.text(dm * 1.04, 1.6e-5, f"Melin {light}\n{dm:.2f} m", color="#333333", fontsize=14, va="bottom")
    ax.axhline(DETECT, color="#888888", lw=1, ls="--")
    ax.text(2.2, DETECT * 1.3, "detection threshold 0.001", color="#555555", fontsize=13)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(floor, 2.0)  # a square wave's fundamental starts at 4/pi = 1.27
    ax.set_xlim(2, 300)
    ax.set_xlabel("Viewing distance (m)")
    ax.set_ylabel("Stripe contrast after zebra eye")
    ax.set_title("ZebraEye vs Melin et al. 2016\nplains zebra flank stripe, 5.23 cm wide, full contrast")
    ax.grid(True, which="major", color="#dddddd", lw=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(loc="lower left", frameon=False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def acuityview_comparison(acuity=23.3, b_frac=0.10, c_peak=50.0, taper=0.02, dpp=0.01):
    curves = acuityview.compare_curves(acuity, b_frac, c_peak)
    # test image: wavy full-contrast coat, 512 x 512 at 0.01 deg/px, stripe period 6 px (~16.7 cpd) and 20 px (5 cpd)
    n = 512
    yy, xx = torch.meshgrid(torch.arange(n, dtype=torch.float64), torch.arange(n, dtype=torch.float64), indexing="ij")
    out = {"mtf": curves}
    for period in (20.0, 6.0):
        img = ((torch.sin(2 * math.pi * (xx + 0.15 * period * torch.sin(2 * math.pi * yy / (3 * period))) / period) > 0)
               .double())[None, None] * 0.9 + 0.05
        a = acuityview.acuityview_filter(img, dpp, acuity)
        o = acuityview.ours_acuity_only(img, dpp, acuity, b_frac, c_peak, taper)
        sl = (..., slice(64, -64), slice(64, -64))
        ca, co = float(a[sl].std() / a[sl].mean()), float(o[sl].std() / o[sl].mean())
        rmse = float(((a[sl] - o[sl]) ** 2).mean().sqrt() / img[sl].mean())
        out[f"period_{int(period)}px"] = {"cpd": 1 / (period * dpp), "rms_contrast_acuityview": ca,
                                          "rms_contrast_ours": co, "rmse_rel_mean": rmse}
    return out


def main() -> None:
    t0 = time.perf_counter()
    rows = list(load_config("melin_table2").rows)
    pipe = make_pipe()

    acuity_only = calibrate(rows, pipe)
    passed = sum(abs(r["err_pct"]) <= 5.0 for r in acuity_only)
    print(f"acuity-only: {passed}/{len(acuity_only)} within 5%")

    checks = acuity_check(pipe)
    print("day acuity check:", checks)

    d, curves = contrast_curves(pipe)
    figure(d, curves, rows[0], DOCS / "figures" / "calibration.png")

    av = acuityview_comparison()
    print("acuityview:", json.dumps(av, indent=1)[:600])

    full = make_pipe(full_eye=True)
    variants = {
        "contrast 0.6": calibrate(rows, pipe, contrast=0.6, tag="c06", search="bisect"),
        "contrast 0.6 + wavy": calibrate(rows, pipe, contrast=0.6, wavy=True, tag="c06w", search="bisect"),
        "full eye (0.6, wavy, noise)": calibrate(rows, full, contrast=0.6, wavy=True, noise=True, tag="full",
                                                 search="bisect", n_seeds=5),
    }
    for k, v in variants.items():
        print(k, "mean ratio to Melin:", np.nanmean([r["ours_m"] / r["melin_m"] for r in v]))

    results = {"acuity_only": acuity_only, "acuity_check": {str(k): v for k, v in checks.items()},
               "acuityview": av, "variants": variants,
               "curves": {"distance_m": d.tolist(), **{k: list(map(float, v)) for k, v in curves.items()}},
               "runtime_s": time.perf_counter() - t0}
    (DOCS / "calibration_results.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    write_markdown(results)
    print(f"done in {results['runtime_s']:.0f} s")


def write_markdown(res) -> None:
    from zebrasight.calibration.report import render

    (DOCS / "calibration.md").write_text(render(res), encoding="utf-8")


def figure_only() -> None:
    """Redraw the figure from docs/calibration_results.json (no simulation)."""
    res = json.loads((DOCS / "calibration_results.json").read_text(encoding="utf-8"))
    c = res["curves"]
    figure(np.asarray(c["distance_m"]), {k: c[k] for k in LIGHTS}, load_config("melin_table2").rows[0],
           DOCS / "figures" / "calibration.png")


if __name__ == "__main__":
    import sys

    figure_only() if "--figure-only" in sys.argv else main()
