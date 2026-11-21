"""Resolution distance of the 5.23 cm plains-zebra stripe vs atmospheric extinction beta.

    python -m zebrasight.calibration.haze_curve

Calibration method (docs/calibration.md): full-contrast black/white square-wave grating, day
light, noise and motion off, detection = output contrast of the fundamental > 1e-3, distance by
bisection to 1%. Here the atmosphere stage is ON (Koschmieder + Angstrom), for haze (alpha 1.3)
and dust (alpha 0.2), beta_550 from clear (0.0002/m) to dust storm (0.03/m).
Writes docs/figures/haze_resolution.png and docs/haze_resolution.json.
"""

from __future__ import annotations

import json
import math

import matplotlib
import numpy as np
import torch
from omegaconf import OmegaConf

from zebrasight.calibration.gratings import fundamental_contrast, grating
from zebrasight.config import REPO_ROOT, load_config, make_cond
from zebrasight.pipeline import Pipeline

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

DETECT = 1e-3
WIDTH_CM = 5.23
MELIN_DAY_M = 139.64
BETAS = np.geomspace(0.0002, 0.03, 11)
MARKS = {"c0 clear": 0.0002, "c1 hazy": 0.002, "c2 dusty": 0.01, "c3 dust storm": 0.03}


def make_pipe() -> Pipeline:
    eye = load_config("eye")
    OmegaConf.update(eye, "eye_stages.noise", False, merge=False)
    OmegaConf.update(eye, "eye_stages.motion", False, merge=False)
    return Pipeline(list(load_config("grid").all_stages), eye_cfg=eye)


def contrast_at(pipe, g, d: float, beta: float, aerosol: str) -> float:
    cond = make_cond("c0", d, beta_haze=beta, aerosol=aerosol)
    p = pipe.run(g.img, g.cm_per_px, torch.ones(1, 1, *g.img.shape[-2:]), cond, meta={"image_id": "haze"},
                 img_is_linear=True)
    period_deg = math.degrees(math.atan(g.period_cm / 100.0 / d))
    f_px = p.deg_per_px / period_deg
    return 0.0 if f_px >= 0.5 else fundamental_contrast(p.y, f_px)


def detection_distance(pipe, g, beta: float, aerosol: str, tol: float = 0.01) -> float:
    lo, hi = 1.0, 1.15 * MELIN_DAY_M
    if contrast_at(pipe, g, lo, beta, aerosol) <= DETECT:
        return float("nan")
    while hi / lo > 1 + tol:
        mid = math.sqrt(lo * hi)
        if contrast_at(pipe, g, mid, beta, aerosol) > DETECT:
            lo = mid
        else:
            hi = mid
    return lo


def main() -> None:
    import sys

    js = REPO_ROOT / "docs" / "haze_resolution.json"
    if "--plot-only" in sys.argv:
        saved = json.loads(js.read_text(encoding="utf-8"))
        res = {a: saved[a] for a in ("haze", "dust")}
    else:
        pipe = make_pipe()
        g = grating(WIDTH_CM)
        res = {a: [detection_distance(pipe, g, float(b), a) for b in BETAS] for a in ("haze", "dust")}
        js.write_text(json.dumps({"beta_550": BETAS.tolist(), **res, "width_cm": WIDTH_CM}, indent=1), encoding="utf-8")
    plt.rcParams.update({"font.size": 18, "axes.labelsize": 20, "axes.titlesize": 21, "legend.fontsize": 15})
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(BETAS, res["haze"], "-o", color="#2a78d6", lw=2.5, ms=7, label="haze (Ångström α 1.3)")
    ax.plot(BETAS, res["dust"], "--s", color="#eb6834", lw=2.5, ms=7, label="dust (Ångström α 0.2)")
    ax.plot(BETAS, 3.912 / BETAS, ":", color="#888888", lw=1.8, label="Koschmieder visual range 3.912/β")
    ax.axhline(MELIN_DAY_M, color="#1baf7a", lw=1.5, ls="-.", label=f"acuity limit, clear air ({MELIN_DAY_M} m, Melin 2016)")
    for name, b in MARKS.items():
        ax.axvline(b, color="#cccccc", lw=1, zorder=0)
        ax.text(b * 1.05, 380, name, rotation=90, va="top", fontsize=13, color="#555555")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(40, 400)
    ax.set_xlabel("Atmospheric extinction β at 550 nm (1/m)")
    ax.set_ylabel("Stripe resolution distance (m)")
    ax.set_title("Haze and dust shorten how far a zebra resolves another zebra's stripes\n"
                 "plains zebra flank stripe, 5.23 cm wide, full contrast, daylight")
    ax.grid(True, which="major", color="#e5e5e5")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(loc="lower left", frameon=False)
    fig.tight_layout()
    out = REPO_ROOT / "docs" / "figures" / "haze_resolution.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    for a in res:
        print(a, [f"{b:.4g}:{d:.1f}" for b, d in zip(BETAS, res[a])])
    print(out)


if __name__ == "__main__":
    main()
