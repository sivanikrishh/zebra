"""Three headline slide charts from app_data/results.json (reads results only, computes nothing).

    python -m zebrasight.plots.headline [results.json] [out_dir]

1. headline_1_position_vs_style.png : AUC at 2 m, clear day, every observer with 95% CI
   (face | flank). Stripe POSITION observers vs stripe STYLE observers vs raw-pixel baseline.
2. headline_2_distance.png          : flank, clear day, AUC vs distance for the best position
   observer, a second position observer, and the best style observer.
3. headline_3_light.png             : flank, MegaDescriptor, AUC vs distance in day / dusk / night.

Labels are placed by hand for the current results; re-check the PNGs if results.json changes.
Colour = observer kind (position blue, style orange, baseline grey), the same in every chart;
chart 3 uses an ordinal one-hue blue ramp (lighter = brighter light). Palettes validated with
the dataviz validator (categorical: all pass, aqua not used; ordinal: all pass). Every series
is direct-labelled, so identity never depends on colour alone.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

from zebrasight.config import REPO_ROOT

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RESULTS = REPO_ROOT / "app_data" / "results.json"
OUT_DIR = REPO_ROOT / "app_data" / "figures"

# palette (dataviz reference instance, light mode)
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
POSITION = "#2a78d6"  # categorical slot 1
STYLE = "#eb6834"  # categorical slot 2
BASELINE = "#898781"
LIGHT_RAMP = {"c0": "#86b6ef", "c8": "#2a78d6", "c9": "#104281"}  # ordinal blue 250/450/650
KIND_COLOUR = {"positional": POSITION, "summary_statistics": STYLE, "baseline": BASELINE}

SHORT = {
    "megadesc": "Learned re-ID (MegaDescriptor)",
    "hotspotter": "Local stripe matcher (HotSpotter)",
    "registered_layout": "Aligned stripe layout",
    "spatial_nn": "Stripe statistics + coarse layout",
    "interpretable_nn": "Stripe statistics F1-F10",
    "raw_pixel": "Raw pixels",
}
GROUP_TITLE = {"positional": "Stripe POSITION", "summary_statistics": "Stripe STYLE", "baseline": "Baseline"}


def _style_axes(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(1)
    ax.tick_params(colors=MUTED, labelcolor=INK2, labelsize=10, length=3)
    ax.grid(True, color=GRID, linewidth=0.8, linestyle="-")
    ax.set_axisbelow(True)


def _fig(w: float, h: float, ncols: int = 1):
    fig, axes = plt.subplots(1, ncols, figsize=(w, h), facecolor=SURFACE, squeeze=False)
    for ax in axes[0]:
        _style_axes(ax)
    return fig, axes[0]


def _titles(fig, title: str, subtitle: str, footer: str) -> None:
    fig.text(0.015, 0.965, title, fontsize=16, fontweight="bold", color=INK, va="top")
    fig.text(0.015, 0.895, subtitle, fontsize=11, color=INK2, va="top")
    fig.text(0.015, 0.02, footer, fontsize=8.5, color=MUTED, va="bottom")


def _curve(r: dict, region: str, cond: str, obs: str) -> list[dict]:
    for c in r["curves"]:
        if (c["region"], c["condition"], c["observer"]) == (region, cond, obs):
            return c["points"]
    raise KeyError((region, cond, obs))


def _footer(r: dict) -> str:
    return (f"{r['n_zebras']} zebras; query and gallery photos taken on different days. AUC = probability that a "
            f"same-zebra pair looks more alike than a different-zebra pair (0.5 = chance, 1 = perfect).\n"
            f"{r['bootstrap_note']}. Eye parameters are a horse proxy (no zebra acuity measurement exists).")


def _distance_axis(ax, r: dict) -> None:
    ax.set_xscale("log")
    ax.set_xlim(1.8, 230)
    ticks = [2, 5, 10, 20, 50, 100, 200]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t}" for t in ticks])
    ax.minorticks_off()
    ax.set_xlabel("Viewing distance (m, log scale)", color=INK2, fontsize=11)
    ax.set_ylim(0.4, 0.95)
    ax.set_ylabel("AUC  (same vs different zebra)", color=INK2, fontsize=11)
    lim = r["markers"]["photo_limited_max_m"]
    ax.axvspan(1.8, lim, color="#f0efec", zorder=0, linewidth=0)
    ax.text(2.0, 0.925, "Photo-limited zone:\nthe camera, not the eye,\nlimits detail here",
            fontsize=8.5, color=MUTED, va="top")
    ax.axhline(0.5, color=INK2, linewidth=1, zorder=1)
    ax.text(2.05, 0.496, "chance", fontsize=9, color=INK2, ha="left", va="top")


def _line(ax, pts: list[dict], colour: str, label: str, dashed: bool = False, band: bool = True,
          label_at: int = 0, text_xy: tuple[float, float] | None = None) -> None:
    d = [p["distance_m"] for p in pts]
    a = [p["auc"] for p in pts]
    if band:
        ax.fill_between(d, [p["auc_ci"][0] for p in pts], [p["auc_ci"][1] for p in pts],
                        color=colour, alpha=0.10, linewidth=0, zorder=2)
    ax.plot(d, a, color=colour, linewidth=2.2, linestyle=(0, (5, 3)) if dashed else "-",
            solid_capstyle="round", solid_joinstyle="round", zorder=3)
    i = label_at
    ax.plot([d[i]], [a[i]], "o", color=colour, markersize=7, markeredgecolor=SURFACE,
            markeredgewidth=1.5, zorder=4)
    tx, ty = text_xy if text_xy is not None else (d[i] * 1.06, a[i] + 0.012)
    ax.annotate(label, (d[i], a[i]), xytext=(tx, ty), textcoords="data", fontsize=10, color=INK,
                va="center", zorder=5, arrowprops=dict(arrowstyle="-", color=MUTED, linewidth=0.8,
                                                       shrinkA=2, shrinkB=5))


def chart_position_vs_style(r: dict, out: Path) -> Path:
    kinds = {o["id"]: o["kind"] for o in r["observers"]}
    order = ["megadesc", "hotspotter", "registered_layout", "spatial_nn", "interpretable_nn", "raw_pixel"]
    fig, axes = _fig(13, 6.2, ncols=2)
    for ax, region in zip(axes, ("flank", "face")):
        y, ylabels, ypos = 0.0, [], []
        prev = None
        rows = []
        for obs in order:
            if prev is not None and kinds[obs] != prev:
                y += 0.6  # gap between groups
            rows.append((obs, y))
            prev = kinds[obs]
            y += 1
        for obs, yy in rows:
            p = _curve(r, region, "c0", obs)[0]
            col = KIND_COLOUR[kinds[obs]]
            lo, hi = p["auc_ci"]
            ax.plot([lo, hi], [yy, yy], color=col, linewidth=2.2, solid_capstyle="round", zorder=3)
            ax.plot([p["auc"]], [yy], "o", color=col, markersize=9, markeredgecolor=SURFACE,
                    markeredgewidth=1.5, zorder=4)
            ax.text(hi + 0.012, yy, f"{p['auc']:.2f}", fontsize=10, color=INK, va="center")
            ylabels.append(SHORT[obs])
            ypos.append(yy)
        # group headers
        firsts = {}
        for obs, yy in rows:
            firsts.setdefault(kinds[obs], yy)
        for k, yy in firsts.items():
            ax.text(0.385, yy - 0.55, GROUP_TITLE[k], fontsize=9.5, fontweight="bold", zorder=5,
                    color=KIND_COLOUR[k] if k != "baseline" else INK2, va="center",
                    bbox=dict(facecolor=SURFACE, edgecolor="none", pad=1.5))
        ax.set_yticks(ypos)
        ax.set_yticklabels(ylabels if region == "flank" else [""] * len(ylabels))
        ax.invert_yaxis()
        ax.set_xlim(0.38, 1.0)
        ax.set_ylim(rows[-1][1] + 0.7, -1.1)
        ax.axvline(0.5, color=INK2, linewidth=1, zorder=1)
        ax.text(0.505, rows[-1][1] + 0.6, "chance", fontsize=9, color=INK2, va="bottom")
        ax.grid(False, axis="y")
        ax.set_xlabel("AUC at 2 m, clear day (dot = estimate, bar = 95% CI)", color=INK2, fontsize=10.5)
        ax.set_title(region.capitalize(), loc="left", fontsize=13, color=INK, fontweight="bold", pad=10)
    _titles(fig, "Identity is in WHERE the stripes are, not WHAT they look like",
            "Flank: observers that match stripe positions sit well above chance; stripe statistics sit on it. "
            "Faces carry little identity in our photos.", _footer(r))
    fig.subplots_adjust(left=0.20, right=0.98, top=0.78, bottom=0.19, wspace=0.08)
    return _save(fig, out)


def chart_distance(r: dict, out: Path) -> Path:
    fig, (ax,) = _fig(11, 6.2)
    _distance_axis(ax, r)
    _line(ax, _curve(r, "flank", "c0", "spatial_nn"), STYLE, "Stripe statistics + layout (style)",
          band=True, label_at=8, text_xy=(30, 0.43))
    _line(ax, _curve(r, "flank", "c0", "hotspotter"), POSITION, "Local stripe matcher (position)",
          dashed=True, band=False, label_at=2, text_xy=(2.3, 0.765))
    _line(ax, _curve(r, "flank", "c0", "megadesc"), POSITION, "Learned re-ID, MegaDescriptor (position)",
          band=True, label_at=4, text_xy=(11, 0.885))
    pts = _curve(r, "flank", "c0", "megadesc")
    for di, off in ((6, (6, 8)), (9, (6, 8)), (12, (6, 8))):
        p = pts[di]
        ax.annotate(f"{p['distance_m']:.0f} m: {p['auc']:.2f}", (p["distance_m"], p["auc"]), xytext=off,
                    textcoords="offset points", fontsize=9, color=INK2, ha="left" if off[0] > 0 else "right")
        ax.plot([p["distance_m"]], [p["auc"]], "o", color=POSITION, markersize=6,
                markeredgecolor=SURFACE, markeredgewidth=1.2, zorder=4)
    _titles(fig, "The flank barcode is readable to ~10-15 m and gone by ~150 m",
            "Flank, clear day. Shaded band = 95% CI. Stripe statistics never rise above chance at any distance.",
            _footer(r))
    fig.subplots_adjust(left=0.08, right=0.98, top=0.82, bottom=0.19)
    return _save(fig, out)


def chart_light(r: dict, out: Path) -> Path:
    fig, (ax,) = _fig(11, 6.2)
    _distance_axis(ax, r)
    labels = {c["id"]: c["label"] for c in r["conditions"]}
    for cond, at, xy in (("c0", 7, (22, 0.80)), ("c8", 6, (6.5, 0.64)), ("c9", 1, (3.2, 0.69))):
        _line(ax, _curve(r, "flank", cond, "megadesc"), LIGHT_RAMP[cond], labels[cond],
              band=True, label_at=at, text_xy=xy)
    day, dusk = _curve(r, "flank", "c0", "megadesc"), _curve(r, "flank", "c8", "megadesc")
    _titles(fig, "Dusk roughly halves the range; at night identity is almost gone",
            f"Flank, learned re-ID (MegaDescriptor). Shaded band = 95% CI. AUC ~0.79 is reached at "
            f"{day[6]['distance_m']:.0f} m by day but already at {dusk[4]['distance_m']:.1f} m at dusk.", _footer(r))
    fig.subplots_adjust(left=0.08, right=0.98, top=0.82, bottom=0.19)
    return _save(fig, out)


def _save(fig, out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    return out


def main() -> None:
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else RESULTS
    out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else OUT_DIR
    r = json.loads(src.read_text(encoding="utf-8"))
    if r.get("status") != "final":
        print(f"WARNING: results status is {r.get('status')!r}, not 'final'")
    print(chart_position_vs_style(r, out_dir / "headline_1_position_vs_style.png"))
    print(chart_distance(r, out_dir / "headline_2_distance.png"))
    print(chart_light(r, out_dir / "headline_3_light.png"))


if __name__ == "__main__":
    main()
