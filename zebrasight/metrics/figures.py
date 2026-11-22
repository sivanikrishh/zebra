"""Slide figures from outputs/metrics/*: python -m zebrasight.metrics.figures

Writes outputs/figures/:
  auc_face_flank_<cond>.png  face | flank panels, all observers, AUC vs distance (one per light)
  dstar_by_condition.png     d*_AUC per condition x region, dot + whisker (bootstrap)
  family_heatmap_<cond>.png  single-clue-family AUC x distance, face and flank; below a gap, whole observers
Statistics only; every figure states "95% CI, 200 bootstrap resamples over zebras (final)".
"""

from __future__ import annotations

import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from zebrasight.config import REPO_ROOT  # noqa: E402

MET = REPO_ROOT / "outputs" / "metrics"
FIG = REPO_ROOT / "outputs" / "figures"
COND_LABEL = {"c0": "clear day", "c8": "dusk", "c9": "night (moonlight)"}
PHOTO_LIMITED_MAX_M = 9.283
BOOT_NOTE = "95% CI: 200 bootstrap resamples over zebras (final)"
OBS = [  # id, label, colour, style, width
    ("megadesc", "learned re-ID (MegaDescriptor)", "#1b6ca8", "-", 2.8),
    ("hotspotter", "local matcher (HotSpotter-style)", "#7a3fb0", "-", 2.2),
    ("registered_layout", "registered layout (aligned black fraction)", "#2a9d8f", "-", 1.8),
    ("spatial_nn", "stripe statistics + layout", "#d9822b", "-", 1.8),
    ("interpretable_nn", "stripe statistics F1-F10", "#e0a96d", "-", 1.4),
    ("raw_pixel", "raw pixels", "#666666", "--", 1.4),
]
FAMILY_LABEL = {"F1": "width", "F2": "black fraction", "F3": "curvature", "F4": "orientation", "F5": "junctions",
                "F6": "endings", "F7": "spacing", "F8": "shadow stripes", "F9": "contrast", "F10": "topology",
                "L": "coarse layout", "M": "minutiae map"}

plt.rcParams.update({"font.size": 15, "axes.titlesize": 17, "axes.labelsize": 16, "legend.fontsize": 12})


def size_rule() -> dict:
    p = REPO_ROOT / "outputs" / "features" / "size_limits_c0.csv"
    if not p.exists():
        return {}
    sl = pd.read_csv(p)
    return {r: float(g.d_size_rule_m.replace(np.inf, np.nan).median()) for r, g in sl.groupby("region")}


def originals() -> pd.DataFrame:
    p = MET / "originals.csv"
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


def _decorate(ax, region, sr):
    ax.axvspan(1.7, PHOTO_LIMITED_MAX_M * 1.08, color="#999999", alpha=0.15, lw=0)
    ax.text(2.05, 0.985, "photo-resolution\nlimited", fontsize=11, color="#555555", va="top")
    ax.axhline(0.5, color="#333333", lw=1)
    for yv in (0.80, 0.95):
        ax.axhline(yv, color="#888888", lw=1, ls="--")
        ax.text(225, yv, f"{yv:.2f}", fontsize=11, color="#555555", va="center")
    if sr.get(region) and math.isfinite(sr[region]):
        ax.axvline(sr[region], color="#444444", lw=1.2, ls="-.")
        ax.text(sr[region] * 0.96, 0.42, "size rule bites\n(median photo)", rotation=90, ha="right", va="bottom",
                fontsize=10, color="#444444")
    ax.set_xscale("log")
    ax.set_xlim(1.7, 215)
    ax.set_ylim(0.4, 1.0)
    ax.set_xticks([2, 5, 10, 20, 50, 100, 200])
    ax.set_xticklabels(["2", "5", "10", "20", "50", "100", "200"])
    ax.set_xlabel("Viewing distance (m)")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def auc_chart(cond: str, m: pd.DataFrame, sr: dict, orig: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=True)
    for ax, region in zip(axes, ("face", "flank")):
        for oid, lab, col, ls, lw in OBS:
            a = m[(m.region == region) & (m.observer == oid) & (m.metric == "auc")].sort_values("distance_m")
            if a.empty:
                continue
            ax.plot(a.distance_m, a.value, ls, color=col, lw=lw, label=lab)
            if oid in ("megadesc", "hotspotter"):
                ax.fill_between(a.distance_m, a.ci_low, a.ci_high, color=col, alpha=0.15, lw=0)
            if not orig.empty:
                o = orig[(orig.region == region) & (orig.observer == oid)]
                if len(o):
                    o = o.iloc[0]
                    ax.errorbar([1.85], [o.auc], yerr=[[o.auc - o.auc_ci_low], [o.auc_ci_high - o.auc]], fmt="D",
                                color=col, ms=8, capsize=3)
        if region == "face":
            a = m[(m.region == "face_owlv2") & (m.observer == "megadesc") & (m.metric == "auc")].sort_values("distance_m")
            if len(a):
                ax.plot(a.distance_m, a.value, ":", color="#1b6ca8", lw=2.2, label="learned, OWLv2-detected faces only")
        _decorate(ax, region, sr)
        ax.set_title(region.capitalize())
    axes[0].set_ylabel("ROC-AUC, same zebra vs different zebra")
    axes[0].text(1.9, 0.405, "◆ = original photo (no eye model)", fontsize=11, color="#333333")
    h, l = axes[0].get_legend_handles_labels()
    h2, l2 = axes[1].get_legend_handles_labels()
    seen = dict(zip(l, h)) | {k: v for k, v in zip(l2, h2)}
    fig.legend(seen.values(), seen.keys(), loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(f"Can a zebra tell individuals apart? {COND_LABEL.get(cond, cond)}, 50 zebras, different-day photos\n"
                 f"{BOOT_NOTE}; 0.5 = chance; usable = lower CI ≥ 0.80", fontsize=16)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    out = FIG / f"auc_face_flank_{cond}.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def dstar_chart(tabs: dict):
    fig, ax = plt.subplots(figsize=(12, 6.5))
    ylab, y = [], 0
    colours = {"face": "#d9822b", "flank": "#1b6ca8"}
    for cond, tab in tabs.items():
        for region in ("face", "flank"):
            for oid in ("megadesc", "hotspotter"):
                r = tab[(tab.region == region) & (tab.observer == oid)]
                if r.empty:
                    continue
                r = r.iloc[0]
                lo, hi = max(r.dstar_auc_ci_low_m, 1.0), max(r.dstar_auc_ci_high_m, 1.0)
                mk = "o" if oid == "megadesc" else "s"
                ax.plot([lo, hi], [y, y], color=colours[region], lw=3, alpha=0.6)
                if math.isfinite(r.dstar_auc_m):
                    ax.plot([r.dstar_auc_m], [y], mk, color=colours[region], ms=11)
                else:
                    ax.text(1.15, y, "never usable (point estimate)", va="center", fontsize=11, color=colours[region])
                ylab.append(f"{COND_LABEL.get(cond, cond)} · {region} · {'learned' if oid == 'megadesc' else 'matcher'}")
                y += 1
        y += 0.6
        ylab.append("")
    ticks = [i for i, t in enumerate(ylab) if t]
    pos = []
    k = 0
    yy = 0.0
    for t in ylab:
        if t:
            pos.append(yy)
            yy += 1
        else:
            yy += 0.6
    ax.set_yticks(pos)
    ax.set_yticklabels([t for t in ylab if t], fontsize=12)
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlim(1.0, 220)
    ax.set_xticks([1, 2, 5, 10, 20, 50, 100, 200])
    ax.set_xticklabels(["<2", "2", "5", "10", "20", "50", "100", "200"])
    ax.set_xlabel("d* = farthest distance still usable (lower CI of AUC ≥ 0.80), m")
    ax.set_title(f"How far can a zebra still recognise an individual?\nbar = range over bootstrap draws; {BOOT_NOTE}",
                 fontsize=15)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    del ticks, k
    fig.tight_layout()
    out = FIG / "dstar_by_condition.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


HEAT_OBS = [("hotspotter", "local keypoints (HotSpotter)"), ("registered_layout", "registered layout"),
            ("megadesc", "learned (MegaDescriptor)"), ("raw_pixel", "raw pixels")]


def family_heatmap(cond: str, fam: pd.DataFrame, m: pd.DataFrame | None = None):
    """Rows: single clue families (summary-statistics observer), a gap, then whole observers (AUC)."""
    fig, axes = plt.subplots(1, 2, figsize=(17, 9), sharey=True)
    order = list(FAMILY_LABEL)
    cmap = matplotlib.colormaps["viridis"].copy()
    cmap.set_bad("white")
    for ax, region in zip(axes, ("face", "flank")):
        p = fam[fam.region == region].pivot_table(index="family", columns="distance_m", values="auc").reindex(order)
        labels = [FAMILY_LABEL[f] for f in order]
        rows = [p.to_numpy()]
        if m is not None:
            rows.append(np.full((1, p.shape[1]), np.nan))
            labels.append("")
            for oid, lab in HEAT_OBS:
                a = m[(m.region == region) & (m.observer == oid) & (m.metric == "auc")]
                if a.empty:
                    continue
                rows.append(a.set_index("distance_m").value.reindex(p.columns).to_numpy()[None, :])
                labels.append(lab)
        im = ax.imshow(np.vstack(rows), aspect="auto", cmap=cmap, vmin=0.45, vmax=0.90, interpolation="nearest")
        ax.set_xticks(range(p.shape[1]))
        ax.set_xticklabels([f"{d:.0f}" if d >= 10 else f"{d:.1f}" for d in p.columns], rotation=90, fontsize=11)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels)
        ax.tick_params(axis="y", length=0)
        ax.set_xlabel("Viewing distance (m)")
        ax.set_title(region.capitalize())
    cb = fig.colorbar(im, ax=axes, shrink=0.85)
    cb.set_label("AUC, same vs different zebra (0.5 = chance)")
    fig.suptitle(f"Which observer carries identity? {COND_LABEL.get(cond, cond)}, different-day photos\n"
                 "above the gap: one stripe clue at a time (nearest neighbour); below: whole observers", fontsize=15)
    out = FIG / f"family_heatmap_{cond}.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    sr, orig = size_rule(), originals()
    tabs = {}
    for cond in ("c0", "c8", "c9"):
        mp = MET / f"metrics_{cond}.parquet"
        if not mp.exists():
            print(f"skip {cond}: no {mp.name}")
            continue
        m = pd.read_parquet(mp)
        print(auc_chart(cond, m, sr, orig if cond == "c0" else pd.DataFrame()))
        fp = MET / f"family_{cond}.parquet"
        if fp.exists():
            print(family_heatmap(cond, pd.read_parquet(fp), m))
        tabs[cond] = pd.read_csv(MET / f"dstar_{cond}.csv")
    if tabs:
        print(dstar_chart(tabs))


if __name__ == "__main__":
    main()
