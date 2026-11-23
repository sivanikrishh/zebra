"""Pattern Lab: reusable scientific processing behind the ZebraSight "Pattern Lab" experiments.

No UI code. The 2D research prototype (scripts/build_app_demo.py) and the separate 3D application can call the
same functions. Everything reuses the study's pipeline and observers; nothing is trained.

Stripe modifications (on the full-resolution original flank crop, inside the region mask only, each RGB
channel with the same operation; the tested generators in zebrasight.positional.generators):
    thickness    morph_style(width_change)   dark stripes thicker (+) / thinner (-), stripe centres kept
    contrast     morph_style(contrast)       black-white contrast scaled around the region mean
    arrangement  morph_positional(amp)       smooth random displacement, RMS = amp x stripe period (seeded)
    spacing      coat_rescale(factor)        coat resampled about the mask centroid: spacing x factor
                                             (density = 1 / spacing, so it is one control)

Observers, with exactly the study's preprocessing:
    MegaDescriptor   metrics.megadesc.to_input on the full-resolution crop -> L2-normalised embedding; cosine
    HotSpotter       crop capped at metrics.hotspotter.MAX_SIDE, grey SIFT in the mask, ratio test, RANSAC;
                     score = inliers
Identification = closed set of the known zebras (their earliest-day photos): a zebra's score is its best
gallery photo; rank of the true zebra = 1 + number of zebras scoring strictly higher; ties are reported.
check_against_study() verifies that the unmodified photo reproduces the study's stored scores.

Distance experiment: like-with-like, as in the study (both photos through the zebra eye at the same distance
and light); the scores are the study's stored inference outputs for that exact pair (official tables).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import cv2
import numpy as np
import pandas as pd

from zebrasight.config import REPO_ROOT, load_config

FEAT = REPO_ROOT / "outputs" / "features"
REGIONS = REPO_ROOT / "data" / "regions_50.parquet"


def cfg():
    return load_config("positional").pattern_lab


# ---- photos -----------------------------------------------------------------------------------------------

AVAILABLE_REGIONS = ("flank", "face")       # validated study results exist; "body" results are pending


def region_table(region: str = "flank") -> pd.DataFrame:
    if region not in AVAILABLE_REGIONS:
        raise ValueError(f"no validated results for region {region!r} yet (available: {AVAILABLE_REGIONS})")
    r = pd.read_parquet(REGIONS)
    r = r[r.region == region].copy()
    r["key"] = r.ann_id.astype(int).astype(str) + "_" + region
    return r.sort_values("ann_id").reset_index(drop=True)


def load_crop(row) -> tuple[np.ndarray, np.ndarray]:
    """Full-resolution original crop: (h, w, 3) the photo's sRGB values in [0, 1], (h, w) bool mask."""
    from zebrasight.data.load import load_region

    img, mask, _ = load_region(row)
    return img[0].permute(1, 2, 0).numpy().astype(np.float32), mask[0, 0].numpy() > 0.5


@lru_cache(maxsize=None)
def body_length_px(ann_id: int) -> float:
    """Body length in photo px (= full-resolution crop px) from the positional canonical frame."""
    from zebrasight.positional import data_io as dio

    return float(dio.frames(dio.regions())[int(ann_id)].length_px)


# ---- stripe modifications ---------------------------------------------------------------------------------

CONTROLS = ("thickness", "contrast", "arrangement", "spacing")


def modify(rgb: np.ndarray, mask: np.ndarray, bl_px: float, control: str, value: float) -> np.ndarray:
    """One stripe modification inside the mask; outside the mask the photo is untouched. -> (h, w, 3) [0, 1]."""
    from zebrasight.positional import generators as G

    c = cfg()
    if control == "thickness":
        f = lambda y: G.morph_style(y, mask, bl_px, width_change=value)
    elif control == "contrast":
        f = lambda y: G.morph_style(y, mask, bl_px, contrast=value)
    elif control == "arrangement":
        f = lambda y: G.morph_positional(y, mask, bl_px, value, float(c.smooth_period), int(c.seed))
    elif control == "spacing":
        f = lambda y: G.coat_rescale(y, mask, value)
    else:
        raise ValueError(f"unknown control {control!r}; expected one of {CONTROLS}")
    out = np.stack([f(rgb[..., k]) for k in range(3)], axis=-1)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


# ---- observers --------------------------------------------------------------------------------------------

def hs_input(rgb: np.ndarray, mask: np.ndarray):
    """HotSpotter features exactly as metrics.hotspotter.originals (crop capped at MAX_SIDE)."""
    from zebrasight.metrics.hotspotter import MAX_SIDE, describe, grey_u8

    s = MAX_SIDE / max(mask.shape)
    if s < 1:
        size = (round(mask.shape[1] * s), round(mask.shape[0] * s))
        rgb = cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST) > 0
    return describe(grey_u8(rgb), mask)


@dataclass
class Observers:
    """MegaDescriptor (loaded once) + HotSpotter, with the known-zebra gallery (earliest-day originals)."""
    device: str = "cuda"
    region: str = "flank"

    def __post_init__(self):
        import torch

        from zebrasight.metrics.megadesc import load_model

        if self.device == "cuda" and not torch.cuda.is_available():
            self.device = "cpu"
        self.model = load_model(self.device)
        t = region_table(self.region)
        self.gallery = t[t.split == "gallery"].reset_index(drop=True)
        E = pd.read_parquet(FEAT / "megadesc_orig.parquet").set_index("key")
        ec = [c for c in E.columns if c.startswith("E_")]
        self.g_emb = E.loc[self.gallery.key, ec].to_numpy(np.float32)          # official gallery embeddings
        self.g_hs = [hs_input(*load_crop(r)) for r in self.gallery.to_dict("records")]

    def embed(self, rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
        from zebrasight.metrics.megadesc import embed, to_input

        return embed(self.model, [to_input(rgb, mask)], self.device)[0]

    def identify(self, rgb: np.ndarray, mask: np.ndarray, true_zebra: str) -> dict:
        """Both observers against every known zebra; returns per-observer ranking of the true zebra."""
        from zebrasight.metrics.hotspotter import match

        e = self.embed(rgb, mask)
        f = hs_input(rgb, mask)
        md = self.g_emb @ e
        hs = np.array([match(f, g)[0] for g in self.g_hs], float)
        out = {"_emb": e, "_hs": f}
        for name, s in (("megadesc", md), ("hotspotter", hs)):
            per = pd.Series(s).groupby(self.gallery.zebra_id.to_numpy()).max().sort_values(ascending=False)
            t = float(per[true_zebra])
            out[name] = {"true_score": round(t, 4), "rank": int((per > t).sum()) + 1,
                         "tied_with": int((per == t).sum()) - 1, "n_known": int(len(per)),
                         "top": [[z, round(float(v), 4)] for z, v in per.head(5).items()]}
        return out

    def self_similarity(self, ident: dict, ref: dict) -> dict:
        """Modified vs unmodified version of the same photo (MegaDescriptor cosine, HotSpotter inliers)."""
        from zebrasight.metrics.hotspotter import match

        return {"megadesc": round(float(ident["_emb"] @ ref["_emb"]), 4),
                "hotspotter": int(match(ident["_hs"], ref["_hs"])[0])}

    def check_against_study(self, key: str, ident: dict) -> dict:
        """Unmodified photo: our embedding vs the stored one, our inliers vs the official table."""
        from zebrasight.metrics.hotspotter import match

        E = pd.read_parquet(FEAT / "megadesc_orig.parquet").set_index("key")
        ec = [c for c in E.columns if c.startswith("E_")]
        cos = float(E.loc[key, ec].to_numpy(np.float32) @ ident["_emb"])
        H = pd.read_parquet(FEAT / "hotspotter_orig.parquet")
        H = H[(H.region == self.region) & (H.qkey == key)].set_index("gkey").inliers
        ours = np.array([match(ident["_hs"], g)[0] for g in self.g_hs])
        ref = H.reindex(self.gallery.key).fillna(-1).to_numpy()
        return {"megadesc_cosine_to_stored": round(cos, 5), "hotspotter_mismatches": int((ours != ref).sum())}


# ---- distance experiment (stored study inference, like-with-like) -----------------------------------------

def distance_measurements(query_key: str, ref_key: str, true_zebra: str, region: str = "flank",
                          conds=("c0", "c8", "c9")) -> dict:
    """Per light and distance: the study's stored scores for (later-day photo, its earliest-day reference),
    both seen through the zebra eye at that distance, and the true zebra's rank among the known zebras."""
    t = region_table(region)
    gal = t[t.split == "gallery"]
    gz = gal.zebra_id.to_numpy()
    E = pd.concat([pd.read_parquet(FEAT / f) for f in ("megadesc_c0.parquet", "megadesc_c8c9.parquet")])
    E = E[E.key.str.endswith("_" + region)]
    ec = [c for c in E.columns if c.startswith("E_")]
    H = pd.read_parquet(FEAT / "hotspotter_percepts.parquet")
    H = H[(H.region == region) & (H.qkey == query_key)]
    out = {}
    for c in conds:
        rows = []
        for d in range(16):
            e = E[(E.cond == c) & (E.d_index == d)].set_index("key")
            q = e.loc[query_key, ec].to_numpy(float)
            md = None if np.isnan(q).any() else np.nan_to_num(e.loc[gal.key, ec].to_numpy(float)) @ q
            h = H[(H.cond == c) & (H.d_index == d)].set_index("gkey").inliers.reindex(gal.key).fillna(0).to_numpy(float)
            rec = {"d_index": d}
            for name, s in (("megadesc", md), ("hotspotter", h)):
                if s is None:
                    rec[name] = None
                    continue
                per = pd.Series(s).groupby(gz).max()
                tv = float(per[true_zebra])
                rec[name] = {"ref_score": round(float(s[gal.key.tolist().index(ref_key)]), 4),
                             "rank": int((per > tv).sum()) + 1, "tied_with": int((per == tv).sum()) - 1,
                             "n_known": int(len(per))}
            rows.append(rec)
        out[c] = rows
    return out
