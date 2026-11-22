"""Recognition metrics from the feature Parquet alone (no store needed). Definitions: configs/metrics.yaml.

Per (region, cond, distance): gallery = split "gallery", query = split "query" (different dates, never
the same encounter). An observer turns gallery and query rows into a similarity matrix S
(query x gallery); every fit (z-score, PCA, LDA) uses gallery rows only.
NaN rule: a percept whose F features are all NaN (invalid) gets the minimum similarity (a miss).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import roc_auc_score

LAYOUT_PCA = 20


def feature_columns(df: pd.DataFrame) -> dict[str, list[str]]:
    cols = df.columns
    F = [c for c in cols if c[:1] == "F" and c.split("_")[0][1:].isdigit() and "rawotsu" not in c]
    fams = {f"F{i}": [c for c in F if c.split("_")[0] == f"F{i}"] for i in range(1, 11)}
    return {"F": F, "L": [c for c in cols if c.startswith("L_") and "rawotsu" not in c],
            "M": [c for c in cols if c.startswith("M_")], "R": [c for c in cols if c.startswith("R_")],
            "L_rawotsu": [c for c in cols if c.startswith("L_rawotsu_")], **fams}


@dataclass
class Split:
    g: pd.DataFrame  # gallery rows
    q: pd.DataFrame  # query rows
    gz: np.ndarray  # gallery zebra ids
    qz: np.ndarray


def split(df: pd.DataFrame) -> Split:
    g = df[df.split == "gallery"].reset_index(drop=True)
    q = df[df.split == "query"].reset_index(drop=True)
    return Split(g, q, g.zebra_id.to_numpy(), q.zebra_id.to_numpy())


def _zscore(gx: np.ndarray, qx: np.ndarray):
    mu = np.nanmean(gx, axis=0)
    sd = np.nanstd(gx, axis=0)
    sd = np.where(np.isfinite(sd) & (sd > 1e-12), sd, 1.0)
    mu = np.where(np.isfinite(mu), mu, 0.0)
    gz = np.nan_to_num((gx - mu) / sd, nan=0.0)  # per-feature NaN -> gallery mean (0 after z-scoring)
    qz = np.nan_to_num((qx - mu) / sd, nan=0.0)
    return gz, qz


def _invalid(rows: pd.DataFrame, F: list[str]) -> np.ndarray:
    return rows[F].isna().all(axis=1).to_numpy()


def _neg_euclid(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    d2 = (a**2).sum(1)[:, None] + (b**2).sum(1)[None, :] - 2 * a @ b.T
    return -np.sqrt(np.maximum(d2, 0.0))


def _cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a - a.mean(1, keepdims=True)
    b = b - b.mean(1, keepdims=True)
    an = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)
    bn = b / np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-12)
    return an @ bn.T


def _apply_misses(S: np.ndarray, q_bad: np.ndarray, g_bad: np.ndarray) -> np.ndarray:
    S = S.copy()
    floor = np.nanmin(S) - 1.0 if np.isfinite(S).any() else -1.0
    S[q_bad, :] = floor
    S[:, g_bad] = floor
    return S


def similarity(sp: Split, observer: str, cols: dict, families: list[str] | None = None) -> np.ndarray:
    """S (n_query x n_gallery) for one observer."""
    F = cols["F"]
    if observer == "raw_pixel":
        ga = np.nan_to_num(sp.g[cols["R"]].to_numpy(float), nan=1.0)
        qa = np.nan_to_num(sp.q[cols["R"]].to_numpy(float), nan=1.0)
        return _cosine(qa, ga)  # works at any size: no min-size rule
    if families is None:
        families = [f"F{i}" for i in range(1, 11)] + (["L", "M"] if observer.startswith("spatial") else [])
    blocks_g, blocks_q = [], []
    for fam in families:
        if fam == "L":
            lg, lq = sp.g[cols["L"]].to_numpy(float), sp.q[cols["L"]].to_numpy(float)
            ok = ~np.isnan(lg).any(1)
            if ok.sum() >= 3:
                fill = np.nanmean(lg[ok], 0)
                lg, lq = np.where(np.isnan(lg), fill, lg), np.where(np.isnan(lq), fill, lq)
                pca = PCA(n_components=min(LAYOUT_PCA, ok.sum() - 1)).fit(lg[ok])
                blocks_g.append(pca.transform(lg))
                blocks_q.append(pca.transform(lq))
            continue
        blocks_g.append(sp.g[cols[fam]].to_numpy(float))
        blocks_q.append(sp.q[cols[fam]].to_numpy(float))
    gx, qx = _zscore(np.hstack(blocks_g), np.hstack(blocks_q))
    if observer.endswith("_lda"):
        y = sp.gz
        if len(np.unique(y)) >= 2 and gx.shape[0] > len(np.unique(y)):
            lda = LinearDiscriminantAnalysis(solver="eigen", shrinkage="auto").fit(gx, y)
            gx, qx = lda.transform(gx), lda.transform(qx)
    S = _neg_euclid(qx, gx)
    return _apply_misses(S, _invalid(sp.q, F), _invalid(sp.g, F))


# ---- pair metrics ---------------------------------------------------------------------------------

def pairs(sp: Split, S: np.ndarray, rng: np.random.Generator, max_ratio: int = 10):
    """(scores, labels, query_zebra) for genuine pairs and <= max_ratio x as many impostors."""
    same = sp.qz[:, None] == sp.gz[None, :]
    qi, gi = np.nonzero(same)
    ii, ij = np.nonzero(~same)
    k = min(len(ii), max_ratio * len(qi))
    pick = rng.choice(len(ii), size=k, replace=False) if k < len(ii) else np.arange(len(ii))
    scores = np.r_[S[qi, gi], S[ii[pick], ij[pick]]]
    labels = np.r_[np.ones(len(qi)), np.zeros(k)]
    qzeb = np.r_[sp.qz[qi], sp.qz[ii[pick]]]
    return scores, labels, qzeb


def auc(scores, labels) -> float:
    if labels.min() == labels.max():
        return math.nan
    return float(roc_auc_score(labels, scores))


def boot_auc(scores, labels, qzeb, zebras: np.ndarray, n_boot: int, rng: np.random.Generator) -> np.ndarray:
    """AUC over bootstrap resamples of query zebras (cluster bootstrap)."""
    idx_by_z = {z: np.flatnonzero(qzeb == z) for z in zebras}
    out = np.empty(n_boot)
    for b in range(n_boot):
        draw = rng.choice(zebras, size=len(zebras), replace=True)
        idx = np.concatenate([idx_by_z[z] for z in draw])
        out[b] = auc(scores[idx], labels[idx])
    return out


def eer_dprime_tar(scores, labels, far=0.01) -> dict:
    from sklearn.metrics import roc_curve

    if labels.min() == labels.max():
        return {"eer": math.nan, "dprime": math.nan, "tar_at_far": math.nan}
    fpr, tpr, _ = roc_curve(labels, scores)
    fnr = 1 - tpr
    i = np.nanargmin(np.abs(fnr - fpr))
    g, im = scores[labels == 1], scores[labels == 0]
    dprime = (g.mean() - im.mean()) / math.sqrt(0.5 * (g.var() + im.var()) + 1e-12)
    return {"eer": float((fnr[i] + fpr[i]) / 2), "dprime": float(dprime), "tar_at_far": float(np.interp(far, fpr, tpr))}


# ---- herd (closed-set) metrics ------------------------------------------------------------------------

def fano_bits(acc: float, n: int) -> float:
    """Fano lower bound on identity information from top-1 accuracy among n identities."""
    pe = min(max(1.0 - acc, 0.0), 1.0 - 1.0 / n)
    h = 0.0 if pe in (0.0, 1.0) else -(pe * math.log2(pe) + (1 - pe) * math.log2(1 - pe))
    return max(0.0, math.log2(n) - h - pe * math.log2(max(n - 1, 1)))


def herd_top1(sp: Split, S: np.ndarray, n: int, draws: int, rng: np.random.Generator) -> np.ndarray:
    """Top-1 accuracy over random herds of n zebras (gallery holds only those n)."""
    zebras = np.intersect1d(np.unique(sp.gz), np.unique(sp.qz))
    n = min(n, len(zebras))
    acc = np.empty(draws)
    for d in range(draws):
        herd = rng.choice(zebras, size=n, replace=False)
        gm = np.isin(sp.gz, herd)
        qm = np.isin(sp.qz, herd)
        Sub = S[np.ix_(qm, gm)]
        best = sp.gz[gm][np.argmax(Sub, axis=1)]
        acc[d] = float((best == sp.qz[qm]).mean())
    return acc


def ci(a, lo=2.5, hi=97.5):
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    return (float(np.percentile(a, lo)), float(np.percentile(a, hi))) if a.size else (math.nan, math.nan)
