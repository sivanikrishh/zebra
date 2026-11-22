"""Recognition metrics on synthetic feature tables."""

import math

import numpy as np
import pandas as pd
import pytest

from zebrasight.metrics.core import (auc, boot_auc, fano_bits, feature_columns, herd_top1, pairs, similarity, split)


def table(n_zebras=30, per=2, noise=0.05, seed=0, nan_query=0):
    rng = np.random.default_rng(seed)
    rows = []
    centres = rng.normal(size=(n_zebras, 6))
    for z in range(n_zebras):
        for s in ("gallery", "query"):
            for k in range(per):
                f = centres[z] + noise * rng.normal(size=6)
                r = {"zebra_id": f"z{z}", "split": s, "key": f"{z}{s}{k}"}
                r.update({f"F{i + 1}_a": f[i] for i in range(6)})
                r.update({f"R_{i:03d}": f[i % 6] + 0.01 * i for i in range(12)})
                rows.append(r)
    df = pd.DataFrame(rows)
    if nan_query:
        q = df.index[df.split == "query"][:nan_query]
        df.loc[q, [c for c in df.columns if c.startswith("F")]] = np.nan
    return df


@pytest.mark.parametrize("observer", ["interpretable_nn", "interpretable_lda", "raw_pixel"])
def test_separable_gives_auc_one(observer):
    df = table(noise=0.01)
    sp = split(df)
    S = similarity(sp, observer, feature_columns(df))
    s, l, _ = pairs(sp, S, np.random.default_rng(0))
    assert auc(s, l) == pytest.approx(1.0, abs=1e-9)
    assert herd_top1(sp, S, 5, 50, np.random.default_rng(0)).mean() == pytest.approx(1.0)


def test_shuffled_labels_give_chance():
    df = table(noise=0.01)
    sp = split(df)
    S = similarity(sp, "interpretable_nn", feature_columns(df))
    rng = np.random.default_rng(1)
    sp.qz = rng.permutation(sp.qz)  # label-shuffle null
    s, l, qz = pairs(sp, S, np.random.default_rng(0))
    assert auc(s, l) == pytest.approx(0.5, abs=0.07)
    top1 = herd_top1(sp, S, 5, 200, np.random.default_rng(0)).mean()
    assert top1 == pytest.approx(1 / 5, abs=0.07)


def test_nan_percepts_count_as_misses():
    df = table(noise=0.01, nan_query=10)  # 10 invalid query percepts
    sp = split(df)
    S = similarity(sp, "interpretable_nn", feature_columns(df))
    bad = sp.q[[c for c in df.columns if c.startswith("F")]].isna().all(axis=1).to_numpy()
    assert bad.sum() == 10 and (S[bad] == S.min()).all()  # minimum similarity, not dropped
    s, l, _ = pairs(sp, S, np.random.default_rng(0))
    assert auc(s, l) < 1.0  # the misses cost AUC
    assert len(s) >= (sp.qz[:, None] == sp.gz[None, :]).sum()  # every genuine pair still counted


def test_bootstrap_and_fano():
    df = table(noise=0.5)
    sp = split(df)
    S = similarity(sp, "interpretable_nn", feature_columns(df))
    s, l, qz = pairs(sp, S, np.random.default_rng(0))
    b = boot_auc(s, l, qz, np.unique(sp.qz), 100, np.random.default_rng(0))
    assert np.nanpercentile(b, 2.5) <= auc(s, l) <= np.nanpercentile(b, 97.5)
    assert fano_bits(1.0, 20) == pytest.approx(math.log2(20))
    assert fano_bits(1 / 20, 20) == pytest.approx(0.0, abs=1e-9)
