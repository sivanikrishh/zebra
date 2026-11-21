"""Data-layer unit tests (no images needed)."""

import numpy as np
import pandas as pd

from zebrasight.data.gzgc import _session, assign_splits, left_subset
from zebrasight.data.masks import LABELS, OTHER, head_on_left, rule_regions


def test_session_parsing():
    assert _session("NNP GZC Car '10WHITE', Person 'A', Image 0005", "2015-03-01") == "2015-03-01_10WHITE_A"
    assert _session("", "2015-03-01") == "2015-03-01_unknown"


def test_rule_regions_left_view():
    mask = np.zeros((100, 200), dtype=bool)
    mask[10:90, 20:180] = True  # body 160 px long, 80 px tall
    lab = rule_regions(mask, head_left=True)
    cols = lambda code: np.nonzero((lab == code).any(axis=0))[0]
    rows = lambda code: np.nonzero((lab == code).any(axis=1))[0]
    assert cols(LABELS["face"]).max() < 20 + 0.2 * 160  # face at the image left (head end)
    assert cols(LABELS["rump"]).min() >= 20 + 0.75 * 160 - 1
    assert 20 + 0.3 * 160 - 1 <= cols(LABELS["flank"]).min() and cols(LABELS["flank"]).max() < 20 + 0.7 * 160
    for code in LABELS.values():
        assert rows(code).max() < 10 + 0.65 * 80  # legs (lower 35%) dropped
    assert (lab[70:90, 20:180] == OTHER).all()  # the leg band is "other body"
    assert ((lab > 0) == mask).all()  # labels never leave the body mask


def test_rule_regions_right_view_mirrors():
    mask = np.zeros((50, 100), dtype=bool)
    mask[:, 10:90] = True
    left = rule_regions(mask, head_left=True)
    right = rule_regions(mask[:, ::-1], head_left=False)
    np.testing.assert_array_equal(left, right[:, ::-1])
    assert head_on_left("left") and not head_on_left("right")


def _toy():
    rows = []
    for z, dates in (("A", ["d1", "d1", "d2"]), ("B", ["d2", "d3"]), ("C", ["d1", "d1"])):
        for i, d in enumerate(dates):
            rows.append({"ann_id": len(rows), "zebra_id": z, "date": d, "viewpoint": "left",
                         "session_id": f"{d}_car1_A", "encounter_id": f"{z}@{d}_car1_A"})
    return pd.DataFrame(rows)


def test_subset_and_splits():
    sub = left_subset(_toy())
    assert set(sub.zebra_id) == {"A", "B"}  # C only on one date
    split = assign_splits(sub)
    sub = sub.assign(split=split)
    assert (sub.groupby("encounter_id").split.nunique() == 1).all()
    assert (sub.groupby("zebra_id").split.nunique() == 2).all()
    assert set(sub[sub.zebra_id == "A"].query("split == 'gallery'").date) == {"d1"}
