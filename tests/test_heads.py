"""Region rule v2 (no model download needed: synthetic masks and boxes)."""

import numpy as np

from zebrasight.data.heads import FACE, FLANK, RUMP, _box_mask, pick_head, regions_v2


def zebra_mask():
    m = np.zeros((100, 220), bool)
    m[20:60, 60:200] = True  # trunk
    m[25:45, 30:60] = True  # neck (thinner)
    m[40:80, 10:35] = True  # lowered head (grazing) at the front (left)
    m[60:95, 70:78] = m[60:95, 180:188] = True  # legs
    m[30:34, 200:215] = True  # tail
    return m


def test_box_mask_clamps_negative_edges():
    m = _box_mask((50, 80), [-97, 10, -65, 40])  # entirely left of the image
    assert not m.any()
    assert _box_mask((50, 80), [70, 40, 120, 90]).sum() == 10 * 10


def test_face_from_head_box_and_trunk_regions():
    m = zebra_mask()
    lab, method = regions_v2(m, True, [8, 38, 37, 82])
    assert method == "owlv2"
    assert (lab == FACE).any() and np.nonzero((lab == FACE).any(0))[0].max() < 40  # face is the head
    fc, rc = np.nonzero((lab == FLANK).any(0))[0], np.nonzero((lab == RUMP).any(0))[0]
    assert fc.min() > 60 and fc.max() < rc.min()  # flank behind the shoulder, before the rump
    assert rc.max() < 200  # rump excludes the tail
    assert not (lab[:, 25:60] == FLANK).any()  # no neck in the flank


def test_silhouette_tip_fallback_catches_lowered_head():
    m = zebra_mask()
    lab, method = regions_v2(m, True, None)
    assert method == "silhouette_tip"
    assert (lab[60:80, 10:35] == FACE).any()  # the lowered head (below the back line) is face


def test_pick_head_rejects_whole_body_and_rear_boxes():
    m = zebra_mask()
    dets = [(0.7, [5, 15, 215, 95]),  # whole zebra
            (0.5, [150, 20, 200, 60]),  # rear
            (0.3, [-97, 10, -65, 40]),  # off the body
            (0.2, [8, 38, 37, 82])]  # the head
    got = pick_head(dets, m, True, (0, 0), score_min=0.15)
    assert got is not None and got[0] == 0.2
