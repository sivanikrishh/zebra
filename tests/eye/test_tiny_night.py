"""Regression: a tiny night percept with no frequency band used to crash peli_csf_batch (IndexError)."""

import torch

from zebrasight.eye.csf import peli_csf, peli_csf_batch


def test_item_without_band_passes_through():
    # the real crash (crop 2439_face, c9): a 1x1 px percept batched with a 1x2 px one. The canvas then
    # tops out at 0.5 cycles/px, below the 1x1 item's lowest frequency, so it has no band at all.
    torch.manual_seed(0)
    tiny = torch.rand(1, 1, 1, 1) + 0.5
    big = torch.rand(1, 1, 1, 2) + 0.5
    args = dict(acuity=1.61, b_frac=0.1, c=50.0, s=0.5, delta=0.05, taper=0.02, chroma_ratio=0.5)
    lums, _, info = peli_csf_batch([tiny, big], None, [0.25, 0.25], acuity_maps=[None, None], **args)
    assert info.get("passthrough") == 1
    assert torch.equal(lums[0], tiny)  # DC only: unchanged
    alone, _, _ = peli_csf(big, None, 0.25, args["acuity"], None, args["b_frac"], args["c"], args["s"],
                           args["delta"], args["taper"], args["chroma_ratio"])
    torch.testing.assert_close(lums[1], alone)  # the rest of the batch is computed as before


def test_night_pipeline_on_tiny_crops_runs():
    from zebrasight.config import make_cond
    from zebrasight.pipeline import Pipeline

    img = torch.rand(1, 3, 40, 60)
    pipe = Pipeline(["geometry", "eye"])
    out = pipe.run_distances(img, 0.5, torch.ones(1, 1, 40, 60), make_cond("c9", 2.0),
                             [2.0, 23.318, 79.621, 147.128, 200.0])
    assert len(out) == 5 and all(torch.isfinite(p.y).all() for p in out)
