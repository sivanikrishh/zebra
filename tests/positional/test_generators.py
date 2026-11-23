import numpy as np

from tests.positional import synth
from zebrasight.positional import generators as G

MASK = synth.ellipse()
ALL = np.ones((synth.H, synth.W), bool)


def test_phase_scramble_keeps_power_destroys_phase():
    y = synth.stripes(seed=1)
    _, field = G.phase_scramble(y, ALL, seed=0, return_field=True)
    x = y - y.mean()
    p0, p1 = np.abs(np.fft.fft2(x)) ** 2, np.abs(np.fft.fft2(field)) ** 2
    assert np.abs(p1 - p0).sum() / p0.sum() < 0.01
    assert abs(np.corrcoef(field.ravel(), x.ravel())[0, 1]) < 0.1


def test_patch_shuffle_keeps_each_tile():
    y = synth.stripes(seed=2)
    out, tiles, perm = G.patch_shuffle(y, MASK, synth.BL, 0.05, seed=0)
    s = max(2, int(round(0.05 * synth.BL)))
    assert len(tiles) > 10 and not np.allclose(out, y)
    for (r, c), k in zip(tiles, perm):
        rr, cc = tiles[k]
        assert np.array_equal(out[r:r + s, c:c + s], y[rr:rr + s, cc:cc + s])
    assert np.array_equal(out[~MASK], y[~MASK])


def test_silhouette_has_no_stripe_energy():
    y = synth.stripes()
    u = G.silhouette(y, MASK, "uniform")
    assert u[MASK].std() < 1e-6
    sc = G.silhouette(y, MASK, "scrambled", seed=1)
    assert sc[MASK].std() > 0.1   # stripes' energy kept but scrambled


def test_band_filter_passes_only_its_band():
    y = synth.stripes(seed=3)
    out = G.band_filter(y, ALL, synth.BL, (12, 24), keep=True)
    F = np.fft.fft2(out - out.mean())
    r = np.hypot(np.fft.fftfreq(synth.W)[None, :], np.fft.fftfreq(synth.H)[:, None]) * synth.BL
    outside = ~((r >= 12) & (r < 24))
    assert (np.abs(F[outside]) ** 2).sum() / (np.abs(F) ** 2).sum() < 1e-6
    rem = G.band_filter(y, ALL, synth.BL, (12, 24), keep=False)
    Fr = np.fft.fft2(rem - rem.mean())
    assert (np.abs(Fr[~outside]) ** 2).sum() / max((np.abs(Fr) ** 2).sum(), 1e-12) < 1e-6


def test_morph_positional_amplitude():
    y = synth.stripes()
    for amp in (0.01, 0.1, 0.2):
        out, d = G.morph_positional(y, MASK, synth.BL, amp, 2.0, seed=0, return_field=True)
        rms = np.sqrt((d[0][MASK] ** 2 + d[1][MASK] ** 2).mean())
        assert abs(rms - amp * synth.PERIOD) / (amp * synth.PERIOD) < 1e-6
        assert not np.allclose(out[MASK], y[MASK])


def _centre_shift(a, b, period):
    """Mean shift of the stripe fundamental between two vertical-stripe images, in periods."""
    k = synth.W / period
    rows = slice(synth.H // 3, 2 * synth.H // 3)
    fa = np.fft.rfft(a[rows] - a[rows].mean(), axis=1)[:, int(round(k))]
    fb = np.fft.rfft(b[rows] - b[rows].mean(), axis=1)[:, int(round(k))]
    return np.abs(np.angle((fa * np.conj(fb)).sum())) / (2 * np.pi)


def test_morph_style_keeps_stripe_centres():
    period = synth.W / 26.0   # integer number of periods across the image: clean fundamental
    y = synth.stripes(period=period, wobble=0.0)
    bl = 24 * period          # so that the stripe period is the configured 24 cpbl
    for wc in (-0.4, 0.4):
        out = G.morph_style(y, ALL, bl, width_change=wc, contrast=0.7)
        assert _centre_shift(y, out, period) < 0.1
        assert abs((out < 0.5).mean() - (y < 0.5).mean()) > 0.02   # widths really changed


def test_transplant_control_returns_original():
    y = synth.stripes(seed=4)
    v = synth.view(y, MASK)
    out, region = G.transplant(v, v, "mid_flank")
    assert region.sum() > 100 and np.abs(out - y).max() < 1e-4
    donor = synth.view(synth.stripes(seed=9), MASK)
    out2, region2 = G.transplant(v, donor, "mid_flank")
    assert np.abs(out2 - y)[region2].mean() > 0.05 and np.allclose(out2[~MASK], y[~MASK])


def test_occlusions_fill_with_local_mean():
    y = synth.stripes()
    n = 0
    for out, (r, c, s) in G.occlusions(y, MASK, synth.BL, 0.1, 0.05):
        sub = MASK[r:r + s, c:c + s]
        assert np.allclose(out[r:r + s, c:c + s][sub], y[r:r + s, c:c + s][sub].mean(), atol=1e-6)
        n += 1
    assert n > 10


def test_determinism():
    y = synth.stripes(seed=1)
    assert np.array_equal(G.phase_scramble(y, MASK, 3), G.phase_scramble(y, MASK, 3))
    assert np.array_equal(G.patch_shuffle(y, MASK, synth.BL, 0.1, 3)[0], G.patch_shuffle(y, MASK, synth.BL, 0.1, 3)[0])
    assert np.array_equal(G.morph_positional(y, MASK, synth.BL, 0.1, 2, 3), G.morph_positional(y, MASK, synth.BL, 0.1, 2, 3))
    assert not np.array_equal(G.phase_scramble(y, MASK, 3), G.phase_scramble(y, MASK, 4))


def _period_x(img, rows, cols=slice(None)):
    """Dominant stripe period (px) along x from the mean FFT of the given rows (and columns)."""
    x = img[rows, cols] - img[rows, cols].mean(axis=1, keepdims=True)
    p = (np.abs(np.fft.rfft(x, axis=1)) ** 2).mean(axis=0)
    f = np.fft.rfftfreq(img.shape[1])
    k = int(np.argmax(p[1:]) + 1)
    return 1.0 / f[k]


def test_coat_rescale_identity_outside_and_spacing():
    y = synth.stripes(seed=3, wobble=0.0)
    assert np.allclose(G.coat_rescale(y, MASK, 1.0)[MASK], y[MASK], atol=1e-4)
    out = G.coat_rescale(y, MASK, 1.5)
    assert np.array_equal(out[~MASK], y[~MASK].astype(np.float32))
    big = np.zeros_like(MASK)
    big[10:-10, 10:-10] = True
    # centre region only: there the resampled coat comes from inside the mask (edges may need extension)
    rows = slice(synth.H // 2 - 10, synth.H // 2 + 10)
    cols = slice(synth.W // 2 - synth.W // 4, synth.W // 2 + synth.W // 4)
    p0 = _period_x(y, rows, cols)
    for f in (0.75, 1.5):
        p = _period_x(G.coat_rescale(y, big, f), rows, cols)
        assert abs(p / p0 - f) < 0.1 * f, (f, p0, p)
