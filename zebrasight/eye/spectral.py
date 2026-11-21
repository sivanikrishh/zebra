"""Spectral model behind the dichromacy stage (numpy, precomputed once).

Derivation of the 2x3 linear-RGB -> [L, S] matrix ``M``:

1. Photoreceptor sensitivity T_i(lambda): Govardovskii et al. (2000) A1 template
   (alpha + beta band) at lambda_max = 545 nm (L) and 430 nm (S); rod 500 nm.
   No lens or media filtering (grade C; the horse lens transmits little UV, so S is at most
   slightly overestimated).
2. Display primaries P_j(lambda), j = R, G, B: the "Typical CRT Brainard 1997" spectra from
   colour-science, standing in for sRGB (sRGB is defined on BT.709 CRT phosphors). Each is
   scaled so its CIE 1931 luminance Y equals the sRGB weight (0.2126, 0.7152, 0.0722); then
   R+G+B is the display white.
3. Quantum catch: M_raw[i, j] = sum_lambda P_j(lambda) * lambda * T_i(lambda) (energy -> photons).
4. Von Kries adaptation to the display white: each row is divided by its catch of
   R+G+B, so white (1,1,1) gives L = S = 1 and any grey gives S/L = 1.

Monochromatic lights are handled with the same templates and white normalisation
(``mono_catch``), which is how the neutral point and the colour-confusion tests are done:
monochromatic lights are outside the sRGB gamut.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

WL = np.arange(380.0, 781.0, 1.0)  # nm
SRGB_Y = np.array([0.2126, 0.7152, 0.0722])


def govardovskii_a1(wl: np.ndarray, lmax: float) -> np.ndarray:
    """Govardovskii et al. 2000 A1 absorbance template, alpha + beta band, peak normalised to 1."""
    x = lmax / wl
    a = 0.8795 + 0.0459 * np.exp(-((lmax - 300.0) ** 2) / 11940.0)
    alpha = 1.0 / (np.exp(69.7 * (a - x)) + np.exp(28.0 * (0.922 - x)) + np.exp(-14.9 * (1.104 - x)) + 0.674)
    lmb = 189.0 + 0.315 * lmax
    bb = -40.5 + 0.195 * lmax
    beta = 0.26 * np.exp(-(((wl - lmb) / bb) ** 2))
    return alpha + beta


@lru_cache(maxsize=None)
def display_primaries() -> np.ndarray:
    """(3, len(WL)) R, G, B spectral power, each scaled to its sRGB luminance weight."""
    import colour

    msds = colour.MSDS_DISPLAY_PRIMARIES["Typical CRT Brainard 1997"].copy()
    msds = msds.align(colour.SpectralShape(WL[0], WL[-1], 1.0), extrapolator_kwargs={"method": "Constant", "left": 0, "right": 0})
    p = np.clip(np.asarray(msds.values).T, 0.0, None)  # (3, n)
    ybar = colour.MSDS_CMFS["CIE 1931 2 Degree Standard Observer"].copy().align(
        colour.SpectralShape(WL[0], WL[-1], 1.0)).values[:, 1]
    y = p @ ybar
    return p * (SRGB_Y / y)[:, None]


def _raw_catch(spectra: np.ndarray, lmax: float) -> np.ndarray:
    return spectra @ (WL * govardovskii_a1(WL, lmax))


@lru_cache(maxsize=None)
def white_catch(lmax: float) -> float:
    return float(_raw_catch(display_primaries().sum(axis=0), lmax))


@lru_cache(maxsize=None)
def rgb_to_receptor_matrix(lmaxes: tuple[float, ...]) -> np.ndarray:
    """(len(lmaxes), 3) linear RGB -> receptor catch, white-normalised (white -> 1)."""
    p = display_primaries()
    return np.stack([_raw_catch(p, lm) / white_catch(lm) for lm in lmaxes])


def mono_catch(wavelength_nm: float | np.ndarray, lmax: float) -> np.ndarray:
    """Catch of an equal-photon monochromatic light, in the same white-normalised units."""
    wl = np.asarray(wavelength_nm, dtype=float)
    return govardovskii_a1(wl, lmax) * wl / white_catch(lmax) * (WL[1] - WL[0])


def neutral_point_nm(l_peak: float, s_peak: float) -> float:
    """Wavelength (between the two peaks) whose S/L ratio equals that of the white."""
    wl = np.arange(s_peak, l_peak, 0.1)
    r = np.log(mono_catch(wl, s_peak) / mono_catch(wl, l_peak))
    i = np.nonzero(np.diff(np.sign(r)))[0][0]
    return float(wl[i] - r[i] * (wl[i + 1] - wl[i]) / (r[i + 1] - r[i]))


def chroma_coord(l: np.ndarray, s: np.ndarray) -> np.ndarray:
    """Dichromat chromaticity s / (l + s): 0.5 for white/grey, 0 = pure L, 1 = pure S."""
    return s / (l + s)


def dichromat_display_matrix(m_ls: np.ndarray) -> np.ndarray:
    """(3, 2) [L, S] -> linear sRGB, Vienot, Brettel & Mollon (1999) adapted to two cones.

    Vienot et al. keep the display's blue primary and yellow (R+G) unchanged, so the
    simulated colours lie in the plane through black, blue and yellow (white is
    preserved by linearity). With only two cones the dichromat colour space is 2-D,
    so D is fixed exactly by D @ M @ [b, y] = [b, y].
    """
    anchors = np.array([[0.0, 1.0], [0.0, 1.0], [1.0, 0.0]])  # columns: blue (0,0,1), yellow (1,1,0)
    return anchors @ np.linalg.inv(m_ls @ anchors)
