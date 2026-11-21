"""Shader parameter JSONs, generated from configs/eye.yaml, configs/env.yaml, configs/grid.yaml."""

from __future__ import annotations

import math

import numpy as np
from omegaconf import OmegaConf

from zebrasight.config import load_config
from zebrasight.env.atmosphere import airlight_rgb, channel_betas, visual_range_m
from zebrasight.eye import spectral
from zebrasight.eye.streak import LIGHT_TO_REGIME

SRGB_Y = (0.2126, 0.7152, 0.0722)
REGIMES = ("day", "dusk", "night")
REGIME_LIGHT = {"day": "day", "dusk": "dusk", "night": "night"}


def _v(node):
    return OmegaConf.to_object(node.value) if OmegaConf.is_config(node.value) else node.value


def matrices():
    eye = load_config("eye")
    l, s, rod = (float(_v(eye.cones.L_peak)), float(_v(eye.cones.S_peak)), float(_v(eye.cones.rod_peak)))
    m = spectral.rgb_to_receptor_matrix((l, s))  # (2,3) linear RGB -> [L, S], white -> (1, 1)
    d = spectral.dichromat_display_matrix(m)  # (3,2) [L, S] -> linear sRGB for human viewing
    r = spectral.rgb_to_receptor_matrix((rod,))  # (1,3) linear RGB -> rod, white -> 1
    return l, s, rod, m, d, r


def dichromat() -> dict:
    l, s, rod, m, d, r = matrices()
    return {
        "description": "Zebra dichromat matrices. All RGB here is LINEAR sRGB: decode textures with the sRGB EOTF "
                       "before using them and encode after.",
        "cones_nm": {"L": l, "S": s, "rod": rod},
        "rgb_to_LS": m.tolist(),
        "LS_to_rgb": d.tolist(),
        "rgb_to_LS_display_inverse": np.linalg.pinv(d).tolist(),
        "rgb_to_rgb_simulated": (d @ m).tolist(),
        "rgb_to_rod": r.tolist(),
        "neutral_point_nm": spectral.neutral_point_nm(l, s),
        "method": ("rgb_to_LS: Govardovskii et al. 2000 A1 templates x 'Typical CRT Brainard 1997' primaries scaled to "
                   "sRGB luminance, quantum catch, von Kries to display white. LS_to_rgb: Vienot, Brettel & Mollon 1999 "
                   "adapted to S/L (blue (0,0,1) and yellow (1,1,0) map to themselves). rgb_to_LS_display_inverse = "
                   "pinv(LS_to_rgb): recovers [L, S] from a display texture exactly (the display lies in its 2-D range). "
                   "Night (< 0.02 cd/m2): rod only, shown grey."),
        "generated_by": "python -m zebrasight.vrpack",
    }


def haze() -> dict:
    grid, env = load_config("grid"), load_config("env")
    _, _, _, m, _, r = matrices()
    wls = list(_v(env.atmosphere.wavelengths_nm))
    ref = float(_v(env.atmosphere.reference_nm))
    k2, mor = float(_v(env.atmosphere.koschmieder_threshold)), float(_v(env.atmosphere.mor_threshold))
    eye = load_config("eye")
    out = {
        "formula": "per channel c: I = I0 * t + A * (1 - t),  t = exp(-beta_c * d)   (d in metres, linear light)",
        "angstrom": "beta_c = beta_550 * (lambda_c / 550 nm)^(-alpha)",
        "channel_wavelengths_nm": {"R": wls[0], "G": wls[1], "B": wls[2]},
        "cone_betas": ("beta_L = sum_c rgb_to_LS[0][c] * beta_c, beta_S = sum_c rgb_to_LS[1][c] * beta_c, "
                       "beta_rod = sum_c rgb_to_rod[0][c] * beta_c (rows sum to 1); exact for grey airlight mixing, "
                       "a close approximation otherwise"),
        "visual_range": f"Koschmieder V = ln(1/{k2}) / beta_550 = 3.912 / beta_550; WMO MOR = ln(1/{mor}) / beta_550 = 2.996 / beta_550",
        "conditions": {},
    }
    for cid in grid.core_conditions:
        c = grid.conditions[cid]
        aerosol = str(c.get("aerosol", "clear"))
        b550 = float(c.beta_haze)
        kind = "haze" if aerosol == "clear" else aerosol
        alpha = float(_v(env.atmosphere.aerosols[kind].alpha))
        brgb = channel_betas(b550, alpha, wls, ref)
        a_rel = airlight_rgb(list(_v(env.atmosphere.aerosols[kind].airlight_rgb)),
                             float(_v(env.atmosphere.aerosols[kind].airlight_luminance)))
        lum = float(_v(env.atmosphere.aerosols[kind].airlight_luminance))
        applied = aerosol != "clear" or bool(_v(env.atmosphere.apply_to_clear))
        out["conditions"][cid] = {
            "aerosol": aerosol, "alias_of": OmegaConf.to_container(grid.store_aliases).get(cid),
            "applied_in_model_textures": applied,
            "beta_550_per_m": b550, "angstrom_alpha": alpha,
            "beta_rgb_per_m": dict(zip("RGB", brgb)),
            "beta_LS_per_m": {"L": float(m[0] @ np.array(brgb)), "S": float(m[1] @ np.array(brgb))},
            "beta_rod_per_m": float(r[0] @ np.array(brgb)),
            "airlight_linear_rgb_relative": a_rel,
            "airlight_LS_relative": (m @ np.array(a_rel)).tolist(),
            "airlight_rod_relative": float(r[0] @ np.array(a_rel)),
            "airlight_by_regime": {
                reg: {"relative_linear_rgb": a_rel,
                      "absolute_luminance_cd_m2": lum * float(_v(eye.luminance[reg]))} for reg in REGIMES},
            "koschmieder_range_m": visual_range_m(b550, k2), "mor_m": visual_range_m(b550, mor),
        }
    out["note"] = ("Airlight is relative to a sunlit white (= 1) because the textures are relative too; the haze/scene "
                   "ratio is the same in every regime. Only the absolute airlight (cd/m2) dims at dusk/night. In the "
                   "model, aerosol 'clear' (c0, c4-c9) is NOT hazed: c0 textures contain no haze at all.")
    return out


def light() -> dict:
    eye, grid = load_config("eye"), load_config("grid")
    thr = float(_v(eye.colour.threshold_cd_m2))
    q0 = float(_v(eye.noise.photon_scale))
    sigma = float(_v(eye.noise.neural_sigma))
    out = {"regimes": {}, "aliases": {"dawn": "dusk"},
           "conditions": {cid: str(grid.conditions[cid].light) for cid in grid.core_conditions}}
    for reg in REGIMES:
        lum = float(_v(eye.luminance[reg]))
        t_ms = float(_v(eye.temporal.integration_ms_night if reg == "night" else eye.temporal.integration_ms_day))
        n_white = q0 * lum * t_ms / 1000.0
        acuity = float(_v(eye.acuity[LIGHT_TO_REGIME[REGIME_LIGHT[reg]]]))
        out["regimes"][reg] = {
            "luminance_cd_m2": lum, "acuity_cpd": acuity, "colour": lum >= thr,
            "channels": "L + S (colour)" if lum >= thr else "rod only (grey)",
            "integration_ms": t_ms, "photon_count_white_per_px": n_white,
            "photon_noise_rel_sd_at_mid_grey": 1.0 / math.sqrt(0.5 * n_white),
            "neural_noise_sd_x_mean": sigma,
        }
    out["note"] = ("dawn (c7) is an alias of dusk (c8): same luminance regime, computed once. Colour is on when "
                   f"luminance >= {thr} cd/m2 (Roth et al. 2008). Noise levels are already baked into the textures.")
    return out


def field_of_view() -> dict:
    eye = load_config("eye")
    sigma = float(_v(eye.streak.sigma_el_deg))
    fall = float(_v(eye.streak.az_falloff_at_90))
    floor = float(_v(eye.streak.periphery_floor))
    return {
        "azimuth_convention": "0 deg = straight ahead of the zebra's head, positive to the right; elevation 0 = horizon, positive up",
        "acuity_map": {
            "formula": "A(az, el) = A_regime * max(floor, exp(-el^2 / (2 sigma_el^2)) * max(0, 1 - (1 - falloff_90) * |az| / 90))",
            "sigma_el_deg": sigma, "falloff_90": fall, "floor": floor, "grade": "A (floor, streak) / C (sigma, falloff)",
            "source": "configs/eye.yaml streak block (Timney & Keil 1992; Guo & Sugita 2000; Harman et al. 1999)",
        },
        "total_horizontal_fov_deg": {"value": 350, "grade": "C", "status": "approximate",
                                     "note": "no verified source found; commonly quoted for horses"},
        "binocular_zone_deg": {"value": 65, "sweep": [65, 80], "grade": "B",
                               "source": "65 deg Crispin 1990, 80 deg Harman et al. 1999, as cited in Frontiers Vet Sci 2025 "
                                         "doi 10.3389/fvets.2025.1647911 (secondary)",
                               "centre_az_deg": 0, "note": "pointed down the nose"},
        "blind_zones": [
            {"name": "behind the head", "az_deg": [175, 185], "grade": "C", "status": "approximate",
             "note": "360 - total FOV; width depends on head carriage"},
            {"name": "in front of the forehead / below the nose", "az_deg": [-15, 15], "el_deg": [-90, -40],
             "grade": "C", "status": "approximate", "note": "Harman et al. 1999 describe a blind area in front of the forehead"},
        ],
        "panoramic_addon": "not implemented: values here come from configs/eye.yaml plus the approximate zones above",
    }


def loupe() -> dict:
    eye = load_config("eye")
    acuity = float(_v(eye.acuity.photopic))
    ppd = 25.0
    display_max = ppd / 2.0
    return {
        "headset": "Meta Quest 3", "pixels_per_degree": ppd, "pixels_per_degree_grade": "C (approximate headset spec)",
        "display_max_cpd": display_max, "zebra_acuity_cpd": acuity,
        "magnification": acuity / display_max,
        "formula": "M = zebra acuity / display Nyquist = 23.3 / (25 / 2) = 1.86",
        "hud": "the HUD always shows the TRUE distance; the loupe only enlarges the image angularly by M",
        "why": ("The headset can show at most ~12.5 cycles/deg, the zebra resolves 23.3. Without the loupe, detail the "
                "zebra can see would be lost on the display. The loupe magnifies the zebra's view by M so the finest "
                "zebra-resolvable stripes fit the headset's pixels."),
    }


ALL = {"dichromat_matrix.json": dichromat, "haze.json": haze, "light.json": light,
       "field_of_view.json": field_of_view, "loupe.json": loupe}
