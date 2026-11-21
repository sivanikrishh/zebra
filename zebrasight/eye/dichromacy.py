"""Dichromacy stage: linear RGB -> zebra cone signals [L, S], or rod at night.

The input is already linear RGB: ``Pipeline.run`` decodes sRGB at the input, because scene
and atmosphere also need linear RGB. Here a precomputed 2x3 matrix (derivation in
``zebrasight/eye/spectral.py``: Govardovskii 2000 A1 templates at S 430 / L 545 nm x CRT
primary spectra scaled to sRGB luminance, von Kries normalised to the display white) maps
RGB to [L, S]; grey stays grey (S/L = 1).

Below the colour threshold (0.02 cd/m2, Roth et al. 2008) only rods respond: one rod
channel (template at 500 nm, grade C), cones = None, no colour.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from zebrasight.eye import spectral
from zebrasight.registry import Stage, register_stage
from zebrasight.types import ViewState


def cone_matrix(l_peak: float, s_peak: float) -> torch.Tensor:
    return torch.tensor(spectral.rgb_to_receptor_matrix((float(l_peak), float(s_peak))), dtype=torch.float32)


def rod_matrix(rod_peak: float) -> torch.Tensor:
    return torch.tensor(spectral.rgb_to_receptor_matrix((float(rod_peak),)), dtype=torch.float32)


def apply_matrix(m: torch.Tensor, rgb: torch.Tensor) -> torch.Tensor:
    return torch.einsum("oc,bchw->bohw", m.to(rgb.device, rgb.dtype), rgb)


def export_display_matrices(path: Path, l_peak: float = 545.0, s_peak: float = 430.0, rod_peak: float = 500.0) -> dict:
    """Write the matrices V's VR shader uses, so it applies exactly the numbers the model uses."""
    m = spectral.rgb_to_receptor_matrix((float(l_peak), float(s_peak)))
    d = spectral.dichromat_display_matrix(m)
    payload = {
        "description": "ZebraSight zebra dichromat matrices. All RGB is LINEAR sRGB (decode before, encode after).",
        "cones_nm": {"L": l_peak, "S": s_peak, "rod": rod_peak},
        "rgb_to_LS": m.tolist(),
        "LS_to_rgb": d.tolist(),
        "rgb_to_rgb_simulated": (d @ m).tolist(),
        "rgb_to_rod": spectral.rgb_to_receptor_matrix((float(rod_peak),)).tolist(),
        "neutral_point_nm": spectral.neutral_point_nm(l_peak, s_peak),
        "method": ("rgb_to_LS: Govardovskii et al. 2000 A1 templates x 'Typical CRT Brainard 1997' primaries "
                   "scaled to sRGB luminance, quantum catch, von Kries to display white. LS_to_rgb: Vienot, "
                   "Brettel & Mollon 1999 adapted to S/L: blue (0,0,1) and yellow (1,1,0) map to themselves. "
                   "Night (< 0.02 cd/m2): grey = rgb_to_rod applied to all three channels."),
        "generated_by": "python -m zebrasight.eye.export_vr",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


@register_stage("dichromacy")
class Dichromacy(Stage):
    def __init__(self, eye_cfg=None):
        super().__init__(eye_cfg)
        self.l_peak = float(self.param("cones.L_peak"))
        self.s_peak = float(self.param("cones.S_peak"))
        self.rod_peak = float(self.param("cones.rod_peak"))
        self.register_buffer("m_cones", cone_matrix(self.l_peak, self.s_peak), persistent=False)
        self.register_buffer("m_rod", rod_matrix(self.rod_peak), persistent=False)

    def forward(self, state: ViewState) -> ViewState:
        rgb = state.rgb
        if rgb is None:
            raise RuntimeError("dichromacy needs linear RGB; an earlier stage dropped it")
        threshold = float(self.param("colour.threshold_cd_m2"))
        rod_only = state.cond.luminance_cd_m2 < threshold
        if rod_only:
            state = state.evolve(rgb=None, rod=apply_matrix(self.m_rod, rgb), cones=None)
        else:
            state = state.evolve(rgb=None, cones=apply_matrix(self.m_cones, rgb), rod=None)
        self.record(state, rod_only=rod_only, colour_threshold_cd_m2=threshold,
                    luminance_cd_m2=state.cond.luminance_cd_m2, L_peak_nm=self.l_peak, S_peak_nm=self.s_peak,
                    rod_peak_nm=self.rod_peak,
                    matrix=(self.m_rod if rod_only else self.m_cones).tolist())
        return state
