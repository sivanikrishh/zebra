"""Zebra-eye model. The ``eye`` stage runs the sub-stages listed in eye.yaml ``eye_order``
whose ``eye_stages.<name>`` flag is true."""

from __future__ import annotations

from torch import nn

from zebrasight.eye import csf, dichromacy, geometry, motion, noise, streak  # noqa: F401  (register stages)
from zebrasight.registry import STAGES, Stage, register_stage
from zebrasight.types import ViewState

EYE_SUBSTAGES = ("dichromacy", "motion", "streak", "csf", "noise")


@register_stage("eye")
class Eye(Stage):
    def __init__(self, eye_cfg=None):
        super().__init__(eye_cfg)
        order = list(self.eye_cfg.get("eye_order", EYE_SUBSTAGES))
        bad = [n for n in order if n not in EYE_SUBSTAGES]
        if bad:
            raise ValueError(f"unknown eye sub-stages {bad}; allowed: {EYE_SUBSTAGES}")
        if not order or order[0] != "dichromacy":
            raise ValueError("eye_order must start with dichromacy (RGB -> cones)")
        flags = self.eye_cfg.get("eye_stages", {})
        if not flags.get("dichromacy", True):
            raise ValueError("dichromacy cannot be switched off: it is the only RGB -> receptor step")
        self.enabled = [n for n in order if flags.get(n, True)]
        self.substages = nn.ModuleList(STAGES[n](self.eye_cfg) for n in self.enabled)

    def forward(self, state: ViewState) -> ViewState:
        self.record(state, order=self.enabled)
        for s in self.substages:
            state = s(state)
        return state
