"""Stage base class and name registry."""

from __future__ import annotations

from typing import Any, Callable

from omegaconf import DictConfig, OmegaConf
from torch import nn

from zebrasight.types import ViewState

STAGES: dict[str, type["Stage"]] = {}


def register_stage(name: str) -> Callable[[type["Stage"]], type["Stage"]]:
    def deco(cls: type["Stage"]) -> type["Stage"]:
        if name in STAGES:
            raise KeyError(f"stage {name!r} already registered by {STAGES[name].__name__}")
        cls.stage_name = name
        STAGES[name] = cls
        return cls

    return deco


class Stage(nn.Module):
    """A pipeline stage: ``forward(state) -> state``.

    ``eye_cfg`` is the full eye.yaml config; stages pick the blocks they need.
    """

    stage_name: str = "?"

    def __init__(self, eye_cfg: DictConfig | None = None):
        super().__init__()
        self.eye_cfg = eye_cfg if eye_cfg is not None else OmegaConf.create({})

    def param(self, key: str) -> Any:
        """Value of a parameter in eye.yaml, e.g. ``param("acuity.photopic")``."""
        node = OmegaConf.select(self.eye_cfg, key)
        if node is None:
            raise KeyError(f"eye config has no parameter {key!r}")
        if not OmegaConf.is_dict(node):  # bare override, e.g. noise.neural_sigma=0
            return OmegaConf.to_object(node) if OmegaConf.is_config(node) else node
        value = node["value"]
        return OmegaConf.to_object(value) if OmegaConf.is_config(value) else value

    def record(self, state: ViewState, **params: Any) -> None:
        """Append the parameter values this stage used to ``state.log`` (provenance)."""
        state.log.append({"stage": self.stage_name, "cond_id": state.cond.cond_id, **params})

    def forward(self, state: ViewState) -> ViewState:  # pragma: no cover - abstract
        raise NotImplementedError
