"""Feature registry: mirrors the stage registry in zebrasight/registry.py."""

from __future__ import annotations

from typing import Callable

FEATURES: dict[str, Callable] = {}


def register_feature(name: str):
    def deco(fn):
        if name in FEATURES:
            raise KeyError(f"feature {name!r} already registered")
        fn.feature_name = name
        FEATURES[name] = fn
        return fn

    return deco


def extract(percept: dict, distance_m: float, names=None, cfg=None) -> dict[str, float]:
    """All (or the named) features of one percept; NaN everywhere if the percept is too small."""
    from zebrasight.features.common import prepare

    ctx = prepare(percept, distance_m, cfg)
    out: dict[str, float] = {}
    for name in names or FEATURES:
        out.update(FEATURES[name](percept, distance_m, ctx))
    return out
