"""Stripe features on zebra-eye percepts (person B).

Each feature is registered with ``@register_feature("F1")`` and is a pure function
``f(percept: dict, distance_m: float, ctx: Ctx | None = None) -> dict[str, float]``.
Shared preprocessing (Weber normalisation, Otsu, skeleton, structure tensor) lives in
``zebrasight.features.common.prepare`` and is computed once per percept.
Thresholds: configs/features.yaml. Runner: ``python -m zebrasight.features.run``.
"""

from zebrasight.features.registry import FEATURES, extract, register_feature  # noqa: F401
from zebrasight.features import stripes  # noqa: F401,E402  (registers F1-F10, L, M)
