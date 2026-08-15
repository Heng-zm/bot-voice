"""Safe bounded environment parsing for modern runtime components."""

from __future__ import annotations

import math
import os
from collections.abc import Mapping


def bounded_env_int(
    name: str,
    default: int,
    *,
    minimum: int,
    maximum: int,
    environ: Mapping[str, str] | None = None,
) -> int:
    source = os.environ if environ is None else environ
    try:
        value = int(str(source.get(name, default) or default).strip())
    except (TypeError, ValueError):
        value = int(default)
    return max(int(minimum), min(int(maximum), value))


def bounded_env_float(
    name: str,
    default: float,
    *,
    minimum: float,
    maximum: float,
    environ: Mapping[str, str] | None = None,
) -> float:
    source = os.environ if environ is None else environ
    try:
        value = float(str(source.get(name, default) or default).strip())
    except (TypeError, ValueError):
        value = float(default)
    if not math.isfinite(value):
        value = float(default)
    return max(float(minimum), min(float(maximum), value))


__all__ = ["bounded_env_float", "bounded_env_int"]
