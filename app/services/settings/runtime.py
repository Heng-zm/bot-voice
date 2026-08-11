"""Pure validation and presentation helpers for runtime configuration."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from typing import Any
from urllib.parse import urlsplit


def coerce_runtime_value(key: str, value: Any, spec: Mapping[str, Any]) -> Any:
    kind = str(spec.get("kind") or "str")
    if kind == "mode":
        mode = str(value or "POLLING").strip().upper()
        if mode not in {"POLLING", "WEBHOOK"}:
            raise ValueError("BOT_MODE must be POLLING or WEBHOOK")
        return mode
    if kind == "int":
        number = int(str(value).strip())
        return max(int(spec.get("min", number)), min(int(spec.get("max", number)), number))
    if kind == "float":
        number = float(str(value).strip())
        if not math.isfinite(number):
            raise ValueError(f"{key} must be a finite number")
        return max(
            float(spec.get("min", number)),
            min(float(spec.get("max", number)), number),
        )
    if kind == "bool":
        if isinstance(value, bool):
            return value
        text = str(value).strip().lower()
        if text in {"1", "true", "yes", "on", "enable", "enabled"}:
            return True
        if text in {"0", "false", "no", "off", "disable", "disabled"}:
            return False
        raise ValueError(f"{key} must be true/false or 1/0")
    if kind == "url":
        candidate = str(value or "").strip().rstrip("/")
        if not candidate:
            return ""
        if len(candidate) > 2_048:
            raise ValueError(f"{key} is too long")
        parsed = urlsplit(candidate)
        if parsed.scheme.lower() != "https" or not parsed.hostname:
            raise ValueError(f"{key} must be an HTTPS URL with a hostname")
        if parsed.username is not None or parsed.password is not None:
            raise ValueError(f"{key} must not contain credentials")
        if parsed.query or parsed.fragment:
            raise ValueError(f"{key} must not contain a query or fragment")
        try:
            port = parsed.port
        except ValueError as exc:
            raise ValueError(f"{key} contains an invalid port") from exc
        if port == 0:
            raise ValueError(f"{key} contains an invalid port")
        if any(character.isspace() for character in parsed.hostname):
            raise ValueError(f"{key} contains an invalid hostname")
        if "/tg-webhook-" in parsed.path.lower():
            raise ValueError(f"{key} must be the base URL without a webhook path")
        return candidate
    if kind == "secret":
        token = str(value or "").strip()
        if key == "TELEGRAM_WEBHOOK_SECRET_TOKEN":
            if not re.fullmatch(r"[A-Za-z0-9_-]{64}", token):
                raise ValueError(
                    "TELEGRAM_WEBHOOK_SECRET_TOKEN must contain exactly "
                    "64 URL-safe characters."
                )
        elif not token:
            raise ValueError(f"{key} cannot be empty")
        return token
    return str(value if value is not None else "").strip()


def coerce_runtime_updates(
    values: Mapping[str, Any],
    specs: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    coerced: dict[str, Any] = {}
    for key, value in values.items():
        try:
            coerced[key] = coerce_runtime_value(key, value, specs.get(key, {}))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{key}: {exc}") from exc
    return coerced


def build_runtime_settings_payload(
    source: Any,
    keys: tuple[str, ...],
) -> dict[str, dict[str, Any]]:
    values: dict[str, dict[str, Any]] = {}
    specs = getattr(source, "_RUNTIME_CONFIG_SPECS", {})
    run_state = getattr(source, "RUN_STATE", {})
    for key in keys:
        spec = dict(specs.get(key) or {})
        values[key] = {
            "value": run_state.get(key, getattr(source, key, None)),
            "kind": str(spec.get("kind") or "str"),
            "label": str(spec.get("label") or key.replace("_", " ").title()),
            "help": str(spec.get("help") or ""),
            "min": spec.get("min"),
            "max": spec.get("max"),
        }
    return values


__all__ = [
    "build_runtime_settings_payload",
    "coerce_runtime_updates",
    "coerce_runtime_value",
]
