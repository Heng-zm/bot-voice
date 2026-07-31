"""Telegram Mini App administration API."""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, ConfigDict

from app._legacy_bridge import legacy_module
from app.api.dependencies import AdminPrincipal, require_admin, require_admin_write

router = APIRouter(prefix="/api/admin", tags=["admin"])
logger = logging.getLogger(__name__)

# Deliberately exclude bot mode, webhook URLs/secrets, connection pools, and
# executor sizes. Those controls have wider deployment or process-level impact
# and remain in the established advanced admin console.
MINI_APP_RUNTIME_KEYS = (
    "USER_RATE_LIMIT_PER_SECOND",
    "USER_RATE_LIMIT_WINDOW_S",
    "API_RATE_LIMIT_PER_SECOND",
    "API_RATE_LIMIT_WINDOW_S",
    "MAX_CONCURRENT_TTS_USERS",
    "MAX_CONCURRENT_AI",
    "TTS_AUDIO_CACHE_ENABLED",
    "TTS_AUDIO_CACHE_TTL_S",
)


class AdminSettingsPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    maintenance_mode: bool | None = None
    runtime: dict[str, Any] | None = None


async def _redis_health(redis_client: Any | None) -> dict[str, Any]:
    if redis_client is None:
        return {"ok": False, "latency_ms": None, "status": "not_configured"}
    started = time.perf_counter()
    try:
        pong = await asyncio.wait_for(
            asyncio.to_thread(redis_client.ping),
            timeout=2.5,
        )
    except Exception as exc:  # noqa: BLE001 - health output is intentionally safe
        logger.warning("Admin Mini App Redis health check failed: %s", exc)
        return {"ok": False, "latency_ms": None, "status": "unavailable"}
    return {
        "ok": bool(pong),
        "latency_ms": round((time.perf_counter() - started) * 1_000, 1),
        "status": "healthy" if pong else "unavailable",
    }


def _runtime_settings_payload(legacy: Any) -> dict[str, dict[str, Any]]:
    values: dict[str, dict[str, Any]] = {}
    specs = getattr(legacy, "_RUNTIME_CONFIG_SPECS", {})
    run_state = getattr(legacy, "RUN_STATE", {})
    for key in MINI_APP_RUNTIME_KEYS:
        spec = dict(specs.get(key) or {})
        value = run_state.get(key, getattr(legacy, key, None))
        values[key] = {
            "value": value,
            "kind": str(spec.get("kind") or "str"),
            "label": str(spec.get("label") or key.replace("_", " ").title()),
            "help": str(spec.get("help") or ""),
            "min": spec.get("min"),
            "max": spec.get("max"),
        }
    return values


@router.get("/me")
async def get_admin_profile(
    principal: Annotated[AdminPrincipal, Depends(require_admin)],
) -> dict[str, Any]:
    user = principal.telegram_user
    if user is not None:
        profile = user.as_public_dict()
    else:
        profile = {
            "id": principal.admin_id,
            "first_name": "Administrator",
            "last_name": "",
            "username": "",
            "language_code": "",
            "photo_url": "",
            "is_premium": False,
        }
    return {
        "ok": True,
        "auth_method": principal.auth_method,
        "user": profile,
    }


@router.get("/stats")
async def get_admin_stats(
    principal: Annotated[AdminPrincipal, Depends(require_admin)],
) -> dict[str, Any]:
    del principal
    legacy = legacy_module()
    counts_result, settings_result, redis_health = await asyncio.gather(
        asyncio.to_thread(legacy._web_counts, False),
        legacy.get_bot_settings_async(),
        _redis_health(getattr(legacy, "redis_client", None)),
        return_exceptions=True,
    )

    counts = counts_result if isinstance(counts_result, dict) else {}
    if isinstance(settings_result, tuple) and len(settings_result) == 2:
        settings, settings_status = settings_result
    else:
        settings = dict(getattr(legacy, "BOT_SETTING_DEFAULTS", {}))
        settings_status = {"db_ok": False, "memory": True}
    if not isinstance(redis_health, dict):
        redis_health = {"ok": False, "latency_ms": None, "status": "unavailable"}

    metrics = dict(getattr(legacy, "_RUNTIME_METRICS", {}))
    message_count = sum(
        int(metrics.get(key) or 0)
        for key in ("tts", "ocr", "voice", "audio", "audio_to_voice")
    )
    maintenance_mode = bool(
        legacy._setting_bool_from(settings, "maintenance_mode", False)
    )
    bot_mode = str(legacy._run_state_bot_mode())
    polling_active = bool(getattr(legacy, "_TELEGRAM_POLLING_ACTIVE", False))
    telegram_app_ready = bool(getattr(legacy, "_TELEGRAM_APP", None))

    return {
        "ok": True,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "bot": {
            "active": telegram_app_ready or polling_active,
            "mode": bot_mode,
            "polling_active": polling_active,
            "maintenance_mode": maintenance_mode,
            "uptime": str(legacy._format_uptime()),
        },
        "usage": {
            "total_users": int(counts.get("users") or 0),
            "blocked_users": int(counts.get("blocked") or 0),
            "message_count": message_count,
            "metrics": metrics,
        },
        "redis": redis_health,
        "database": {
            "ok": bool(settings_status.get("db_ok")),
            "memory_fallback": bool(settings_status.get("memory")),
        },
    }


@router.get("/settings")
async def get_admin_settings(
    principal: Annotated[AdminPrincipal, Depends(require_admin)],
) -> dict[str, Any]:
    del principal
    legacy = legacy_module()
    settings, settings_status = await legacy.get_bot_settings_async()
    return {
        "ok": True,
        "maintenance_mode": bool(
            legacy._setting_bool_from(settings, "maintenance_mode", False)
        ),
        "runtime": _runtime_settings_payload(legacy),
        "settings_store": {
            "database_ok": bool(settings_status.get("db_ok")),
            "memory_fallback": bool(settings_status.get("memory")),
        },
    }


@router.post("/settings")
async def update_admin_settings(
    payload: AdminSettingsPayload,
    principal: Annotated[AdminPrincipal, Depends(require_admin_write)],
) -> dict[str, Any]:
    if payload.maintenance_mode is None and not payload.runtime:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Provide maintenance_mode or at least one runtime setting.",
        )

    legacy = legacy_module()
    requested_runtime = dict(payload.runtime or {})
    unsupported = sorted(set(requested_runtime) - set(MINI_APP_RUNTIME_KEYS))
    if unsupported:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Unsupported runtime setting(s): {', '.join(unsupported)}",
        )

    coerced_runtime: dict[str, Any] = {}
    for key, value in requested_runtime.items():
        try:
            coerced_runtime[key] = legacy._coerce_run_state_value(key, value)
        except (TypeError, ValueError) as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"{key}: {exc}",
            ) from exc

    changed: list[str] = []
    if payload.maintenance_mode is not None:
        ok, message = await asyncio.to_thread(
            legacy.db_bot_setting_set,
            "maintenance_mode",
            payload.maintenance_mode,
            principal.admin_id,
        )
        if not ok:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Could not update maintenance mode: {message}",
            )
        await legacy.get_bot_settings_async(force=True)
        changed.append("maintenance_mode")

    for key, value in coerced_runtime.items():
        current = getattr(legacy, "RUN_STATE", {}).get(
            key,
            getattr(legacy, key, None),
        )
        if current == value:
            continue
        try:
            await legacy._update_run_state(key, value, persist=True)
        except Exception as exc:
            logger.exception(
                "Mini App runtime update failed admin_id=%s key=%s",
                principal.admin_id,
                key,
            )
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Could not update {key}.",
            ) from exc
        changed.append(key)

    try:
        legacy._web_admin_audit(
            "mini_app_settings_update",
            f"admin_id={principal.admin_id} changed={','.join(changed) or 'none'}",
        )
    except Exception:
        logger.debug("Could not record Mini App settings audit.", exc_info=True)

    settings, _settings_status = await legacy.get_bot_settings_async(force=True)
    return {
        "ok": True,
        "changed": changed,
        "maintenance_mode": bool(
            legacy._setting_bool_from(settings, "maintenance_mode", False)
        ),
        "runtime": _runtime_settings_payload(legacy),
    }


__all__ = [
    "MINI_APP_RUNTIME_KEYS",
    "AdminSettingsPayload",
    "router",
]
