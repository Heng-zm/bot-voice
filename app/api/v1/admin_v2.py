"""Admin Panel V2 operations: analytics, exports, backups and operations."""

from __future__ import annotations

import csv
import io
import json
import logging
from collections import defaultdict
from datetime import UTC, date, datetime, timedelta
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from pydantic import BaseModel, ConfigDict, Field

from app._legacy_bridge import legacy_module
from app.api.dependencies import AdminPrincipal, require_admin, require_admin_write
from app.api.v1.admin import MINI_APP_RUNTIME_KEYS
from app.services.ai.providers import get_provider_manager

router = APIRouter(prefix="/api/admin", tags=["admin-v2"])
logger = logging.getLogger(__name__)

_SENSITIVE_SETTING_PARTS = ("token", "secret", "password", "api_key", "private_key")
_USAGE_FETCH_LIMIT = 20_000


class CacheClearPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    deep: bool = False
    templates: bool = False
    tts_audio: bool = False


class BackupRestorePayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    settings: dict[str, Any] = Field(default_factory=dict)
    runtime: dict[str, Any] = Field(default_factory=dict)


class BroadcastTestPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1, max_length=4096)
    photo_file_id: str | None = Field(default=None, max_length=512)
    parse_mode: str = Field(default="auto", max_length=32)
    link_preview: bool = True


def _audit(legacy: Any, action: str, detail: str) -> None:
    try:
        legacy._web_admin_audit(action, detail)
    except Exception:
        logger.debug("Could not record admin V2 audit.", exc_info=True)


def _is_sensitive(key: str) -> bool:
    normalized = str(key or "").lower()
    return any(part in normalized for part in _SENSITIVE_SETTING_PARTS)


def _safe_settings(settings: dict[str, Any]) -> dict[str, Any]:
    return {
        str(key): value
        for key, value in settings.items()
        if not _is_sensitive(str(key))
    }


def _parse_event_day(value: Any) -> date | None:
    try:
        if isinstance(value, datetime):
            dt = value
        else:
            raw = str(value or "").strip().replace("Z", "+00:00")
            dt = datetime.fromisoformat(raw)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        return dt.astimezone(UTC).date()
    except (TypeError, ValueError, OverflowError):
        return None


def _fetch_text_cache_usage(legacy: Any, start: datetime) -> list[dict[str, Any]]:
    supabase = getattr(legacy, "supabase", None)
    if not supabase:
        return []

    def _query() -> Any:
        return (
            supabase.table("text_cache")
            .select("created_at,user_id")
            .gte("created_at", start.isoformat())
            .order("created_at", desc=False)
            .limit(_USAGE_FETCH_LIMIT)
            .execute()
        )

    try:
        result = legacy.db_call_sync(
            "admin_v2_usage_events",
            _query,
            default=None,
            attempts=1,
            critical=False,
        )
        return [dict(row) for row in (getattr(result, "data", None) or []) if isinstance(row, dict)]
    except Exception:
        logger.debug("Could not read text_cache usage events.", exc_info=True)
        return []


def _usage_payload(legacy: Any, days: int) -> dict[str, Any]:
    days = max(1, min(int(days or 30), 90))
    today = datetime.now(UTC).date()
    start_day = today - timedelta(days=days - 1)
    persistent_rows = _fetch_text_cache_usage(
        legacy,
        datetime.combine(start_day, datetime.min.time(), tzinfo=UTC),
    )
    daily: dict[date, dict[str, Any]] = {
        start_day + timedelta(days=index): {
            "date": (start_day + timedelta(days=index)).isoformat(),
            "requests": 0,
            "users": set(),
            "audio_generation_ms": 0.0,
        }
        for index in range(days)
    }
    for row in persistent_rows:
        day = _parse_event_day(row.get("created_at"))
        if day in daily:
            daily[day]["requests"] += 1
            try:
                uid = int(row.get("user_id") or 0)
            except (TypeError, ValueError):
                uid = 0
            if uid:
                daily[day]["users"].add(uid)

    memory_events = list(getattr(legacy, "_admin_usage_events_snapshot", lambda: [])())
    for event in memory_events:
        day = _parse_event_day(event.get("at"))
        if day not in daily:
            continue
        feature = str(event.get("feature") or "")
        # The generation event enriches the preceding request; it is not a
        # second request in the chart.
        if (not persistent_rows) and not feature.endswith("_generation"):
            daily[day]["requests"] += max(1, int(event.get("amount") or 1))
        try:
            uid = int(event.get("user_id") or 0)
        except (TypeError, ValueError):
            uid = 0
        if uid:
            daily[day]["users"].add(uid)
        daily[day]["audio_generation_ms"] += max(0.0, float(event.get("duration_ms") or 0.0))

    daily_rows = [
        {
            "date": item["date"],
            "requests": int(item["requests"]),
            "users": len(item["users"]),
            "audio_generation_ms": round(float(item["audio_generation_ms"]), 3),
        }
        for item in daily.values()
    ]

    def _rollup(bucket: Literal["week", "month"]) -> list[dict[str, Any]]:
        grouped: dict[str, dict[str, Any]] = {}
        for item in daily_rows:
            current = date.fromisoformat(item["date"])
            if bucket == "week":
                key = (current - timedelta(days=current.weekday())).isoformat()
            else:
                key = current.replace(day=1).isoformat()
            target = grouped.setdefault(
                key,
                {"period": key, "requests": 0, "users": 0, "audio_generation_ms": 0.0},
            )
            target["requests"] += int(item["requests"])
            target["users"] += int(item["users"])
            target["audio_generation_ms"] += float(item["audio_generation_ms"])
        return [
            {**item, "audio_generation_ms": round(item["audio_generation_ms"], 3)}
            for item in sorted(grouped.values(), key=lambda value: value["period"])
        ]

    return {
        "ok": True,
        "generated_at": datetime.now(UTC).isoformat(),
        "days": days,
        "source": "supabase_text_cache+process_events" if persistent_rows else "process_events",
        "daily": daily_rows,
        "weekly": _rollup("week"),
        "monthly": _rollup("month"),
    }


def _user_usage_payload(legacy: Any, limit: int) -> dict[str, Any]:
    limit = max(1, min(int(limit or 50), 500))
    start = datetime.now(UTC) - timedelta(days=90)
    rows = _fetch_text_cache_usage(legacy, start)
    grouped: dict[int, dict[str, Any]] = defaultdict(
        lambda: {"user_id": 0, "request_count": 0, "audio_generation_ms": 0.0, "last_request_at": ""}
    )
    for row in rows:
        try:
            uid = int(row.get("user_id") or 0)
        except (TypeError, ValueError):
            uid = 0
        if uid <= 0:
            continue
        item = grouped[uid]
        item["user_id"] = uid
        item["request_count"] += 1
        created = str(row.get("created_at") or "")
        if created > str(item["last_request_at"] or ""):
            item["last_request_at"] = created

    for event in list(getattr(legacy, "_admin_usage_events_snapshot", lambda: [])()):
        try:
            uid = int(event.get("user_id") or 0)
        except (TypeError, ValueError):
            uid = 0
        if uid <= 0:
            continue
        item = grouped[uid]
        item["user_id"] = uid
        feature = str(event.get("feature") or "")
        if (not rows) and not feature.endswith("_generation"):
            item["request_count"] += max(1, int(event.get("amount") or 1))
        item["audio_generation_ms"] += max(0.0, float(event.get("duration_ms") or 0.0))
        item["last_request_at"] = max(str(item["last_request_at"] or ""), str(event.get("at") or ""))

    names: dict[int, str] = {}
    try:
        user_rows = legacy._get_user_search_rows_cached(force=False)
    except Exception:
        user_rows = []
    if not user_rows:
        try:
            user_rows = legacy.get_all_users_with_names()
        except Exception:
            user_rows = []
    for row in user_rows or []:
        try:
            uid = int(row.get("user_id") or 0)
        except (TypeError, ValueError):
            uid = 0
        if uid:
            names[uid] = str(row.get("username") or row.get("first_name") or "").strip()
            grouped.setdefault(
                uid,
                {"user_id": uid, "request_count": 0, "audio_generation_ms": 0.0, "last_request_at": ""},
            )

    result = []
    for item in grouped.values():
        item["username"] = names.get(int(item["user_id"]), "")
        item["audio_generation_ms"] = round(float(item["audio_generation_ms"]), 3)
        result.append(item)
    result.sort(key=lambda value: (int(value["request_count"]), float(value["audio_generation_ms"])), reverse=True)
    return {"ok": True, "window_days": 90, "users": result[:limit], "count": len(result)}


async def _backup_payload(legacy: Any) -> dict[str, Any]:
    settings, settings_status = await legacy.get_bot_settings_async()
    runtime = {}
    run_state = getattr(legacy, "RUN_STATE", {})
    for key in MINI_APP_RUNTIME_KEYS:
        runtime[key] = run_state.get(key, getattr(legacy, key, None))
    return {
        "version": 2,
        "created_at": datetime.now(UTC).isoformat(),
        "settings": _safe_settings(dict(settings or {})),
        "runtime": runtime,
        "settings_store": settings_status,
    }


@router.get("/analytics")
async def admin_analytics(
    principal: Annotated[AdminPrincipal, Depends(require_admin)],
    days: Annotated[int, Query(ge=1, le=90)] = 30,
) -> dict[str, Any]:
    del principal
    return await __import__("asyncio").to_thread(_usage_payload, legacy_module(), days)


@router.get("/usage/users")
async def admin_user_usage(
    principal: Annotated[AdminPrincipal, Depends(require_admin)],
    limit: Annotated[int, Query(ge=1, le=500)] = 50,
) -> dict[str, Any]:
    del principal
    return await __import__("asyncio").to_thread(_user_usage_payload, legacy_module(), limit)


@router.get("/cache")
async def admin_cache_snapshot(
    principal: Annotated[AdminPrincipal, Depends(require_admin)],
) -> dict[str, Any]:
    del principal
    legacy = legacy_module()
    snapshot = await __import__("asyncio").to_thread(legacy._admin_runtime_cache_snapshot_sync)
    return {"ok": True, "snapshot": snapshot}


@router.get("/activity")
async def admin_activity(
    principal: Annotated[AdminPrincipal, Depends(require_admin)],
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
) -> dict[str, Any]:
    """Return the in-process admin activity timeline used by V2 operations."""
    del principal
    legacy = legacy_module()
    entries = [dict(item) for item in list(getattr(legacy, "_WEB_ADMIN_AUDIT", ()))[:limit]]
    return {"ok": True, "entries": entries, "count": len(entries), "scope": "process"}


@router.post("/cache/clear")
async def admin_cache_clear(
    payload: CacheClearPayload,
    principal: Annotated[AdminPrincipal, Depends(require_admin_write)],
) -> dict[str, Any]:
    legacy = legacy_module()
    message = await __import__("asyncio").to_thread(
        legacy._admin_clear_runtime_caches_sync,
        deep=payload.deep,
        templates=payload.templates,
        tts_audio=payload.tts_audio,
    )
    _audit(legacy, "admin_v2_cache_clear", f"admin_id={principal.admin_id} {message[:500]}")
    snapshot = await __import__("asyncio").to_thread(legacy._admin_runtime_cache_snapshot_sync)
    return {"ok": True, "message": message, "snapshot": snapshot}


@router.get("/backup")
async def admin_backup(
    principal: Annotated[AdminPrincipal, Depends(require_admin)],
) -> dict[str, Any]:
    del principal
    return await _backup_payload(legacy_module())


@router.post("/backup/restore")
async def admin_backup_restore(
    payload: BackupRestorePayload,
    principal: Annotated[AdminPrincipal, Depends(require_admin_write)],
) -> dict[str, Any]:
    legacy = legacy_module()
    allowed_settings = set(getattr(legacy, "BOT_SETTING_DEFAULTS", {}))
    unknown = sorted(set(payload.settings) - allowed_settings)
    if unknown:
        raise HTTPException(status_code=422, detail=f"Unsupported setting(s): {', '.join(unknown)}")
    unknown_runtime = sorted(set(payload.runtime) - set(MINI_APP_RUNTIME_KEYS))
    if unknown_runtime:
        raise HTTPException(status_code=422, detail=f"Unsupported runtime setting(s): {', '.join(unknown_runtime)}")

    changed: list[str] = []
    for key, value in payload.settings.items():
        if _is_sensitive(key):
            continue
        ok, message = await __import__("asyncio").to_thread(
            legacy.db_bot_setting_value_set,
            key,
            value,
            principal.admin_id,
        )
        if not ok:
            raise HTTPException(status_code=503, detail=f"Could not restore {key}: {message}")
        changed.append(key)
    for key, value in payload.runtime.items():
        try:
            coerced = legacy._coerce_run_state_value(key, value)
            await legacy._update_run_state(key, coerced, persist=True)
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=f"{key}: {exc}") from exc
        changed.append(key)
    await legacy.get_bot_settings_async(force=True)
    _audit(legacy, "admin_v2_backup_restore", f"admin_id={principal.admin_id} changed={','.join(changed) or 'none'}")
    return {"ok": True, "changed": changed, "backup": await _backup_payload(legacy)}


@router.get("/export/{dataset}")
async def admin_export(
    dataset: Literal["users", "settings", "usage"],
    principal: Annotated[AdminPrincipal, Depends(require_admin)],
    format: Annotated[Literal["json", "csv"], Query()] = "json",
    days: Annotated[int, Query(ge=1, le=90)] = 30,
) -> Response:
    legacy = legacy_module()
    if dataset == "users":
        payload = await __import__("asyncio").to_thread(_user_usage_payload, legacy, 500)
        rows = payload["users"]
    elif dataset == "usage":
        payload = await __import__("asyncio").to_thread(_usage_payload, legacy, days)
        rows = payload["daily"]
    else:
        payload = await _backup_payload(legacy)
        rows = [{"key": key, "value": value} for key, value in payload["settings"].items()]
    _audit(legacy, "admin_v2_export", f"admin_id={principal.admin_id} dataset={dataset} format={format}")
    if format == "csv":
        output = io.StringIO()
        if rows:
            fields = sorted({str(key) for row in rows for key in row})
            writer = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        content = output.getvalue()
        return Response(
            content=content,
            media_type="text/csv; charset=utf-8",
            headers={"Content-Disposition": f'attachment; filename="bot-{dataset}.csv"'},
        )
    return Response(
        content=json.dumps(payload, ensure_ascii=False, default=str),
        media_type="application/json",
        headers={"Content-Disposition": f'attachment; filename="bot-{dataset}.json"'},
    )


@router.post("/broadcast/test")
async def admin_broadcast_test(
    payload: BroadcastTestPayload,
    principal: Annotated[AdminPrincipal, Depends(require_admin_write)],
) -> dict[str, Any]:
    legacy = legacy_module()
    application = getattr(legacy, "_TELEGRAM_APP", None) or getattr(legacy, "telegram_application", None)
    if application is None or getattr(application, "bot", None) is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Telegram bot is not ready.")
    max_chars = 1024 if payload.photo_file_id else int(getattr(legacy, "TELE_MSG_LIMIT", 4096))
    try:
        prepared, mode, link_preview = legacy._broadcast_prepare_text(
            payload.text,
            payload.parse_mode,
            max_chars=max_chars,
            default_link_preview=payload.link_preview,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if not prepared:
        raise HTTPException(status_code=422, detail="Broadcast text is empty after formatting directives.")
    try:
        message = await legacy._send_telegram_broadcast_message(
            application.bot,
            chat_id=principal.admin_id,
            text=prepared,
            parse_mode=mode,
            photo_file_id=payload.photo_file_id,
            link_preview=link_preview,
        )
    except Exception as exc:
        _audit(legacy, "admin_v2_broadcast_test_failed", f"admin_id={principal.admin_id} error={str(exc)[:300]}")
        raise HTTPException(status_code=502, detail="Telegram test delivery failed.") from exc
    _audit(legacy, "admin_v2_broadcast_test", f"admin_id={principal.admin_id} photo={bool(payload.photo_file_id)}")
    return {"ok": True, "message_id": getattr(message, "message_id", None), "parse_mode": mode}


@router.get("/schedules/failures")
async def admin_schedule_failures(
    principal: Annotated[AdminPrincipal, Depends(require_admin)],
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
) -> dict[str, Any]:
    del principal
    legacy = legacy_module()
    supabase = getattr(legacy, "supabase", None)
    if not supabase:
        return {"ok": True, "failures": [], "count": 0, "persistent": False}

    def _query() -> Any:
        return (
            supabase.table("scheduled_broadcasts")
            .select("id,admin_id,broadcast_at,status,error_msg,failed_count,plain_text,caption")
            .eq("status", "failed")
            .order("broadcast_at", desc=True)
            .limit(limit)
            .execute()
        )

    result = await __import__("asyncio").to_thread(
        legacy.db_call_sync,
        "admin_v2_schedule_failures",
        _query,
        default=None,
        attempts=1,
        critical=False,
    )
    rows = [dict(row) for row in (getattr(result, "data", None) or [])]
    return {"ok": True, "failures": rows, "count": len(rows), "persistent": True}


@router.post("/schedules/{schedule_id}/retry")
async def admin_schedule_retry(
    schedule_id: int,
    principal: Annotated[AdminPrincipal, Depends(require_admin_write)],
) -> dict[str, Any]:
    legacy = legacy_module()
    row = await __import__("asyncio").to_thread(legacy.db_sched_fetch_one, schedule_id)
    if not row:
        raise HTTPException(status_code=404, detail="Schedule not found.")
    if str(row.get("status") or "").lower() != "failed":
        raise HTTPException(status_code=409, detail="Only failed schedules can be retried.")
    now = datetime.now(UTC)
    saved = await __import__("asyncio").to_thread(
        legacy.db_sched_set_status,
        schedule_id,
        "pending",
        critical=True,
        broadcast_at=legacy._sched_iso(now),
        error_msg=None,
    )
    if not saved:
        raise HTTPException(status_code=503, detail="Could not queue schedule retry.")
    _audit(legacy, "admin_v2_schedule_retry", f"admin_id={principal.admin_id} schedule_id={schedule_id}")
    return {"ok": True, "schedule_id": schedule_id, "status": "pending", "retry_at": now.isoformat()}


@router.get("/providers/health")
async def admin_provider_health(
    principal: Annotated[AdminPrincipal, Depends(require_admin)],
) -> dict[str, Any]:
    del principal
    providers = get_provider_manager().snapshot()
    # Keep cards stable even before the first request touches a provider.
    for name in ("gemini", "huggingface", "edge_tts"):
        providers.setdefault(
            name,
            {
                "scope": "process",
                "available": True,
                "health_score": 100.0,
                "successes": 0,
                "failures": 0,
                "latency_ewma_ms": None,
                "cooldown_remaining_seconds": 0,
                "capabilities": [],
                "unobserved": True,
            },
        )
    return {"ok": True, "providers": providers, "count": len(providers), **get_provider_manager().metadata()}


__all__ = ["router"]
