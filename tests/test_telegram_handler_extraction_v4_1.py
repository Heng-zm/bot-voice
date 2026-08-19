from __future__ import annotations

import ast
import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from app.services.telegram import _legacy_runtime

ROOT = Path(__file__).resolve().parents[1]
LEGACY = ROOT / "app" / "legacy.py"
TELEGRAM_DIR = ROOT / "app" / "services" / "telegram"

EXTRACTED = {
    "commands.py": {
        "on_start", "on_help", "cmd_myprefs", "cmd_ttsmodel", "cmd_voxcpm2",
        "cmd_clear", "cmd_security", "cmd_privacy", "cmd_delete_my_data",
        "broadcast_start", "cmd_schedule", "cmd_schedules", "cmd_cancelschedule",
        "cmd_cancel", "admin_stats", "cmd_health", "cmd_admin", "cmd_feature_request",
        "cmd_runtime", "cmd_api", "cmd_botsettings", "cmd_users", "cmd_chat", "cmd_endchat",
    },
    "callbacks.py": {
        "broadcast_callback", "users_page_callback", "sched_callback",
        "_runtime_admin_callback", "on_callback",
    },
    "media.py": {"on_photo", "on_voice", "on_audio_file", "on_any_media", "on_text"},
    "guards.py": {
        "_telegram_rate_limit_guard", "_telegram_user_security_guard",
        "_drop_stale_updates", "error_handler",
    },
}


def _functions(path: Path) -> dict[str, ast.FunctionDef | ast.AsyncFunctionDef]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def test_live_handler_bodies_are_extracted() -> None:
    for filename, expected in EXTRACTED.items():
        functions = _functions(TELEGRAM_DIR / filename)
        assert expected <= functions.keys()
        for name in expected:
            decorators = functions[name].decorator_list
            assert any(
                isinstance(item, ast.Name) and item.id == "legacy_bound_handler"
                for item in decorators
            ), name


def test_legacy_handlers_are_compatibility_wrappers_only() -> None:
    functions = _functions(LEGACY)
    for expected in EXTRACTED.values():
        for name in expected:
            node = functions[name]
            assert node.end_lineno - node.lineno <= 4, name


def test_legacy_runtime_uses_native_router() -> None:
    source = LEGACY.read_text(encoding="utf-8")
    run_bot_start = source.index("async def _run_bot")
    run_bot_source = source[run_bot_start:]
    assert "register_telegram_handlers(app, bot_mode=_run_state_bot_mode())" in run_bot_source
    assert "app.add_handler(CommandHandler" not in run_bot_source
    assert "app.add_handler(MessageHandler" not in run_bot_source


def test_extracted_media_has_no_removed_job_queue_imports() -> None:
    source = (TELEGRAM_DIR / "media.py").read_text(encoding="utf-8")
    assert "app.services.jobs" not in source
    assert "Redis worker queue removed" not in source


def test_runtime_bridge_refreshes_nested_global_dependencies() -> None:
    namespace: dict[str, object] = {"legacy_bound_handler": _legacy_runtime.legacy_bound_handler}
    exec(
        "@legacy_bound_handler\n"
        "async def sample():\n"
        "    callback = lambda: RUNTIME_VALUE\n"
        "    return callback()\n",
        namespace,
    )
    sample = namespace["sample"]
    with patch.object(
        _legacy_runtime,
        "legacy_module",
        return_value=SimpleNamespace(RUNTIME_VALUE="fresh"),
    ):
        assert asyncio.run(sample()) == "fresh"


def test_runtime_bridge_does_not_treat_attribute_names_as_legacy_globals() -> None:
    namespace: dict[str, object] = {"legacy_bound_handler": _legacy_runtime.legacy_bound_handler}
    exec(
        "@legacy_bound_handler\n"
        "async def sample(obj):\n"
        "    return obj.value.strip()\n",
        namespace,
    )
    sample = namespace["sample"]
    dependencies = getattr(sample, "__legacy_dependencies__")
    assert "value" not in dependencies
    assert "strip" not in dependencies


def test_heavy_media_handlers_use_native_workload_admission() -> None:
    source = (TELEGRAM_DIR / "media.py").read_text(encoding="utf-8")
    assert 'run_telegram_workload("ocr"' in source or 'run_telegram_workload(\n            "ocr"' in source
    assert '"transcribe"' in source
    assert '"audio"' in source
    assert "busy_rejected" in source
