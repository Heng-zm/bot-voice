from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_start_script_launches_telegram_only_entrypoint() -> None:
    script = (ROOT / "start.sh").read_text(encoding="utf-8")

    assert script.startswith("#!/usr/bin/env bash")
    assert "set -e" in script
    assert "exec uvicorn app.main:app" in script
    assert "--workers 1" in script


def test_root_launcher_exports_main_and_app() -> None:
    launcher = ROOT / "main.py"
    spec = importlib.util.spec_from_file_location("bot_launcher", launcher)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert callable(module.main)
    assert hasattr(module, "app")

