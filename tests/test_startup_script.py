from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]


class StartupScriptTests(unittest.TestCase):
    def test_start_script_launches_telegram_only_entrypoint(self) -> None:
        script = (ROOT / "start.sh").read_text(encoding="utf-8")

        self.assertTrue(script.startswith("#!/usr/bin/env bash"))
        self.assertIn("set -e", script)
        self.assertIn("exec uvicorn app.main:app", script)
        self.assertIn("--workers 1", script)

    def test_root_launcher_exports_main_and_app(self) -> None:
        launcher = ROOT / "main.py"
        spec = importlib.util.spec_from_file_location("bot_launcher", launcher)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        self.assertTrue(callable(module.main))
        self.assertTrue(hasattr(module, "app"))


if __name__ == "__main__":
    unittest.main()
