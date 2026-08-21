from __future__ import annotations

import hashlib
import hmac
import json
import time
import unittest
from pathlib import Path
from urllib.parse import urlencode

from app.core.telegram_auth import TelegramAdminAuthorizer, validate_telegram_init_data
from app.services.settings.store import SettingsStore

ROOT = Path(__file__).resolve().parents[1]
BOT_TOKEN = "123456789:TEST_bot_token_for_unit_tests"


def signed_init_data(user_id: int) -> str:
    fields = {
        "auth_date": str(int(time.time())),
        "query_id": "AA-test",
        "user": json.dumps({"id": user_id, "first_name": "Admin"}, separators=(",", ":")),
    }
    data_check = "\n".join(f"{key}={value}" for key, value in sorted(fields.items()))
    secret = hmac.new(b"WebAppData", BOT_TOKEN.encode(), hashlib.sha256).digest()
    fields["hash"] = hmac.new(secret, data_check.encode(), hashlib.sha256).hexdigest()
    return urlencode(fields)


class SingleProcessArchitectureTests(unittest.IsolatedAsyncioTestCase):
    async def test_telegram_admin_authorization_without_redis(self) -> None:
        store = SettingsStore()
        authorizer = TelegramAdminAuthorizer().configure(
            settings_store=store,
            fallback_admin_ids={42},
        )
        session = await authorizer.authorize(signed_init_data(42), BOT_TOKEN)
        self.assertEqual(42, session.user.id)
        self.assertEqual(42, validate_telegram_init_data(signed_init_data(42), BOT_TOKEN).user.id)

    def test_runtime_dependency_set_does_not_install_redis(self) -> None:
        requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8").lower().splitlines()
        self.assertFalse(any(line.strip().startswith("redis") for line in requirements))

    def test_polling_preserves_pending_updates_unless_explicitly_enabled(self) -> None:
        source = (ROOT / "app" / "legacy.py").read_text(encoding="utf-8")
        env_example = (ROOT / ".env.example").read_text(encoding="utf-8")
        self.assertIn(
            '_env_bool("TELEGRAM_POLLING_DROP_PENDING_UPDATES", False)',
            source,
        )
        self.assertNotIn("drop_pending=True", source)
        self.assertIn("TELEGRAM_POLLING_DROP_PENDING_UPDATES=false", env_example)

    def test_live_provider_defaults_are_deployable(self) -> None:
        source = (ROOT / "app" / "legacy.py").read_text(encoding="utf-8")
        requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8")
        env_example = (ROOT / ".env.example").read_text(encoding="utf-8")
        self.assertIn('DEFAULT_AI_PROVIDER              = "gemini"', source)
        self.assertIn("AI_PROVIDER=gemini", env_example)
        self.assertIn("edge-tts>=7.2.8,<8", requirements)

    def test_dedicated_worker_and_queue_files_are_removed(self) -> None:
        removed = [
            ROOT / "app" / "worker.py",
            ROOT / "app" / "services" / "jobs" / "queue.py",
            ROOT / "app" / "services" / "jobs" / "runtime.py",
            ROOT / "app" / "services" / "telegram" / "delivery.py",
        ]
        self.assertTrue(all(not path.exists() for path in removed))

    def test_admin_ui_has_no_queue_api_calls(self) -> None:
        script = (ROOT / "static" / "admin" / "app.js").read_text(encoding="utf-8")
        self.assertNotIn("/runtime/jobs", script)
        self.assertNotIn("/runtime/workers", script)
        self.assertIn("Promise.allSettled", script)

    def test_admin_ui_exposes_workload_pressure_and_four_metrics(self) -> None:
        script = (ROOT / "static" / "admin" / "app.js").read_text(encoding="utf-8")
        html = (ROOT / "static" / "admin" / "index.html").read_text(encoding="utf-8")
        css = (ROOT / "static" / "admin" / "styles.css").read_text(encoding="utf-8")
        self.assertIn("telegram_workloads", script)
        self.assertIn('id="workloadGrid"', html)
        self.assertIn("repeat(4, minmax(0, 1fr))", css)

    def test_runtime_status_exposes_webhook_replay_and_workload_snapshots(self) -> None:
        source = (ROOT / "app" / "runtime.py").read_text(encoding="utf-8")
        self.assertIn('"telegram_workloads"', source)
        self.assertIn('"webhook_replay"', source)

    def test_combined_supervisor_restarts_when_a_critical_service_stops(self) -> None:
        source = (ROOT / "app" / "main.py").read_text(encoding="utf-8")
        self.assertIn("critical_tasks", source)
        self.assertIn("return_when=asyncio.FIRST_COMPLETED", source)
        self.assertIn("Critical runtime service stopped unexpectedly", source)
        self.assertNotIn("return_when=asyncio.FIRST_EXCEPTION", source)

    def test_telegram_runtime_panel_exposes_workload_pressure(self) -> None:
        source = (ROOT / "app" / "legacy.py").read_text(encoding="utf-8")
        self.assertIn("Workload pressure", source)
        self.assertIn("get_telegram_workload_limiter", source)
        self.assertIn("get_webhook_replay_snapshot", source)

    def test_web_key_admin_action_matches_no_redis_architecture(self) -> None:
        source = (ROOT / "app" / "legacy.py").read_text(encoding="utf-8")
        start = source.index("async def _admin_generate_web_key")
        end = source.index("_CRM_TELEGRAM_SEGMENTS", start)
        action = source[start:end]
        self.assertIn("WEB_SECRET_KEY", action)
        self.assertIn("candidate", action.lower())
        self.assertNotIn("_web_secret_set_in_redis_sync", action)
        self.assertNotIn("set REDIS_URL", action)

    def test_startup_self_check_does_not_require_redis(self) -> None:
        source = (ROOT / "app" / "legacy.py").read_text(encoding="utf-8")
        start = source.index("def startup_self_check")
        end = source.index("BOT_FEATURE_SETTING_KEYS", start)
        check = source[start:end]
        self.assertNotIn("REDIS_URL is missing", check)
        self.assertNotIn("set REDIS_URL", check)


if __name__ == "__main__":
    unittest.main()
