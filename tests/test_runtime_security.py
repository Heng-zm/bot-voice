from __future__ import annotations

import re
import unittest
from pathlib import Path

from app.core.security import (
    SECRET_NAMES,
    RuntimeSecretError,
    RuntimeSecretManager,
)
from app.legacy import AppSettings


class FakeRedis:
    def __init__(self) -> None:
        self.values: dict[str, str] = {}
        self.set_calls: list[dict] = []

    def get(self, key: str):
        return self.values.get(key)

    def set(
        self,
        key: str,
        value: str,
        *,
        nx: bool = False,
        ex=None,
        px=None,
        **kwargs,
    ):
        del kwargs
        self.set_calls.append(
            {"key": key, "value": value, "nx": nx, "ex": ex, "px": px}
        )
        if nx and key in self.values:
            return False
        self.values[key] = value
        return True

    def eval(self, script: str, number_of_keys: int, key: str, token: str):
        del script, number_of_keys
        if self.values.get(key) == token:
            self.values.pop(key, None)
            return 1
        return 0

    def pipeline(self, transaction: bool = True):
        redis = self

        class Pipeline:
            def __init__(self) -> None:
                self.operations: list[tuple[tuple, dict]] = []

            def set(self, *args, **kwargs):
                self.operations.append((args, kwargs))
                return self

            def execute(self):
                return [redis.set(*args, **kwargs) for args, kwargs in self.operations]

        self.last_pipeline_transaction = transaction
        return Pipeline()


class RuntimeSecretManagerTests(unittest.TestCase):
    def test_first_boot_generates_and_persists_all_secrets_without_expiry(self) -> None:
        redis = FakeRedis()
        manager = RuntimeSecretManager(redis_prefix="tests").configure(
            redis_client=redis
        )

        state = manager.bootstrap_sync()

        self.assertEqual(set(state.records), set(SECRET_NAMES))
        self.assertEqual(state.newly_created, frozenset(SECRET_NAMES))
        for record in state.records.values():
            self.assertRegex(record.value, re.compile(r"^[A-Za-z0-9_-]{64}$"))
            self.assertEqual(redis.values[record.redis_key], record.value)
            secret_set = next(
                call
                for call in redis.set_calls
                if call["key"] == record.redis_key
            )
            self.assertIsNone(secret_set["ex"])
            self.assertIsNone(secret_set["px"])

    def test_second_server_loads_the_same_winning_values(self) -> None:
        redis = FakeRedis()
        first = RuntimeSecretManager(redis_prefix="tests").configure(
            redis_client=redis
        )
        second = RuntimeSecretManager(redis_prefix="tests").configure(
            redis_client=redis
        )

        first_state = first.bootstrap_sync()
        second_state = second.bootstrap_sync()

        self.assertEqual(second_state.newly_created, frozenset())
        self.assertEqual(
            {name: record.value for name, record in first_state.records.items()},
            {name: record.value for name, record in second_state.records.items()},
        )

    def test_invalid_persisted_secret_fails_closed(self) -> None:
        redis = FakeRedis()
        redis.values["tests:web_secret_key:v1"] = "short"
        manager = RuntimeSecretManager(redis_prefix="tests").configure(
            redis_client=redis
        )

        with self.assertRaisesRegex(RuntimeSecretError, "invalid WEB_SECRET_KEY"):
            manager.bootstrap_sync()

    def test_strict_boot_requires_redis(self) -> None:
        manager = RuntimeSecretManager(redis_prefix="tests")

        with self.assertRaisesRegex(RuntimeSecretError, "REDIS_URL"):
            manager.bootstrap_sync(strict=True)

    def test_registered_manual_rotation_updates_canonical_secret_and_marker(
        self,
    ) -> None:
        redis = FakeRedis()
        manager = RuntimeSecretManager(redis_prefix="tests").configure(
            redis_client=redis
        )
        manager.bootstrap_sync()
        rotated = "A" * 64

        record = manager.persist_registered_webhook_secret_sync(rotated)

        self.assertEqual(record.value, rotated)
        self.assertEqual(
            redis.values["tests:telegram_webhook_secret_token:v1"],
            rotated,
        )
        self.assertRegex(
            redis.values[
                "tests:telegram_webhook_secret_token:registered:v1"
            ],
            re.compile(r"^[a-f0-9]{64}$"),
        )


class MinimalSettingsTests(unittest.TestCase):
    def test_runtime_values_are_not_app_settings_fields(self) -> None:
        removed = {
            "TELEGRAM_WEBHOOK_SECRET_TOKEN",
            "WEB_SECRET_KEY",
            "FLASK_SECRET_KEY",
            "FRONTEND_ALLOWED_ORIGINS",
        }
        self.assertTrue(removed.isdisjoint(AppSettings.model_fields))

    def test_env_template_contains_only_core_connections(self) -> None:
        expected = {
            "ADMIN_IDS",
            "REDIS_ENABLED",
            "REDIS_URL",
            "SUPABASE_URL",
            "SUPABASE_SERVICE_ROLE_KEY",
            "TELEGRAM_BOT_TOKEN",
            "GEMINI_API_KEY",
        }
        env_path = Path(__file__).resolve().parents[1] / ".env.example"
        configured = {
            line.split("=", 1)[0]
            for line in env_path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        }
        self.assertEqual(configured, expected)


class RuntimeWebhookRegistrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_webhook_registration_runs_once_and_sets_marker_after_success(
        self,
    ) -> None:
        redis = FakeRedis()
        manager = RuntimeSecretManager(redis_prefix="tests").configure(
            redis_client=redis
        )
        state = manager.bootstrap_sync()
        received: list[str] = []

        async def register(secret: str) -> None:
            received.append(secret)

        first = await manager.ensure_webhook_registered(register)
        second = await manager.ensure_webhook_registered(register)

        self.assertTrue(first)
        self.assertFalse(second)
        self.assertEqual(
            received,
            [state.value("TELEGRAM_WEBHOOK_SECRET_TOKEN")],
        )
        marker_key = "tests:telegram_webhook_secret_token:registered:v1"
        self.assertRegex(redis.values[marker_key], re.compile(r"^[a-f0-9]{64}$"))
        marker_set = [
            call for call in redis.set_calls if call["key"] == marker_key
        ][-1]
        self.assertIsNone(marker_set["ex"])
        self.assertIsNone(marker_set["px"])

    async def test_failed_registration_is_retried_on_next_start(self) -> None:
        redis = FakeRedis()
        manager = RuntimeSecretManager(redis_prefix="tests").configure(
            redis_client=redis
        )
        manager.bootstrap_sync()

        async def fail(_secret: str) -> None:
            raise RuntimeError("telegram unavailable")

        with self.assertRaisesRegex(RuntimeError, "telegram unavailable"):
            await manager.ensure_webhook_registered(fail)

        calls = 0

        async def succeed(_secret: str) -> None:
            nonlocal calls
            calls += 1

        self.assertTrue(await manager.ensure_webhook_registered(succeed))
        self.assertEqual(calls, 1)


if __name__ == "__main__":
    unittest.main()
