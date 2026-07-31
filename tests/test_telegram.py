from __future__ import annotations

import unittest
import warnings
from unittest.mock import patch

from fastapi import FastAPI

from app import legacy
from app.main import app
from app.services.telegram.deduplication import (
    _telegram_webhook_replay_key,
    _telegram_webhook_update_claim,
    _telegram_webhook_update_complete,
    _telegram_webhook_update_id,
    _telegram_webhook_update_release,
)


class TelegramWebhookTests(unittest.TestCase):
    def test_asgi_application_is_exposed(self) -> None:
        self.assertIsInstance(app, FastAPI)

    def test_canonical_webhook_route_is_registered(self) -> None:
        paths = {route.path for route in app.routes}
        self.assertIn("/telegram/webhook", paths)
        self.assertIn("/telegram-webhook", paths)

    def test_openapi_operation_ids_are_unique(self) -> None:
        app.openapi_schema = None
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            schema = app.openapi()

        operation_ids = [
            operation["operationId"]
            for path_item in schema["paths"].values()
            for operation in path_item.values()
            if isinstance(operation, dict) and "operationId" in operation
        ]
        duplicate_warnings = [
            warning for warning in caught if "Duplicate Operation ID" in str(warning.message)
        ]
        self.assertEqual([], duplicate_warnings)
        self.assertEqual(len(operation_ids), len(set(operation_ids)))

    def test_update_id_validation(self) -> None:
        self.assertEqual(123, _telegram_webhook_update_id(123))
        self.assertEqual(123, _telegram_webhook_update_id("123"))
        self.assertIsNone(_telegram_webhook_update_id(None))
        self.assertIsNone(_telegram_webhook_update_id("bad"))

    def test_replay_key_is_namespaced(self) -> None:
        key = _telegram_webhook_replay_key(123)
        self.assertIn("123", key)
        self.assertNotEqual("123", key)


class TelegramDeduplicationTests(unittest.IsolatedAsyncioTestCase):
    async def test_memory_claim_release_and_completion_lifecycle(self) -> None:
        update_id = 9_876_543_210
        with patch.object(legacy, "redis_client", None):
            with legacy._WEBHOOK_UPDATE_MEMORY_LOCK:
                legacy._WEBHOOK_UPDATE_MEMORY.pop(update_id, None)
            try:
                self.assertEqual("claimed", await _telegram_webhook_update_claim(update_id))
                self.assertEqual("processing", await _telegram_webhook_update_claim(update_id))

                await _telegram_webhook_update_release(update_id)
                self.assertEqual("claimed", await _telegram_webhook_update_claim(update_id))

                await _telegram_webhook_update_complete(update_id)
                self.assertEqual("completed", await _telegram_webhook_update_claim(update_id))
            finally:
                with legacy._WEBHOOK_UPDATE_MEMORY_LOCK:
                    legacy._WEBHOOK_UPDATE_MEMORY.pop(update_id, None)

    async def test_expired_worker_cannot_modify_reclaimed_redis_lease(self) -> None:
        class LeaseRedis:
            def __init__(self) -> None:
                self.values: dict[str, str] = {}

            def set(
                self,
                key: str,
                value: str,
                *,
                ex: int | None = None,
                nx: bool = False,
            ) -> bool:
                del ex
                if nx and key in self.values:
                    return False
                self.values[key] = value
                return True

            def get(self, key: str) -> str | None:
                return self.values.get(key)

            def eval(
                self,
                script: str,
                _key_count: int,
                key: str,
                expected: str,
                *args: str,
            ) -> int:
                if self.values.get(key) != expected:
                    return 0
                if "'done'" in script:
                    self.values[key] = "done"
                else:
                    self.values.pop(key, None)
                return 1

        redis = LeaseRedis()
        update_id = 9_876_543_211
        key = _telegram_webhook_replay_key(update_id)
        with patch.object(legacy, "redis_client", redis):
            first_state, first_token = await _telegram_webhook_update_claim(
                update_id,
                include_token=True,
            )
            self.assertEqual("claimed", first_state)
            self.assertIsNotNone(first_token)

            # Model expiry of the first worker's processing lease, followed by
            # a retry being claimed by a different worker.
            redis.values.pop(key)
            second_state, second_token = await _telegram_webhook_update_claim(
                update_id,
                include_token=True,
            )
            self.assertEqual("claimed", second_state)
            self.assertNotEqual(first_token, second_token)
            second_lease = redis.values[key]

            self.assertFalse(
                await _telegram_webhook_update_release(
                    update_id,
                    claim_token=first_token,
                )
            )
            self.assertEqual(second_lease, redis.values[key])
            self.assertFalse(
                await _telegram_webhook_update_complete(
                    update_id,
                    claim_token=first_token,
                )
            )
            self.assertEqual(second_lease, redis.values[key])

            self.assertTrue(
                await _telegram_webhook_update_complete(
                    update_id,
                    claim_token=second_token,
                )
            )
            self.assertEqual("done", redis.values[key])


if __name__ == "__main__":
    unittest.main()
