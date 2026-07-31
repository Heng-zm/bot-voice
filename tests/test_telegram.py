from __future__ import annotations

import unittest
import warnings
from unittest.mock import patch

from fastapi import FastAPI

from app import legacy
from app.main import app
from app.services.telegram.deduplication import (
    _telegram_webhook_replay_key,
    _telegram_webhook_update_id,
    _telegram_webhook_update_claim,
    _telegram_webhook_update_complete,
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


if __name__ == "__main__":
    unittest.main()
