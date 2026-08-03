from __future__ import annotations

import unittest
import warnings
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

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
from app.services.telegram.flow import (
    callback_requires_tts_access,
    classify_callback,
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


class TelegramFlowTests(unittest.IsolatedAsyncioTestCase):
    def tearDown(self) -> None:
        with legacy._tts_request_reservations_guard:
            legacy._tts_request_reservations.clear()
        with legacy._user_locks_guard:
            legacy._user_locks.clear()

    def test_callback_classifier_covers_tts_actions(self) -> None:
        speeds = {"spd_0.5", "spd_1.0"}
        self.assertEqual(
            "speed",
            classify_callback("spd_1.0", speed_callbacks=speeds),
        )
        self.assertEqual(
            "tts_model",
            classify_callback("ttsmodel_auto", speed_callbacks=speeds),
        )
        self.assertEqual(
            "delete",
            classify_callback("audio_del:42", speed_callbacks=speeds),
        )
        self.assertIsNone(
            classify_callback("user_broken", speed_callbacks=speeds)
        )
        self.assertTrue(callback_requires_tts_access("speed", "spd_1.0"))
        self.assertFalse(
            callback_requires_tts_access("voxcpm2", "voxcpm2:refresh")
        )

    async def test_unknown_callback_is_answered_instead_of_swallowed(self) -> None:
        query = SimpleNamespace(
            data="user_broken",
            from_user=SimpleNamespace(id=42),
            message=SimpleNamespace(),
            answer=AsyncMock(),
        )
        update = SimpleNamespace(callback_query=query)

        await legacy.on_callback(update, SimpleNamespace())

        query.answer.assert_awaited_once_with(
            "This button is no longer available. Please reopen the menu.",
            show_alert=False,
        )

    async def test_tts_callback_honors_runtime_access_policy(self) -> None:
        query = SimpleNamespace(
            data="spd_1.0",
            from_user=SimpleNamespace(id=42),
            message=SimpleNamespace(),
            answer=AsyncMock(),
        )
        update = SimpleNamespace(callback_query=query)
        context = SimpleNamespace()

        with (
            patch.object(
                legacy,
                "_ensure_user_allowed",
                AsyncMock(return_value=False),
            ) as allowed,
            patch.object(legacy, "_cb_speed", AsyncMock()) as callback,
        ):
            await legacy.on_callback(update, context)

        query.answer.assert_awaited_once_with()
        allowed.assert_awaited_once()
        callback.assert_not_awaited()

    async def test_callback_failure_is_recorded_and_reported(self) -> None:
        message = SimpleNamespace(reply_text=AsyncMock())
        query = SimpleNamespace(
            data="show_speed",
            from_user=SimpleNamespace(id=42),
            message=message,
            answer=AsyncMock(),
        )
        update = SimpleNamespace(callback_query=query)

        with (
            patch.object(
                legacy,
                "_cb_show_speed",
                AsyncMock(side_effect=RuntimeError("callback failed")),
            ),
            patch.object(legacy, "_metric_inc") as metric,
            patch.object(legacy, "_record_admin_error") as record,
        ):
            await legacy.on_callback(update, SimpleNamespace())

        query.answer.assert_awaited_once_with()
        metric.assert_called_once_with("errors")
        record.assert_called_once()
        message.reply_text.assert_awaited_once()

    async def test_feature_request_wait_state_clears_when_access_denied(self) -> None:
        key = legacy.FEATURE_REQUEST_WAIT_TEXT
        context = SimpleNamespace(user_data={key: True})
        update = SimpleNamespace(
            message=SimpleNamespace(text="new feature"),
            effective_user=SimpleNamespace(id=42),
        )

        with (
            patch.object(
                legacy,
                "_ensure_user_allowed",
                AsyncMock(return_value=False),
            ),
            patch.object(
                legacy,
                "_save_user_feature_request",
                AsyncMock(),
            ) as save,
        ):
            handled = await legacy._handle_feature_request_user_text(
                update,
                context,
            )

        self.assertTrue(handled)
        self.assertNotIn(key, context.user_data)
        save.assert_not_awaited()

    async def test_tts_reservation_closes_preparation_race(self) -> None:
        user_id = 9_999_001
        self.assertTrue(legacy._reserve_tts_request(user_id))
        self.assertTrue(legacy._tts_request_reserved(user_id))
        self.assertFalse(legacy._reserve_tts_request(user_id))

        reply_target = SimpleNamespace(reply_text=AsyncMock())
        self.assertTrue(await legacy._check_cooldown(reply_target, user_id))
        reply_target.reply_text.assert_awaited_once()

        legacy._release_tts_request(user_id)
        self.assertFalse(legacy._tts_request_reserved(user_id))

    async def test_regeneration_releases_reservation_if_progress_start_fails(self) -> None:
        user_id = 9_999_002
        query = SimpleNamespace(
            message=SimpleNamespace(
                chat=SimpleNamespace(id=100),
                reply_text=AsyncMock(),
            ),
        )
        context = SimpleNamespace(bot=SimpleNamespace())

        with (
            patch.object(
                legacy,
                "_check_cooldown",
                AsyncMock(return_value=False),
            ),
            patch.object(
                legacy.TelegramProgress,
                "start",
                AsyncMock(side_effect=RuntimeError("cannot start progress")),
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "cannot start progress"):
                await legacy._regenerate_tts_voice_with_progress(
                    query=query,
                    context=context,
                    user_id=user_id,
                    original_text="hello",
                    gender="female",
                    speed=1.0,
                    tts_model="auto",
                    title="title",
                    final_text="done",
                    error_text="failed",
                    delete_source=False,
                )

        self.assertFalse(legacy._tts_request_reserved(user_id))


if __name__ == "__main__":
    unittest.main()
