from __future__ import annotations

import unittest
import warnings
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

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

    def test_broadcast_markdown_link_and_preview_directives(self) -> None:
        stored = legacy._broadcast_apply_option_directives(
            "[Open site](https://example.com)",
            "markdown",
            False,
        )
        text, mode, link_preview = legacy._broadcast_prepare_text(
            stored,
            "auto",
            max_chars=legacy.TELE_MSG_LIMIT,
        )

        self.assertEqual(
            "::md\n::nopreview\n[Open site](https://example.com)",
            stored,
        )
        self.assertEqual("[Open site](https://example.com)", text)
        self.assertEqual("markdown", mode)
        self.assertFalse(link_preview)

    async def test_broadcast_markdown_link_enables_url_preview(self) -> None:
        bot = SimpleNamespace(
            send_message=AsyncMock(return_value=SimpleNamespace(message_id=1)),
        )

        await legacy._send_telegram_broadcast_message(
            bot,
            chat_id=42,
            text="[Open site](https://example.com)",
            parse_mode="markdown",
            link_preview=True,
        )

        bot.send_message.assert_awaited_once_with(
            chat_id=42,
            text="[Open site](https://example.com)",
            disable_web_page_preview=False,
            parse_mode="Markdown",
        )

    def test_web_broadcast_markdown_link_enables_url_preview(self) -> None:
        response = SimpleNamespace(status_code=200, json=lambda: {"ok": True})
        client = SimpleNamespace(post=Mock(return_value=response))

        with patch.object(legacy, "TELEGRAM_BOT_TOKEN", "test-token"):
            ok, result = legacy._web_send_telegram_message(
                42,
                "[Open site](https://example.com)",
                client=client,
                parse_mode="markdown",
                link_preview=True,
            )

        self.assertTrue(ok, result)
        payload = client.post.call_args.kwargs["json"]
        self.assertEqual("Markdown", payload["parse_mode"])
        self.assertFalse(payload["disable_web_page_preview"])

    def test_daily_schedule_time_uses_next_phnom_penh_occurrence(self) -> None:
        before_eight = datetime(2026, 8, 3, 0, 30, tzinfo=UTC)
        after_eight = datetime(2026, 8, 3, 2, 0, tzinfo=UTC)

        first_run, recurrence = legacy._parse_schedule_request(
            "daily 08:00",
            before_eight,
        )
        next_day, next_recurrence = legacy._parse_schedule_request(
            "every morning 8 AM",
            after_eight,
        )

        self.assertEqual("daily", recurrence)
        self.assertEqual("daily", next_recurrence)
        self.assertEqual(datetime(2026, 8, 3, 1, 0, tzinfo=UTC), first_run)
        self.assertEqual(datetime(2026, 8, 4, 1, 0, tzinfo=UTC), next_day)

    def test_daily_schedule_marker_round_trip_preserves_broadcast_format(self) -> None:
        stored = legacy._sched_apply_recurrence_directive(
            "::md\n::nopreview\n[Morning news](https://example.com)",
            "daily",
        )
        broadcast_content, recurrence = legacy._sched_strip_recurrence_directive(stored)
        text, mode, link_preview = legacy._broadcast_prepare_text(
            broadcast_content,
            "auto",
            max_chars=legacy.TELE_MSG_LIMIT,
        )

        self.assertTrue(stored.startswith("::schedule_daily\n"))
        self.assertEqual("daily", recurrence)
        self.assertEqual("[Morning news](https://example.com)", text)
        self.assertEqual("markdown", mode)
        self.assertFalse(link_preview)

    async def test_daily_schedule_reschedules_instead_of_finishing(self) -> None:
        next_run = datetime(2026, 8, 4, 1, 0, tzinfo=UTC)
        bot = SimpleNamespace(send_message=AsyncMock(return_value=SimpleNamespace(message_id=1)))
        row = {
            "id": 77,
            "admin_id": 42,
            "photo_file_id": None,
            "caption": None,
            "plain_text": "::schedule_daily\n::md\nGood *morning*",
            "broadcast_at": "2026-08-03T01:00:00+00:00",
        }

        with (
            patch.object(legacy, "_run_broadcast_to_all", AsyncMock(return_value=(10, 1, 2))) as run,
            patch.object(legacy, "db_sched_reschedule_daily", return_value=next_run) as reschedule,
            patch.object(legacy, "db_sched_set_status") as set_status,
        ):
            await legacy._fire_scheduled_broadcast(bot, row, already_claimed=True)

        pending = run.await_args.args[2]
        self.assertEqual("::md\nGood *morning*", pending["text"])
        reschedule.assert_called_once()
        set_status.assert_not_called()
        bot.send_message.assert_awaited_once()

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

    async def test_disabled_feature_also_blocks_admin_usage(self) -> None:
        message = SimpleNamespace(reply_text=AsyncMock())
        update = SimpleNamespace(
            effective_user=SimpleNamespace(id=42),
            effective_message=message,
        )
        settings = dict(legacy.BOT_SETTING_DEFAULTS)
        settings["tts_enabled"] = "0"

        with (
            patch.object(legacy, "_is_admin", return_value=True),
            patch.object(
                legacy,
                "get_bot_settings_async",
                AsyncMock(return_value=(settings, {})),
            ),
        ):
            allowed = await legacy._ensure_user_allowed(
                update,
                SimpleNamespace(),
                "tts_enabled",
                "Text to voice",
            )

        self.assertFalse(allowed)
        message.reply_text.assert_awaited_once()

    async def test_disabling_tts_clears_pending_voxcpm2_input(self) -> None:
        context = SimpleNamespace(
            user_data={"voxcpm2_state": legacy.VOXCPM2_WAIT_CONTROL},
        )
        with patch.object(
            legacy,
            "_ensure_user_allowed",
            AsyncMock(return_value=False),
        ):
            allowed = await legacy._ensure_voxcpm2_allowed(
                SimpleNamespace(),
                context,
            )

        self.assertFalse(allowed)
        self.assertNotIn("voxcpm2_state", context.user_data)

    async def test_setting_toggle_updates_runtime_cache_immediately(self) -> None:
        old_memory = dict(legacy._bot_settings_memory)
        old_cache = {
            "data": dict(legacy._bot_settings_cache["data"]),
            "status": dict(legacy._bot_settings_cache["status"]),
            "ts": legacy._bot_settings_cache["ts"],
        }
        try:
            legacy._bot_settings_memory["tts_enabled"] = "1"
            legacy._bot_settings_cache.update(
                data={**legacy.BOT_SETTING_DEFAULTS, "tts_enabled": "1"},
                status={"memory": True},
                ts=9.0,
            )
            with (
                patch.object(legacy, "supabase", None),
                patch.object(legacy, "_submit_db"),
                patch.object(legacy.time, "monotonic", return_value=10.0),
            ):
                ok, _info = legacy.db_bot_setting_set("tts_enabled", False, 42)
                settings, _status = await legacy.get_bot_settings_async()

            self.assertTrue(ok)
            self.assertFalse(legacy._setting_bool_from(settings, "tts_enabled", True))
        finally:
            legacy._bot_settings_memory.clear()
            legacy._bot_settings_memory.update(old_memory)
            legacy._bot_settings_cache.clear()
            legacy._bot_settings_cache.update(old_cache)

    async def test_disabled_audio_features_do_not_bypass_for_admin(self) -> None:
        message = SimpleNamespace(
            chat_id=100,
            document=None,
            audio=SimpleNamespace(
                file_name="sample.mp3",
                mime_type="audio/mpeg",
                file_id="file-id",
                file_unique_id="unique-id",
                file_size=100,
                duration=1.0,
            ),
            reply_text=AsyncMock(),
        )
        update = SimpleNamespace(
            message=message,
            effective_user=SimpleNamespace(id=42, username="admin", first_name="Admin"),
        )
        context = SimpleNamespace(user_data={}, bot=SimpleNamespace())
        settings = dict(legacy.BOT_SETTING_DEFAULTS)
        settings["audio_to_voice_enabled"] = "0"
        settings["audio_transcribe_enabled"] = "0"

        with (
            patch.object(legacy, "_is_admin", return_value=True),
            patch.object(legacy, "_get_admin_for_user", return_value=None),
            patch.object(legacy, "_ensure_user_allowed", AsyncMock(return_value=True)),
            patch.object(
                legacy,
                "get_bot_settings_async",
                AsyncMock(return_value=(settings, {})),
            ),
            patch.object(legacy, "_check_cooldown", AsyncMock(return_value=False)),
            patch.object(legacy.TelegramProgress, "start", AsyncMock()) as progress,
        ):
            await legacy.on_audio_file(update, context)

        progress.assert_not_awaited()
        message.reply_text.assert_awaited_once()

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

    async def test_gender_and_speed_updates_remain_in_local_preferences(self) -> None:
        user_id = 9_999_003
        with (
            patch.object(legacy, "supabase", None),
            patch.object(legacy, "redis_client", None),
        ):
            legacy._cache_prefs_sync(
                user_id,
                {"gender": "female", "speed": 1.0, "tts_model": "auto"},
            )
            legacy.update_user_gender(user_id, "male")
            self.assertEqual("male", (await legacy.get_user_prefs_async(user_id))["gender"])

            legacy.update_user_speed(user_id, 1.25)
            self.assertEqual(1.25, (await legacy.get_user_prefs_async(user_id))["speed"])

    async def test_preference_update_survives_redis_failure(self) -> None:
        user_id = 9_999_004
        with legacy._prefs_cache_thread_lock:
            legacy._prefs_cache.pop(user_id, None)

        with (
            patch.object(legacy, "supabase", None),
            patch.object(legacy, "redis_client", object()),
            patch.object(
                legacy,
                "_redis_get_json_sync",
                side_effect=RuntimeError("redis read failed"),
            ),
            patch.object(
                legacy,
                "_redis_set_json_sync",
                side_effect=RuntimeError("redis write failed"),
            ),
        ):
            legacy.update_user_gender(user_id, "male")
            self.assertEqual("male", legacy._get_cached_prefs_sync(user_id)["gender"])

    async def test_rejected_speed_callback_does_not_change_preference(self) -> None:
        data, (_label, new_speed) = next(iter(legacy.SPEED_OPTIONS.items()))
        query = SimpleNamespace(
            message=SimpleNamespace(
                chat=SimpleNamespace(id=100),
                reply_text=AsyncMock(),
            ),
        )
        prefs = {"gender": "female", "speed": 1.0, "tts_model": "auto"}

        with (
            patch.object(
                legacy,
                "get_callback_original_text",
                AsyncMock(return_value="hello"),
            ),
            patch.object(
                legacy,
                "get_user_prefs_async",
                AsyncMock(return_value=prefs),
            ),
            patch.object(
                legacy,
                "_check_cooldown",
                AsyncMock(return_value=True),
            ),
            patch.object(legacy, "update_user_speed") as update_speed,
        ):
            await legacy._cb_speed(query, 42, SimpleNamespace(), data)

        update_speed.assert_not_called()
        self.assertNotEqual(prefs["speed"], new_speed)

    async def test_expired_broadcast_callback_reports_itself(self) -> None:
        message = SimpleNamespace(reply_text=AsyncMock())
        query = SimpleNamespace(
            data="bc_removed_feature",
            from_user=SimpleNamespace(id=42),
            message=message,
            answer=AsyncMock(),
        )
        update = SimpleNamespace(callback_query=query)

        with patch.object(legacy, "_is_admin", return_value=True):
            await legacy.broadcast_callback(update, SimpleNamespace())

        query.answer.assert_awaited_once_with()
        message.reply_text.assert_awaited_once()

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
            ),self.assertRaisesRegex(RuntimeError, "cannot start progress")
        ):
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
