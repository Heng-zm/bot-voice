from __future__ import annotations

import asyncio
import io
import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

from app.services.jobs.handlers import BotJobHandlers
from app.services.telegram.delivery import (
    IdempotentTelegramDelivery,
    MemoryDeliveryStore,
    RedisDeliveryStore,
    TelegramDeliveryBusy,
)


class FakeRedis:
    def __init__(self) -> None:
        self.hashes: dict[str, dict[str, str]] = {}

    def eval(self, script: str, number_of_keys: int, *values):
        key = str(values[0])
        args = [str(value) for value in values[number_of_keys:]]
        data = self.hashes.setdefault(key, {})
        if "delivery_claim_v1" in script:
            now, token, deadline, _retention = args
            if data.get("state") == "completed":
                return [2, data.get("result", "")]
            if data.get("state") == "processing" and float(
                data.get("lease_deadline", "0")
            ) > float(now):
                return [0, ""]
            data.update(
                state="processing",
                lease_token=token,
                lease_deadline=deadline,
                updated_at=now,
                attempts=str(int(data.get("attempts", "0")) + 1),
            )
            return [1, ""]
        if "delivery_complete_v1" in script:
            token, result, updated_at, _retention = args
            if data.get("state") != "processing" or data.get("lease_token") != token:
                return 0
            data.update(state="completed", result=result, updated_at=updated_at)
            data.pop("lease_token", None)
            data.pop("lease_deadline", None)
            return 1
        if "delivery_release_v1" in script:
            token, error, updated_at, _retention = args
            if data.get("state") != "processing" or data.get("lease_token") != token:
                return 0
            data.update(state="failed", last_error=error, updated_at=updated_at)
            data.pop("lease_token", None)
            data.pop("lease_deadline", None)
            return 1
        raise AssertionError("Unexpected Lua script")


class DeliveryTests(unittest.IsolatedAsyncioTestCase):
    async def test_memory_delivery_is_idempotent_without_redis(self) -> None:
        delivery = IdempotentTelegramDelivery(MemoryDeliveryStore())
        bot = SimpleNamespace(
            send_voice=AsyncMock(return_value=SimpleNamespace(message_id=91)),
        )

        first = await delivery.deliver_voice(
            bot=bot,
            idempotency_key="memory:voice",
            chat_id=42,
            voice=io.BytesIO(b"voice"),
        )
        second = await delivery.deliver_voice(
            bot=bot,
            idempotency_key="memory:voice",
            chat_id=42,
            voice=io.BytesIO(b"voice"),
        )

        self.assertEqual(first, second)
        bot.send_voice.assert_awaited_once()

    async def test_tts_handler_routes_voice_through_idempotent_delivery(self) -> None:
        bot = SimpleNamespace()

        async def generate(
            text,
            gender,
            speed,
            output_path,
            model,
            **kwargs,
        ) -> bytes:
            del text, gender, speed, model, kwargs
            await asyncio.to_thread(Path(output_path).write_bytes, b"voice")
            return b"voice"

        legacy = SimpleNamespace(
            _TELEGRAM_APP=SimpleNamespace(bot=bot),
            generate_user_voice_limited=generate,
        )
        delivery = SimpleNamespace(
            deliver_voice=AsyncMock(
                return_value={"chat_id": 42, "message_id": 99}
            )
        )
        handlers = BotJobHandlers(
            legacy,
            artifacts=SimpleNamespace(),
            delivery=delivery,
        )

        class Context:
            job = SimpleNamespace(id="tts-job", attempts=1, max_attempts=3)

            async def cancelled(self) -> bool:
                return False

            async def progress(self, percent, stage, detail) -> bool:
                del percent, stage, detail
                return True

        result = await handlers.tts(
            {"chat_id": 42, "user_id": 42, "text": "hello"},
            Context(),
        )

        self.assertEqual(99, result["message_id"])
        delivery.deliver_voice.assert_awaited_once()

    async def test_retry_edits_only_once_and_returns_stored_result(self) -> None:
        redis = FakeRedis()
        delivery = IdempotentTelegramDelivery(RedisDeliveryStore(redis))
        bot = SimpleNamespace(
            edit_message_text=AsyncMock(
                return_value=SimpleNamespace(message_id=77)
            ),
            send_message=AsyncMock(),
        )

        first = await delivery.deliver_text(
            bot=bot,
            idempotency_key="job:1:result",
            chat_id=42,
            text="done",
            progress_message_id=77,
        )
        second = await delivery.deliver_text(
            bot=bot,
            idempotency_key="job:1:result",
            chat_id=42,
            text="done",
            progress_message_id=77,
        )

        self.assertEqual(first["message_id"], second["message_id"])
        bot.edit_message_text.assert_awaited_once()
        bot.send_message.assert_not_awaited()
        stored = next(iter(redis.hashes.values()))["result"]
        self.assertEqual(77, json.loads(stored)["message_id"])

    async def test_active_delivery_lease_is_busy(self) -> None:
        redis = FakeRedis()
        store = RedisDeliveryStore(redis)
        first = await store.claim("job:busy")
        self.assertEqual("claimed", first.status)
        second = await store.claim("job:busy")
        self.assertEqual("busy", second.status)

        delivery = IdempotentTelegramDelivery(store)
        with self.assertRaises(TelegramDeliveryBusy):
            await delivery.deliver_text(
                bot=SimpleNamespace(),
                idempotency_key="job:busy",
                chat_id=42,
                text="done",
                progress_message_id=7,
            )

    async def test_voice_retry_sends_only_once(self) -> None:
        redis = FakeRedis()
        delivery = IdempotentTelegramDelivery(RedisDeliveryStore(redis))
        bot = SimpleNamespace(
            send_voice=AsyncMock(return_value=SimpleNamespace(message_id=88)),
        )

        first = await delivery.deliver_voice(
            bot=bot,
            idempotency_key="job:voice:result",
            chat_id=42,
            voice=io.BytesIO(b"voice"),
        )
        second = await delivery.deliver_voice(
            bot=bot,
            idempotency_key="job:voice:result",
            chat_id=42,
            voice=io.BytesIO(b"voice"),
        )

        self.assertEqual(first, second)
        self.assertEqual(88, first["message_id"])
        bot.send_voice.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()
