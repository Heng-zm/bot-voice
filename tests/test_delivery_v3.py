from __future__ import annotations

import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock

from app.services.telegram.delivery import (
    IdempotentTelegramDelivery,
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


if __name__ == "__main__":
    unittest.main()
