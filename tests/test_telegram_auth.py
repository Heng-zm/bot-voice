from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import time
import unittest
from urllib.parse import urlencode

from app.core.telegram_auth import TelegramAdminAuthorizer, TelegramInitDataError, validate_telegram_init_data
from app.services.settings.store import SettingsStore

BOT_TOKEN = "123456789:TEST_bot_token_for_unit_tests"


def signed_init_data(user_id: int, *, auth_date: int, extra_fields: dict[str, str] | None = None) -> str:
    fields = {
        "auth_date": str(auth_date),
        "query_id": "AAH-test-query",
        "user": json.dumps({"id": user_id, "first_name": "Admin", "username": "admin_test"}, separators=(",", ":")),
    }
    fields.update(extra_fields or {})
    check = "\n".join(f"{key}={value}" for key, value in sorted(fields.items()))
    secret = hmac.new(b"WebAppData", BOT_TOKEN.encode(), hashlib.sha256).digest()
    fields["hash"] = hmac.new(secret, check.encode(), hashlib.sha256).hexdigest()
    return urlencode(fields)


class TelegramInitDataTests(unittest.TestCase):
    def test_valid_init_data_returns_trusted_user(self) -> None:
        now = int(time.time())
        result = validate_telegram_init_data(signed_init_data(42, auth_date=now), BOT_TOKEN, now=now)
        self.assertEqual(42, result.user.id)

    def test_tamper_duplicate_expiry_and_future_are_rejected(self) -> None:
        now = int(time.time())
        tampered = signed_init_data(42, auth_date=now).replace("admin_test", "attacker")
        with self.assertRaises(TelegramInitDataError):
            validate_telegram_init_data(tampered, BOT_TOKEN, now=now)
        with self.assertRaisesRegex(TelegramInitDataError, "duplicate"):
            validate_telegram_init_data(signed_init_data(42, auth_date=now) + "&auth_date=1", BOT_TOKEN, now=now)
        with self.assertRaisesRegex(TelegramInitDataError, "expired"):
            validate_telegram_init_data(signed_init_data(42, auth_date=now - 3601), BOT_TOKEN, now=now)
        with self.assertRaisesRegex(TelegramInitDataError, "future"):
            validate_telegram_init_data(signed_init_data(42, auth_date=now + 31), BOT_TOKEN, now=now)


class TelegramAdminAuthorizerTests(unittest.IsolatedAsyncioTestCase):
    async def test_settings_store_allowlist_authorizes_only_members(self) -> None:
        store = SettingsStore()
        authorizer = TelegramAdminAuthorizer().configure(settings_store=store, fallback_admin_ids={42})
        now = int(time.time())
        self.assertEqual(42, (await authorizer.authorize(signed_init_data(42, auth_date=now), BOT_TOKEN)).user.id)
        with self.assertRaises(PermissionError):
            await authorizer.authorize(signed_init_data(99, auth_date=now), BOT_TOKEN)

    async def test_concurrent_cache_misses_share_one_store_read(self) -> None:
        class CountingStore(SettingsStore):
            def __init__(self) -> None:
                super().__init__()
                self.reads = 0
            async def get_json(self, key, default):
                self.reads += 1
                await asyncio.sleep(0.01)
                return await super().get_json(key, default)
        store = CountingStore()
        await store.set_json("security:admin_user_ids:v2", [42])
        authorizer = TelegramAdminAuthorizer().configure(settings_store=store)
        results = await asyncio.gather(*(authorizer.load_ids() for _ in range(12)))
        self.assertEqual(1, store.reads)
        self.assertTrue(all(result == frozenset({42}) for result in results))


if __name__ == "__main__":
    unittest.main()
