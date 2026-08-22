from __future__ import annotations

import asyncio
import unittest

from app.core.telegram_auth import TelegramAdminAuthorizer
from app.services.settings.store import SettingsStore


class TelegramAdminAuthorizerTests(unittest.IsolatedAsyncioTestCase):
    async def test_fallback_ids_are_loaded_and_available_synchronously(self) -> None:
        store = SettingsStore()
        authorizer = TelegramAdminAuthorizer().configure(
            settings_store=store,
            fallback_admin_ids={42},
        )

        self.assertTrue(authorizer.is_admin_sync(42))
        self.assertEqual(frozenset({42}), await authorizer.load_ids())
        self.assertTrue(authorizer.is_admin_sync(42))
        self.assertFalse(authorizer.is_admin_sync(99))

    async def test_saved_ids_replace_the_cached_snapshot(self) -> None:
        authorizer = TelegramAdminAuthorizer().configure(settings_store=SettingsStore())

        await authorizer.save_ids({7, 8})

        self.assertTrue(authorizer.is_admin_sync(7))
        self.assertFalse(authorizer.is_admin_sync(42))

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
