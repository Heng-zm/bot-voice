from __future__ import annotations

import asyncio
import unittest

from app.core.admin_management import LastAdministratorError, SupabaseAdminManager
from app.core.telegram_auth import TelegramAdminAuthorizer
from app.services.ai.providers import NoProviderAvailable, ProviderManager
from app.services.settings.store import SettingsStore


class AdminManagementTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.store = SettingsStore()
        self.authorizer = TelegramAdminAuthorizer().configure(
            settings_store=self.store,
            fallback_admin_ids={42},
        )
        self.manager = SupabaseAdminManager(self.store, self.authorizer)

    async def test_bootstrap_admin_and_add_remove(self) -> None:
        self.assertEqual((42,), await self.manager.list_ids())
        token, _ttl = await self.manager.create_confirmation(
            action="add",
            actor_id=42,
            target_id=7,
        )
        result = await self.manager.add(
            actor_id=42,
            target_id=7,
            confirmation_token=token,
        )
        self.assertTrue(result.changed)
        self.assertEqual((7, 42), await self.manager.list_ids())

        token, _ttl = await self.manager.create_confirmation(
            action="remove",
            actor_id=42,
            target_id=7,
        )
        result = await self.manager.remove(
            actor_id=42,
            target_id=7,
            confirmation_token=token,
        )
        self.assertTrue(result.changed)
        self.assertEqual((42,), await self.manager.list_ids())

    async def test_last_admin_cannot_be_removed(self) -> None:
        with self.assertRaises(LastAdministratorError):
            await self.manager.create_confirmation(
                action="remove",
                actor_id=42,
                target_id=42,
            )


class ProviderManagerTests(unittest.IsolatedAsyncioTestCase):
    async def test_timeout_falls_back_to_next_provider(self) -> None:
        manager = ProviderManager()
        manager.register("slow", capabilities={"tts"}, priority=1, timeout_seconds=0.02)
        manager.register("fast", capabilities={"tts"}, priority=2, timeout_seconds=1)

        async def operation(provider: str) -> str:
            if provider == "slow":
                await asyncio.sleep(0.08)
            return provider

        result, provider = await manager.execute("tts", operation)
        self.assertEqual(("fast", "fast"), (result, provider))
        self.assertEqual(1, manager.snapshot()["slow"]["failures"])

    async def test_exhaustion_reports_each_provider(self) -> None:
        manager = ProviderManager()
        manager.register("one", capabilities={"ocr"})
        manager.register("two", capabilities={"ocr"})

        async def fail(provider: str) -> str:
            raise RuntimeError(provider)

        with self.assertRaises(NoProviderAvailable) as raised:
            await manager.execute("ocr", fail)
        self.assertEqual({"one", "two"}, set(raised.exception.errors))


if __name__ == "__main__":
    unittest.main()
