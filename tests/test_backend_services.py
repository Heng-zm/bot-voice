from __future__ import annotations

import asyncio
import unittest
from types import SimpleNamespace

from app.core.admin_management import (
    AdminConfirmationError,
    LastAdministratorError,
    SupabaseAdminManager,
)
from app.core.telegram_auth import TelegramAdminAuthorizer
from app.services.ai.providers import NoProviderAvailable, ProviderManager
from app.services.settings.store import SettingsStore


class SettingsStoreTests(unittest.IsolatedAsyncioTestCase):
    async def test_batch_read_uses_one_supabase_request(self) -> None:
        class Query:
            def __init__(self) -> None:
                self.execute_calls = 0
                self.keys: list[str] = []

            def select(self, _columns: str):
                return self

            def in_(self, _column: str, keys: list[str]):
                self.keys = keys
                return self

            def execute(self):
                self.execute_calls += 1
                return SimpleNamespace(
                    data=[
                        {"key": "runtime:one", "value": "1"},
                        {"key": "runtime:two", "value": "2"},
                    ]
                )

        query = Query()
        client = SimpleNamespace(table=lambda _name: query)
        store = SettingsStore(client)

        values = await store.get_many_text(
            ["runtime:one", "runtime:two", "runtime:missing"],
            "fallback",
        )

        self.assertEqual(1, query.execute_calls)
        self.assertEqual(
            ["runtime:one", "runtime:two", "runtime:missing"],
            query.keys,
        )
        self.assertEqual(
            {
                "runtime:one": "1",
                "runtime:two": "2",
                "runtime:missing": "fallback",
            },
            values,
        )


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

    async def test_removed_actor_cannot_use_an_existing_confirmation(self) -> None:
        await self.authorizer.save_ids({42, 99})
        token, _ttl = await self.manager.create_confirmation(
            action="add",
            actor_id=42,
            target_id=7,
        )
        await self.authorizer.save_ids({99})
        with self.assertRaises(AdminConfirmationError):
            await self.manager.add(
                actor_id=42,
                target_id=7,
                confirmation_token=token,
            )

    async def test_concurrent_admin_additions_do_not_lose_an_id(self) -> None:
        class SlowWriteStore(SettingsStore):
            async def set_json(self, key, value, *, updated_by=None):
                if key == "security:admin_user_ids:v2":
                    await asyncio.sleep(0.01)
                return await super().set_json(key, value, updated_by=updated_by)

        store = SlowWriteStore()
        await SettingsStore.set_json(store, "security:admin_user_ids:v2", [42])
        authorizer = TelegramAdminAuthorizer().configure(settings_store=store)
        manager = SupabaseAdminManager(store, authorizer)
        first, _ = await manager.create_confirmation(
            action="add", actor_id=42, target_id=7
        )
        second, _ = await manager.create_confirmation(
            action="add", actor_id=42, target_id=8
        )
        await asyncio.gather(
            manager.add(actor_id=42, target_id=7, confirmation_token=first),
            manager.add(actor_id=42, target_id=8, confirmation_token=second),
        )
        self.assertEqual((7, 8, 42), await manager.list_ids())


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
