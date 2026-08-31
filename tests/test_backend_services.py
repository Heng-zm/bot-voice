from __future__ import annotations

import asyncio
import unittest
from types import SimpleNamespace

from app.services.ai.providers import NoProviderAvailable, ProviderManager
from app.services.settings.store import SettingsStore


class SettingsStoreTests(unittest.IsolatedAsyncioTestCase):
    async def test_batch_read_uses_one_supabase_request(self) -> None:
        class Query:
            def __init__(self) -> None:
                self.execute_calls = 0

            def select(self, _columns: str):
                return self

            def in_(self, _column: str, _keys: list[str]):
                return self

            def execute(self):
                self.execute_calls += 1
                return SimpleNamespace(data=[{"key": "runtime:one", "value": "1"}])

        query = Query()
        store = SettingsStore(SimpleNamespace(table=lambda _name: query))

        values = await store.get_many_text(["runtime:one", "runtime:missing"], "fallback")

        self.assertEqual(1, query.execute_calls)
        self.assertEqual({"runtime:one": "1", "runtime:missing": "fallback"}, values)


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
