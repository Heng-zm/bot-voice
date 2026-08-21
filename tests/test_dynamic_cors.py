from __future__ import annotations

import asyncio
import unittest

from app.core.cors import DynamicCorsStore, InvalidOriginError, normalize_origin
from app.services.settings.store import SettingsStore


class DynamicCorsTests(unittest.IsolatedAsyncioTestCase):
    async def test_exact_origins_persist_without_redis(self) -> None:
        store = SettingsStore()
        cors = DynamicCorsStore(store)
        snapshot, changed = await cors.add("https://Admin.Example.com/", admin_id=42)
        self.assertTrue(changed)
        self.assertEqual(("https://admin.example.com",), snapshot.origins)
        self.assertTrue(await cors.is_allowed("https://admin.example.com"))
        self.assertFalse(await cors.is_allowed("https://evil.example.com"))

    def test_invalid_origins_are_rejected(self) -> None:
        for value in (
            "*",
            "example.com",
            "https://user:pass@example.com",
            "https://example.com/path",
            "https://example.com?q=1",
            "https://example.com:bad",
            "https://example.com:99999",
            "https://example.com\\evil",
        ):
            with self.subTest(value=value), self.assertRaises(InvalidOriginError):
                normalize_origin(value)

    def test_origins_are_canonicalized_for_browser_comparison(self) -> None:
        self.assertEqual("https://example.com", normalize_origin("https://example.com:443"))
        self.assertEqual("http://example.com", normalize_origin("http://example.com:80"))
        self.assertEqual("https://[2001:db8::1]", normalize_origin("https://[2001:DB8::1]"))

    async def test_concurrent_additions_do_not_lose_an_origin(self) -> None:
        class SlowReadStore(SettingsStore):
            async def get_json(self, key, default):
                await asyncio.sleep(0.01)
                return await super().get_json(key, default)

        cors = DynamicCorsStore(SlowReadStore())
        await asyncio.gather(
            cors.add("https://one.example", admin_id=42),
            cors.add("https://two.example", admin_id=42),
        )
        self.assertEqual(
            ("https://one.example", "https://two.example"),
            (await cors.load(force=True)).origins,
        )


if __name__ == "__main__":
    unittest.main()
