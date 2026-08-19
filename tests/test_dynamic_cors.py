from __future__ import annotations

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
        for value in ("*", "example.com", "https://user:pass@example.com", "https://example.com/path", "https://example.com?q=1"):
            with self.subTest(value=value), self.assertRaises(InvalidOriginError):
                normalize_origin(value)


if __name__ == "__main__":
    unittest.main()
