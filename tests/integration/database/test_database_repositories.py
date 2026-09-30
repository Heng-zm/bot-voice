"""Integration tests for database repositories."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock

from app.database.client import SupabaseClient
from app.database.repositories.settings import SettingsStore
from app.database.repositories.users import UserRepository


class DatabaseRepositoriesIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_settings_store_in_memory_fallback(self):
        client = SupabaseClient(url="", key="")
        store = SettingsStore(client)
        await store.set("test_key_abc", "value_123")
        val = await store.get("test_key_abc")
        self.assertEqual(val, "value_123")

    async def test_user_repository_fallback(self):
        client = SupabaseClient(url="", key="")
        repo = UserRepository(client)
        prefs = await repo.get_user_prefs(123456)
        self.assertIsInstance(prefs, dict)


if __name__ == "__main__":
    unittest.main()
