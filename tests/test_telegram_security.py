from __future__ import annotations

import unittest
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import Mock, patch

from app.services.telegram.security import is_user_blocked


class TelegramSecurityTests(unittest.IsolatedAsyncioTestCase):
    async def test_cached_block_state_avoids_executor_lookup(self) -> None:
        database_lookup = Mock(return_value=True)
        runtime = SimpleNamespace(
            _blocked_cache_get=Mock(return_value=False),
            _DB_EXECUTOR=None,
            db_user_is_blocked=database_lookup,
        )

        with patch(
            "app.services.telegram.security.legacy_module",
            return_value=runtime,
        ):
            self.assertFalse(await is_user_blocked(42))

        database_lookup.assert_not_called()

    async def test_cache_miss_uses_bounded_database_executor(self) -> None:
        database_lookup = Mock(return_value=True)
        with ThreadPoolExecutor(max_workers=1) as executor:
            runtime = SimpleNamespace(
                _blocked_cache_get=Mock(return_value=None),
                _DB_EXECUTOR=executor,
                db_user_is_blocked=database_lookup,
            )
            with patch(
                "app.services.telegram.security.legacy_module",
                return_value=runtime,
            ):
                self.assertTrue(await is_user_blocked(99))

        database_lookup.assert_called_once_with(99)


if __name__ == "__main__":
    unittest.main()
