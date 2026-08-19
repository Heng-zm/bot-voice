from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from app.core.security import bootstrap_runtime_security, derive_runtime_secret
from app.services.settings.store import SettingsStore

BOT_TOKEN = "123456789:TEST_bot_token_for_unit_tests"


class RuntimeSecurityTests(unittest.IsolatedAsyncioTestCase):
    async def test_token_derivation_is_stable_and_domain_separated(self) -> None:
        web = derive_runtime_secret(BOT_TOKEN, "web-session")
        webhook = derive_runtime_secret(BOT_TOKEN, "telegram-webhook")
        self.assertEqual(64, len(web))
        self.assertEqual(64, len(webhook))
        self.assertNotEqual(web, webhook)
        self.assertEqual(web, derive_runtime_secret(BOT_TOKEN, "web-session"))

    async def test_bootstrap_requires_no_redis_and_persists_webhook_in_store(self) -> None:
        store = SettingsStore()
        with patch.dict(os.environ, {}, clear=False):
            for key in ("WEB_SECRET_KEY", "FLASK_SECRET_KEY", "TELEGRAM_WEBHOOK_SECRET_TOKEN", "TELEGRAM_WEBHOOK_SECRET"):
                os.environ.pop(key, None)
            first = await bootstrap_runtime_security(BOT_TOKEN, settings_store=store)
            second = await bootstrap_runtime_security(BOT_TOKEN, settings_store=store)
        self.assertEqual(first.web_secret_key, second.web_secret_key)
        self.assertEqual(first.webhook_secret_token, second.webhook_secret_token)
        self.assertTrue(first.as_dict()["redis_removed"])

    async def test_explicit_environment_secret_wins(self) -> None:
        explicit = "W" * 64
        with patch.dict(os.environ, {"WEB_SECRET_KEY": explicit}, clear=False):
            state = await bootstrap_runtime_security(BOT_TOKEN, settings_store=SettingsStore())
        self.assertEqual(explicit, state.web_secret_key)
        self.assertEqual("environment", state.web_source)


if __name__ == "__main__":
    unittest.main()
