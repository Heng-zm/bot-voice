from __future__ import annotations

import hashlib
import hmac
import json
import os
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from urllib.parse import urlencode

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.v1.admin import router
from app.core.telegram_auth import configure_telegram_admin_authorizer
from app.services.settings.store import SettingsStore

BOT_TOKEN = "123456789:TEST_bot_token_for_unit_tests"


def signed_init_data(user_id: int) -> str:
    fields = {
        "auth_date": str(int(time.time())),
        "query_id": "query",
        "user": json.dumps(
            {"id": user_id, "first_name": "TMA Admin"},
            separators=(",", ":"),
        ),
    }
    check = "\n".join(f"{key}={value}" for key, value in sorted(fields.items()))
    secret = hmac.new(b"WebAppData", BOT_TOKEN.encode(), hashlib.sha256).digest()
    fields["hash"] = hmac.new(secret, check.encode(), hashlib.sha256).hexdigest()
    return urlencode(fields)


class AdminApiAuthenticationTests(unittest.TestCase):
    def setUp(self) -> None:
        fake_legacy = SimpleNamespace(
            TELEGRAM_ADMIN_BOT_TOKEN="",
            TELEGRAM_BOT_TOKEN=BOT_TOKEN,
            SETTINGS=SimpleNamespace(TELEGRAM_BOT_TOKEN=BOT_TOKEN),
            _web_admin_enabled=lambda: True,
            _web_valid_admin_id=lambda value: value == 42,
            _admin_verify_api_token=lambda token: (
                42 if token == "legacy-api-token" else None
            ),
        )
        self.legacy_patch = patch(
            "app.api.dependencies.legacy_module",
            return_value=fake_legacy,
        )
        self.legacy_patch.start()
        self.env_patch = patch.dict(
            os.environ,
            {"TELEGRAM_ADMIN_BOT_TOKEN": BOT_TOKEN},
            clear=False,
        )
        self.env_patch.start()
        configure_telegram_admin_authorizer(
            settings_store=SettingsStore(),
            fallback_admin_ids={42},
        )
        app = FastAPI()
        app.include_router(router)
        self.client = TestClient(app)

    def tearDown(self) -> None:
        self.env_patch.stop()
        self.legacy_patch.stop()

    def test_missing_telegram_credentials_returns_401(self) -> None:
        self.assertEqual(401, self.client.get("/api/admin/me").status_code)

    def test_valid_non_admin_returns_403(self) -> None:
        response = self.client.get(
            "/api/admin/me",
            headers={"X-Telegram-Init-Data": signed_init_data(99)},
        )
        self.assertEqual(403, response.status_code)

    def test_authorized_admin_supports_header_and_bearer_fallback(self) -> None:
        raw = signed_init_data(42)
        header = self.client.get(
            "/api/admin/me",
            headers={"X-Telegram-Init-Data": raw},
        )
        bearer = self.client.get(
            "/api/admin/me",
            headers={"Authorization": f"Bearer {raw}"},
        )
        self.assertEqual(200, header.status_code)
        self.assertEqual(200, bearer.status_code)
        self.assertEqual(42, bearer.json()["user"]["id"])

    def test_opaque_admin_api_bearer_token_remains_reachable(self) -> None:
        response = self.client.get(
            "/api/admin/me",
            headers={"Authorization": "Bearer legacy-api-token"},
        )
        self.assertEqual(200, response.status_code)
        self.assertEqual(42, response.json()["user"]["id"])


if __name__ == "__main__":
    unittest.main()
