from __future__ import annotations

import hashlib
import hmac
import json
import time
import unittest
from urllib.parse import urlencode

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.dependencies import AdminPrincipal
from app.api.v1.admin import router
from app.core.telegram_auth import configure_telegram_admin_authorizer

BOT_TOKEN = "123456789:TEST_bot_token_for_unit_tests"


def _signed_init_data(user_id: int) -> str:
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


class _FakeRedis:
    def __init__(self, allowed: set[str]) -> None:
        self.allowed = allowed

    def smembers(self, _key: str) -> set[str]:
        return set(self.allowed)

    def sadd(self, _key: str, *values: str) -> int:
        self.allowed.update(values)
        return len(values)


class AdminApiAuthenticationTests(unittest.TestCase):
    def setUp(self) -> None:
        from app import legacy

        self.original_token = legacy.TELEGRAM_BOT_TOKEN
        legacy.TELEGRAM_BOT_TOKEN = BOT_TOKEN
        configure_telegram_admin_authorizer(
            redis_client=_FakeRedis({"42"}),
        )
        app = FastAPI()
        app.include_router(router)
        self.client = TestClient(app)

    def tearDown(self) -> None:
        from app import legacy

        legacy.TELEGRAM_BOT_TOKEN = self.original_token
        configure_telegram_admin_authorizer(redis_client=None)

    def test_missing_telegram_credentials_returns_401(self) -> None:
        response = self.client.get("/api/admin/me")
        self.assertEqual(401, response.status_code)

    def test_valid_non_admin_returns_403(self) -> None:
        response = self.client.get(
            "/api/admin/me",
            headers={"X-Telegram-Init-Data": _signed_init_data(99)},
        )
        self.assertEqual(403, response.status_code)

    def test_authorized_admin_profile_is_returned(self) -> None:
        response = self.client.get(
            "/api/admin/me",
            headers={"X-Telegram-Init-Data": _signed_init_data(42)},
        )
        self.assertEqual(200, response.status_code)
        self.assertEqual(42, response.json()["user"]["id"])
        self.assertEqual("telegram_init_data", response.json()["auth_method"])


class AdminApiSchemaTests(unittest.TestCase):
    def test_admin_principal_supports_legacy_and_telegram_auth(self) -> None:
        legacy = AdminPrincipal(admin_id=1, auth_method="cookie")
        telegram = AdminPrincipal(admin_id=2, auth_method="telegram_init_data")
        self.assertIsNone(legacy.telegram_user)
        self.assertEqual(2, telegram.admin_id)


if __name__ == "__main__":
    unittest.main()
