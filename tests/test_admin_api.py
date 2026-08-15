from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import os
import time
import unittest
from unittest.mock import AsyncMock, patch
from urllib.parse import urlencode

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.dependencies import AdminPrincipal
from app.api.v1.admin import _redis_health, router
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


class _AsyncHealthRedis:
    async def ping(self) -> bool:
        return True


class AdminHealthTests(unittest.TestCase):
    def test_disabled_redis_is_healthy_without_a_client(self) -> None:
        result = asyncio.run(_redis_health(None, enabled=False))

        self.assertEqual(
            {"ok": True, "latency_ms": None, "status": "disabled"},
            result,
        )

    def test_async_redis_client_health_is_awaited(self) -> None:
        result = asyncio.run(_redis_health(_AsyncHealthRedis()))

        self.assertTrue(result["ok"])
        self.assertEqual("healthy", result["status"])
        self.assertIsInstance(result["latency_ms"], float)


class AdminApiAuthenticationTests(unittest.TestCase):
    def setUp(self) -> None:
        from app import legacy

        # The application intentionally supports a separate admin bot token.
        # Isolate these primary-token tests from developer/deployment dotenv
        # files that may define that optional override.
        self.admin_token_env = patch.dict(
            os.environ,
            {"TELEGRAM_ADMIN_BOT_TOKEN": ""},
            clear=False,
        )
        self.admin_token_env.start()
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
        self.admin_token_env.stop()

    def test_missing_telegram_credentials_returns_401(self) -> None:
        response = self.client.get("/api/admin/me")
        self.assertEqual(401, response.status_code)

    def test_malformed_bearer_identity_returns_401_instead_of_500(self) -> None:
        from app import legacy

        with (
            patch.object(legacy, "_web_admin_enabled", return_value=True),
            patch.object(
                legacy,
                "_admin_verify_api_token",
                return_value="not-an-integer",
            ),
        ):
            response = self.client.get(
                "/api/admin/me",
                headers={"Authorization": "Bearer malformed-identity"},
            )

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

    def test_admin_stats_include_build_metadata(self) -> None:
        from app import legacy

        with (
            patch.dict(
                os.environ,
                {
                    "BOT_BUILD_VERSION": "2026.08.12",
                    "RELEASE_SHA": "abcdef1234567890fedcba",
                },
                clear=False,
            ),
            patch.object(legacy, "_web_counts", return_value={"users": 3, "blocked": 1}),
            patch.object(
                legacy,
                "get_bot_settings_async",
                AsyncMock(return_value=({}, {"db_ok": True, "memory": False})),
            ),
            patch.object(legacy, "_run_state_bot_mode", return_value="WEBHOOK"),
            patch.object(legacy, "_format_uptime", return_value="1m 0s"),
            patch.object(legacy, "_TELEGRAM_POLLING_ACTIVE", False),
            patch.object(legacy, "_TELEGRAM_APP", object()),
        ):
            response = self.client.get(
                "/api/admin/stats",
                headers={"X-Telegram-Init-Data": _signed_init_data(42)},
            )

        self.assertEqual(200, response.status_code)
        payload = response.json()
        self.assertEqual("2026.08.12", payload["build"]["version"])
        self.assertEqual("abcdef123456", payload["build"]["commit_short"])
        self.assertEqual("WEBHOOK", payload["bot"]["mode"])


class AdminApiSchemaTests(unittest.TestCase):
    def test_admin_principal_supports_legacy_and_telegram_auth(self) -> None:
        legacy = AdminPrincipal(admin_id=1, auth_method="cookie")
        telegram = AdminPrincipal(admin_id=2, auth_method="telegram_init_data")
        self.assertIsNone(legacy.telegram_user)
        self.assertEqual(2, telegram.admin_id)


if __name__ == "__main__":
    unittest.main()
