from __future__ import annotations

import hashlib
import hmac
import json
import time
import unittest
from urllib.parse import urlencode

from app.core.telegram_auth import (
    TelegramAdminAuthorizer,
    TelegramInitDataError,
    validate_telegram_init_data,
)

BOT_TOKEN = "123456789:TEST_bot_token_for_unit_tests"
PUBLISHED_VECTOR_TOKEN = "7342037359:AAHI25ES9xCOMPokpYoz-p8XVrZUdygo2J4"
PUBLISHED_VECTOR = (
    "user=%7B%22id%22%3A279058397%2C%22first_name%22%3A%22Vladislav%20%2B%20-%20"
    "%3F%20%5C%2F%22%2C%22last_name%22%3A%22Kibenko%22%2C%22username%22%3A%22"
    "vdkfrost%22%2C%22language_code%22%3A%22ru%22%2C%22is_premium%22%3Atrue%2C"
    "%22allows_write_to_pm%22%3Atrue%2C%22photo_url%22%3A%22https%3A%5C%2F%5C%2F"
    "t.me%5C%2Fi%5C%2Fuserpic%5C%2F320%5C%2F4FPEE4tmP3ATHa57u6MqTDih13LTOiMoKo"
    "LDRG4PnSA.svg%22%7D&chat_instance=8134722200314281151&chat_type=private&"
    "auth_date=1733509682&signature=TYJxVcisqbWjtodPepiJ6ghziUL94-KNpG8Pau-"
    "X7oNNLNBM72APCpi_RKiUlBvcqo5L-LAxIc3dnTzcZX_PDg&hash="
    "a433d8f9847bd6addcc563bff7cc82c89e97ea0d90c11fe5729cae6796a36d73"
)


def signed_init_data(
    user_id: int,
    *,
    auth_date: int,
    token: str = BOT_TOKEN,
    first_name: str = "Admin",
    extra_fields: dict[str, str] | None = None,
) -> str:
    fields = {
        "auth_date": str(auth_date),
        "query_id": "AAH-test-query",
        "user": json.dumps(
            {
                "id": user_id,
                "first_name": first_name,
                "username": "admin_test",
            },
            separators=(",", ":"),
            ensure_ascii=False,
        ),
    }
    fields.update(extra_fields or {})
    data_check_string = "\n".join(f"{key}={value}" for key, value in sorted(fields.items()))
    secret_key = hmac.new(b"WebAppData", token.encode(), hashlib.sha256).digest()
    fields["hash"] = hmac.new(
        secret_key,
        data_check_string.encode(),
        hashlib.sha256,
    ).hexdigest()
    return urlencode(fields)


class FakeRedis:
    def __init__(self, members: set[str] | None = None) -> None:
        self.members = set(members or ())
        self.sadd_calls: list[tuple[str, ...]] = []

    def smembers(self, _key: str) -> set[str]:
        return set(self.members)

    def sadd(self, _key: str, *values: str) -> int:
        self.sadd_calls.append(tuple(values))
        before = len(self.members)
        self.members.update(values)
        return len(self.members) - before


class TelegramInitDataTests(unittest.TestCase):
    def test_published_vector_with_new_signature_field_validates(self) -> None:
        result = validate_telegram_init_data(
            PUBLISHED_VECTOR,
            PUBLISHED_VECTOR_TOKEN,
            now=1_733_509_682,
        )
        self.assertEqual(279_058_397, result.user.id)

    def test_valid_init_data_returns_trusted_user(self) -> None:
        now = int(time.time())
        result = validate_telegram_init_data(
            signed_init_data(42, auth_date=now),
            BOT_TOKEN,
            now=now,
        )
        self.assertEqual(42, result.user.id)
        self.assertEqual("Admin", result.user.first_name)
        self.assertEqual("AAH-test-query", result.query_id)

    def test_tampered_user_is_rejected(self) -> None:
        now = int(time.time())
        payload = signed_init_data(42, auth_date=now).replace(
            "admin_test",
            "attacker",
        )
        with self.assertRaisesRegex(TelegramInitDataError, "signature"):
            validate_telegram_init_data(payload, BOT_TOKEN, now=now)

    def test_expired_and_future_data_are_rejected(self) -> None:
        now = int(time.time())
        with self.assertRaisesRegex(TelegramInitDataError, "expired"):
            validate_telegram_init_data(
                signed_init_data(42, auth_date=now - 3_601),
                BOT_TOKEN,
                now=now,
            )
        with self.assertRaisesRegex(TelegramInitDataError, "future"):
            validate_telegram_init_data(
                signed_init_data(42, auth_date=now + 31),
                BOT_TOKEN,
                now=now,
            )

    def test_duplicate_fields_are_rejected(self) -> None:
        now = int(time.time())
        payload = signed_init_data(42, auth_date=now) + "&auth_date=1"
        with self.assertRaisesRegex(TelegramInitDataError, "duplicate"):
            validate_telegram_init_data(payload, BOT_TOKEN, now=now)


class TelegramAdminAuthorizerTests(unittest.IsolatedAsyncioTestCase):
    async def test_existing_redis_allowlist_authorizes_only_members(self) -> None:
        now = int(time.time())
        authorizer = TelegramAdminAuthorizer().configure(
            redis_client=FakeRedis({"42"}),
        )
        session = await authorizer.authorize(
            signed_init_data(42, auth_date=now),
            BOT_TOKEN,
        )
        self.assertEqual(42, session.user.id)
        with self.assertRaises(PermissionError):
            await authorizer.authorize(
                signed_init_data(99, auth_date=now),
                BOT_TOKEN,
            )

    async def test_configured_ids_are_migrated_when_redis_set_is_empty(self) -> None:
        redis = FakeRedis()
        authorizer = TelegramAdminAuthorizer().configure(
            redis_client=redis,
            fallback_admin_ids={42, 7},
        )
        self.assertEqual(frozenset({7, 42}), await authorizer.load_ids(force=True))
        self.assertEqual({("7", "42")}, set(redis.sadd_calls))


if __name__ == "__main__":
    unittest.main()
