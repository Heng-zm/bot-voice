from __future__ import annotations

import asyncio
import json
import unittest
from typing import Any

import httpx
from fastapi import HTTPException
from starlette.requests import Request

from app.api.v1.telegram import (
    TelegramWebhookTransport,
    _read_limited_webhook_body,
)
from app.services.telegram.client import (
    TelegramWebhookClient,
    build_webhook_target_url,
    parse_allowed_updates,
)


def make_request(body: bytes, headers: dict[str, str] | None = None) -> Request:
    sent = False

    async def receive() -> dict[str, Any]:
        nonlocal sent
        if sent:
            return {"type": "http.disconnect"}
        sent = True
        return {"type": "http.request", "body": body, "more_body": False}

    raw_headers = [
        (key.lower().encode("latin-1"), value.encode("latin-1"))
        for key, value in (headers or {}).items()
    ]
    return Request(
        {
            "type": "http",
            "http_version": "1.1",
            "method": "POST",
            "scheme": "https",
            "path": "/telegram/webhook",
            "raw_path": b"/telegram/webhook",
            "query_string": b"",
            "headers": raw_headers,
            "client": ("127.0.0.1", 12345),
            "server": ("testserver", 443),
        },
        receive,
    )


class TelegramWebhookRequestTests(unittest.IsolatedAsyncioTestCase):
    async def test_stream_and_content_length_limits_are_enforced(self) -> None:
        with self.assertRaises(HTTPException) as oversized_header:
            await _read_limited_webhook_body(
                make_request(b"{}", {"content-length": "100"}),
                10,
            )
        self.assertEqual(413, oversized_header.exception.status_code)

        with self.assertRaises(HTTPException) as oversized_stream:
            await _read_limited_webhook_body(make_request(b"x" * 11), 10)
        self.assertEqual(413, oversized_stream.exception.status_code)

        with self.assertRaises(HTTPException) as invalid_header:
            await _read_limited_webhook_body(
                make_request(b"{}", {"content-length": "invalid"}),
                10,
            )
        self.assertEqual(400, invalid_header.exception.status_code)

    async def test_transport_rejects_bad_secret_before_parsing(self) -> None:
        transport = TelegramWebhookTransport(
            bot_mode_provider=lambda: "WEBHOOK",
            secret_provider=lambda: "correct-secret",
            application_provider=lambda: object(),
            ready_provider=lambda: True,
            active_owner_provider=lambda: True,
        )
        request = make_request(
            b'{"update_id":1}',
            {"X-Telegram-Bot-Api-Secret-Token": "wrong-secret"},
        )

        with self.assertRaises(HTTPException) as rejected:
            await transport.process(request, "correct-secret")

        self.assertEqual(403, rejected.exception.status_code)

    async def test_claimed_update_completes_in_background(self) -> None:
        completed = asyncio.Event()
        processed: list[int] = []

        class Application:
            bot = object()

            async def process_update(self, update) -> None:
                processed.append(update.update_id)

        async def claim(_update_id, *, include_token=False):
            self.assertTrue(include_token)
            return "claimed", "owner-token"

        async def complete(update_id, *, claim_token=None):
            self.assertEqual(404, update_id)
            self.assertEqual("owner-token", claim_token)
            completed.set()
            return True

        async def release(_update_id, *, claim_token=None):
            self.fail(f"Unexpected release for {claim_token}")

        transport = TelegramWebhookTransport(
            bot_mode_provider=lambda: "WEBHOOK",
            secret_provider=lambda: "secret",
            application_provider=Application,
            ready_provider=lambda: True,
            active_owner_provider=lambda: True,
            max_body_provider=lambda: 1024,
            json_loader=json.loads,
            claim_update=claim,
            complete_update=complete,
            release_update=release,
        )
        request = make_request(
            b'{"update_id":404}',
            {"X-Telegram-Bot-Api-Secret-Token": "secret"},
        )

        response = await transport.process(request, "secret")
        await asyncio.wait_for(completed.wait(), timeout=1.0)

        self.assertEqual(200, response.status_code)
        self.assertEqual({"status": "ok", "update_id": 404}, json.loads(response.body))
        self.assertEqual([404], processed)

    async def test_completed_duplicate_is_acknowledged_without_processing(self) -> None:
        metrics: list[str] = []

        class Application:
            bot = object()

            async def process_update(self, _update) -> None:
                self.fail("Duplicate update must not be processed")

        async def claim(_update_id, *, include_token=False):
            return "completed", None

        transport = TelegramWebhookTransport(
            bot_mode_provider=lambda: "WEBHOOK",
            secret_provider=lambda: "secret",
            application_provider=Application,
            ready_provider=lambda: True,
            active_owner_provider=lambda: True,
            json_loader=json.loads,
            metric_callback=metrics.append,
            claim_update=claim,
        )
        request = make_request(
            b'{"update_id":505}',
            {"X-Telegram-Bot-Api-Secret-Token": "secret"},
        )

        response = await transport.process(request, "secret")

        self.assertEqual({"status": "ok", "duplicate": True}, json.loads(response.body))
        self.assertEqual(["replay_dropped"], metrics)


class TelegramWebhookClientTests(unittest.IsolatedAsyncioTestCase):
    def test_url_and_allowed_update_normalization(self) -> None:
        self.assertEqual(
            "https://bot.example/tg-webhook-a%2Fb",
            build_webhook_target_url("https://bot.example/", "a/b"),
        )
        self.assertEqual(
            ["message", "callback_query"],
            parse_allowed_updates("message,invalid-name,message,callback_query"),
        )

    async def test_set_webhook_retries_rate_limit_and_preserves_payload(self) -> None:
        class Response:
            text = ""

            def __init__(self, status_code: int, payload: dict[str, Any]) -> None:
                self.status_code = status_code
                self.content = json.dumps(payload).encode()

        class Client:
            def __init__(self) -> None:
                self.responses = [
                    Response(429, {"ok": False, "parameters": {"retry_after": 2}}),
                    Response(200, {"ok": True, "result": True}),
                ]
                self.payloads: list[dict[str, Any]] = []

            async def __aenter__(self):
                return self

            async def __aexit__(self, *_args) -> None:
                return None

            async def post(self, _url: str, *, json: dict[str, Any]):
                self.payloads.append(json)
                return self.responses.pop(0)

        fake_client = Client()
        sleeps: list[float] = []

        async def sleep(seconds: float) -> None:
            sleeps.append(seconds)

        client = TelegramWebhookClient(
            bot_token_provider=lambda: "bot-token",
            target_url_builder=lambda secret: build_webhook_target_url(
                "https://bot.example",
                secret,
            ),
            allowed_updates_provider=lambda: "message,message,callback_query",
            drop_pending_provider=lambda: False,
            limits_provider=httpx.Limits,
            set_max_attempts_provider=lambda: 2,
            json_loader=json.loads,
            client_factory=lambda **_kwargs: fake_client,
            sleep=sleep,
        )

        await client.configure_for_secret("secret")

        self.assertEqual([2], sleeps)
        self.assertEqual(2, len(fake_client.payloads))
        self.assertEqual(
            ["message", "callback_query"],
            fake_client.payloads[-1]["allowed_updates"],
        )
        self.assertEqual("secret", fake_client.payloads[-1]["secret_token"])
