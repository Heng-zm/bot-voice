"""Unit tests for Server Request Lifecycle Logging Middleware."""

from __future__ import annotations

import asyncio
import logging
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from app.core.logging_middleware import (
    Response as FallbackResponse,
    get_active_requests_count,
    get_client_ip,
    get_status_indicator,
    request_lifecycle_logging_middleware,
)

try:
    from fastapi import FastAPI, Request, Response
    from starlette.testclient import TestClient

    HAS_FASTAPI = True
except (ImportError, ModuleNotFoundError):
    FastAPI = None  # type: ignore[assignment,misc]
    Request = None  # type: ignore[assignment,misc]
    Response = FallbackResponse  # type: ignore[assignment,misc]
    TestClient = None  # type: ignore[assignment,misc]
    HAS_FASTAPI = False


class TestServerLoggingMiddlewareCore(unittest.IsolatedAsyncioTestCase):
    """Core logic test suite that verifies request logging lifecycle in pure Python."""

    def test_get_status_indicator(self) -> None:
        self.assertEqual(("✅", "OK"), get_status_indicator(200))
        self.assertEqual(("✅", "OK"), get_status_indicator(204))
        self.assertEqual(("↪️", "REDIRECT"), get_status_indicator(302))
        self.assertEqual(("⚠️", "CLIENT_ERROR"), get_status_indicator(404))
        self.assertEqual(("❌", "SERVER_ERROR"), get_status_indicator(500))

    def test_get_client_ip_headers(self) -> None:
        req = MagicMock()
        req.headers = {"x-forwarded-for": "203.0.113.195, 10.0.0.1"}
        self.assertEqual("203.0.113.195", get_client_ip(req))

        req.headers = {"cf-connecting-ip": "198.51.100.4"}
        self.assertEqual("198.51.100.4", get_client_ip(req))

        req.headers = {"x-real-ip": "192.0.2.1"}
        self.assertEqual("192.0.2.1", get_client_ip(req))

        req.headers = {}
        req.client.host = "127.0.0.1"
        self.assertEqual("127.0.0.1", get_client_ip(req))

        req.client = None
        self.assertEqual("unknown", get_client_ip(req))

    async def test_get_request_lifecycle_logging(self) -> None:
        req = MagicMock()
        req.method = "GET"
        req.url.path = "/api/v1/status"
        req.url.query = "format=json"
        req.headers = {"x-request-id": "req-12345", "x-forwarded-for": "10.0.0.99"}

        mock_resp = FallbackResponse(status_code=200, headers={})
        call_next = AsyncMock(return_value=mock_resp)

        with self.assertLogs("app.server", level=logging.INFO) as log:
            resp = await request_lifecycle_logging_middleware(req, call_next)
            self.assertEqual(200, resp.status_code)
            self.assertEqual("req-12345", resp.headers.get("X-Request-ID"))
            self.assertIn("X-Response-Time-ms", resp.headers)

            log_output = "\n".join(log.output)
            self.assertIn("⏳ [PENDING] GET /api/v1/status?format=json from 10.0.0.99", log_output)
            self.assertIn("✅ [200 OK] GET /api/v1/status?format=json from 10.0.0.99", log_output)

    async def test_post_request_lifecycle_logging(self) -> None:
        req = MagicMock()
        req.method = "POST"
        req.url.path = "/webhook"
        req.url.query = ""
        req.headers = {"x-request-id": "req-9999", "x-forwarded-for": "149.154.167.220"}

        mock_resp = FallbackResponse(status_code=200, headers={})
        call_next = AsyncMock(return_value=mock_resp)

        with self.assertLogs("app.server", level=logging.INFO) as log:
            resp = await request_lifecycle_logging_middleware(req, call_next)
            self.assertEqual(200, resp.status_code)

            log_output = "\n".join(log.output)
            self.assertIn("⏳ [PENDING] POST /webhook from 149.154.167.220", log_output)
            self.assertIn("✅ [200 OK] POST /webhook from 149.154.167.220", log_output)

    async def test_options_request_preflight_handling(self) -> None:
        req = MagicMock()
        req.method = "OPTIONS"
        req.url.path = "/ai-assistant/info"
        req.url.query = ""
        req.headers = {
            "origin": "https://example.com",
            "access-control-request-headers": "Authorization, Content-Type",
        }

        # Simulate route returning 405 Method Not Allowed for OPTIONS
        mock_resp = FallbackResponse(status_code=405, headers={})
        call_next = AsyncMock(return_value=mock_resp)

        with self.assertLogs("app.server", level=logging.INFO) as log:
            resp = await request_lifecycle_logging_middleware(req, call_next)
            self.assertEqual(200, resp.status_code)
            self.assertEqual("https://example.com", resp.headers.get("Access-Control-Allow-Origin"))
            self.assertIn("OPTIONS", resp.headers.get("Access-Control-Allow-Methods", ""))

            log_output = "\n".join(log.output)
            self.assertIn("⏳ [PENDING] OPTIONS /ai-assistant/info", log_output)
            self.assertIn("✅ [200 OK] OPTIONS /ai-assistant/info", log_output)

    async def test_error_request_logging(self) -> None:
        req = MagicMock()
        req.method = "GET"
        req.url.path = "/crash"
        req.url.query = ""
        req.headers = {}
        req.client.host = "127.0.0.1"

        call_next = AsyncMock(side_effect=RuntimeError("Fatal database error"))

        with self.assertLogs("app.server", level=logging.ERROR) as log:
            with self.assertRaises(RuntimeError):
                await request_lifecycle_logging_middleware(req, call_next)

            log_output = "\n".join(log.output)
            self.assertIn("❌ [CRASH] GET /crash", log_output)
            self.assertIn("Fatal database error", log_output)


@unittest.skipUnless(HAS_FASTAPI, "Requires FastAPI and starlette TestClient")
class TestServerLoggingMiddlewareIntegration(unittest.TestCase):
    """Integration test suite using FastAPI TestClient when available."""

    def setUp(self) -> None:
        self.app = FastAPI()
        self.app.middleware("http")(request_lifecycle_logging_middleware)

        @self.app.get("/test/hello")
        async def hello() -> dict[str, str]:
            return {"message": "hello world"}

        @self.app.post("/test/submit")
        async def submit() -> dict[str, str]:
            return {"status": "submitted"}

        self.client = TestClient(self.app, raise_server_exceptions=False)

    def test_fastapi_get_and_post_integration(self) -> None:
        resp = self.client.get("/test/hello")
        self.assertEqual(200, resp.status_code)
        self.assertIn("X-Request-ID", resp.headers)

        resp2 = self.client.post("/test/submit")
        self.assertEqual(200, resp2.status_code)


if __name__ == "__main__":
    unittest.main()
