from __future__ import annotations

import unittest

from starlette.middleware.base import BaseHTTPMiddleware

from app.core.network import web_server_port
from app.main import RuntimeReadyMiddleware, ServerFastPathMiddleware, app


class WebServerPortTests(unittest.TestCase):
    def test_server_allocation_overrides_generic_port(self) -> None:
        port = web_server_port({"SERVER_PORT": "13961", "PORT": "8080"})

        self.assertEqual(13_961, port)

    def test_wispbyte_port_is_supported(self) -> None:
        port = web_server_port({"WISPBYTE_PORT": "13961", "PORT": "8080"})

        self.assertEqual(13_961, port)

    def test_generic_port_and_default_remain_supported(self) -> None:
        self.assertEqual(9_000, web_server_port({"PORT": "9000"}))
        self.assertEqual(8_080, web_server_port({}))

    def test_invalid_allocated_port_fails_instead_of_binding_elsewhere(self) -> None:
        for value in ("invalid", "0", "65536"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                web_server_port({"SERVER_PORT": value, "PORT": "8080"})


class ServerMiddlewareTests(unittest.TestCase):
    def test_readiness_fast_path_does_not_use_base_http_middleware(self) -> None:
        entry = next(item for item in app.user_middleware if item.cls is RuntimeReadyMiddleware)

        self.assertFalse(issubclass(entry.cls, BaseHTTPMiddleware))

    def test_request_guards_and_timing_share_one_pure_asgi_layer(self) -> None:
        entry = next(
            item for item in app.user_middleware if item.cls is ServerFastPathMiddleware
        )

        self.assertFalse(issubclass(entry.cls, BaseHTTPMiddleware))
        self.assertFalse(
            any(issubclass(item.cls, BaseHTTPMiddleware) for item in app.user_middleware)
        )

    def test_untrusted_request_id_is_safe_for_response_headers(self) -> None:
        self.assertEqual(
            "trace-123:child",
            ServerFastPathMiddleware._request_id("trace-123:child"),
        )
        generated = ServerFastPathMiddleware._request_id("bad\r\nheader")

        self.assertEqual(16, len(generated))
        self.assertTrue(generated.isascii())
        self.assertTrue(generated.isalnum())


if __name__ == "__main__":
    unittest.main()
