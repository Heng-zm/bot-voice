from __future__ import annotations

import unittest

from app.core.network import web_server_port


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


if __name__ == "__main__":
    unittest.main()
