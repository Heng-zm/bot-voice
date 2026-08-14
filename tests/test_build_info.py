from __future__ import annotations

import asyncio
import os
import unittest
from unittest.mock import patch

from app.main import app
from app.services.build_info import get_build_info


class BuildInfoTests(unittest.TestCase):
    def test_build_info_uses_environment_metadata(self) -> None:
        with patch.dict(
            os.environ,
            {
                "BOT_BUILD_VERSION": "2026.08.12",
                "RELEASE_SHA": "abcdef1234567890fedcba",
                "RELEASE_CREATED_AT": "2026-08-12T10:30:00Z",
                "PROCESS_ROLE": "worker",
            },
            clear=False,
        ):
            payload = get_build_info(role=None, started_at=1_786_000_000.0)

        self.assertEqual("2026.08.12", payload["version"])
        self.assertEqual("abcdef1234567890fedcba", payload["commit"])
        self.assertEqual("abcdef123456", payload["commit_short"])
        self.assertEqual("2026-08-12T10:30:00Z", payload["deployed_at"])
        self.assertEqual("worker", payload["process_role"])
        self.assertIsNotNone(payload["runtime_started_at"])

    def test_version_routes_are_registered(self) -> None:
        paths = {route.path for route in app.routes}
        self.assertIn("/version", paths)
        self.assertIn("/api/version", paths)

    def test_version_endpoint_returns_build_metadata(self) -> None:
        route = next(route for route in app.routes if route.path == "/api/version")
        with patch.dict(
            os.environ,
            {
                "BOT_BUILD_VERSION": "2026.08.12",
                "RELEASE_SHA": "abcdef1234567890fedcba",
            },
            clear=False,
        ):
            payload = asyncio.run(route.endpoint())

        self.assertTrue(payload["ok"])
        self.assertEqual("2026.08.12", payload["build"]["version"])
        self.assertEqual("abcdef123456", payload["build"]["commit_short"])


if __name__ == "__main__":
    unittest.main()
