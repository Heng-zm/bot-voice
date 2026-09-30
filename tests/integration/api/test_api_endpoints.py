"""Integration tests for FastAPI application endpoints."""

from __future__ import annotations

import unittest
from fastapi.testclient import TestClient

from app.main import app


class APIEndpointsIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_healthz_endpoint(self):
        resp = self.client.get("/healthz")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data.get("status"), "ok")

    def test_health_endpoint(self):
        resp = self.client.get("/health")
        self.assertEqual(resp.status_code, 200)

    def test_logs_endpoint(self):
        resp = self.client.get("/logs")
        self.assertEqual(resp.status_code, 200)
        self.assertTrue(len(resp.text) > 0)


if __name__ == "__main__":
    unittest.main()
