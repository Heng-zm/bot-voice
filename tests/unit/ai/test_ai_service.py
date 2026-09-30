"""Unit tests for app.features.ai."""

from __future__ import annotations

import unittest
from unittest.mock import AsyncMock, patch

from app.features.ai.service import AIService, ai_service


class AIServiceUnitTests(unittest.IsolatedAsyncioTestCase):
    def test_service_initialization(self):
        self.assertIsNotNone(ai_service)
        svc = AIService()
        self.assertTrue(callable(svc.ask))
        self.assertTrue(callable(svc.analyze_image))
        self.assertTrue(callable(svc.summarize))


if __name__ == "__main__":
    unittest.main()
