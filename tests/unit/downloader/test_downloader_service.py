"""Unit tests for app.features.downloader."""

from __future__ import annotations

import unittest
from unittest.mock import AsyncMock, patch

from app.features.downloader.service import DownloaderService, downloader_service


class DownloaderServiceUnitTests(unittest.IsolatedAsyncioTestCase):
    def test_service_initialization(self):
        self.assertIsNotNone(downloader_service)
        svc = DownloaderService()
        self.assertTrue(len(svc.downloaders) >= 4)

    def test_platform_detection(self):
        svc = DownloaderService()
        self.assertEqual(svc.detect_platform("https://www.tiktok.com/@user/video/123"), "tiktok")
        self.assertEqual(svc.detect_platform("https://www.facebook.com/watch/?v=123"), "facebook")
        self.assertEqual(svc.detect_platform("https://www.instagram.com/reel/123/"), "instagram")
        self.assertEqual(svc.detect_platform("https://www.youtube.com/watch?v=123"), "youtube")
        self.assertIsNone(svc.detect_platform("https://example.com/other"))


if __name__ == "__main__":
    unittest.main()
