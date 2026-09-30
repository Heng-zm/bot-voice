"""Unit tests for app.features.tts."""

from __future__ import annotations

import unittest
from unittest.mock import AsyncMock, patch

from app.features.tts.service import TTSService, tts_service
from app.features.tts.providers.edge import EdgeTTSProvider
from app.features.tts.providers.gtts import GTTSProvider


class TTSServiceUnitTests(unittest.IsolatedAsyncioTestCase):
    def test_service_initialization(self):
        self.assertIsNotNone(tts_service)
        svc = TTSService()
        self.assertTrue(len(svc.providers) > 0)

    @patch("app.features.tts.providers.edge.EdgeTTSProvider.synthesize", new_callable=AsyncMock)
    async def test_synthesize_edge_fallback(self, mock_synth):
        mock_synth.return_value = b"test_audio_ogg"
        svc = TTSService()
        audio = await svc.synthesize("សួស្តី", gender="female", speed=1.0, preferred_provider="edge")
        self.assertEqual(audio, b"test_audio_ogg")


if __name__ == "__main__":
    unittest.main()
