"""Unit tests for TikTok Downloader service."""

from __future__ import annotations

import asyncio
from pathlib import Path
import sys
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_venv_site = Path(r"F:\ai project\bot-voice\.venv\Lib\site-packages")
if _venv_site.exists() and str(_venv_site) not in sys.path:
    sys.path.append(str(_venv_site))

if "httpx" not in sys.modules:
    try:
        import httpx  # noqa: F401
    except ImportError:
        sys.modules["httpx"] = MagicMock()

if "fastapi" not in sys.modules or not hasattr(sys.modules["fastapi"], "FastAPI"):
    try:
        import fastapi  # noqa: F401
    except ImportError:
        import types
        fastapi_mod = types.ModuleType("fastapi")
        fastapi_responses = types.ModuleType("fastapi.responses")
        fastapi_responses.JSONResponse = type("JSONResponse", (), {"media_type": "application/json"})
        class HTTPException(Exception):
            def __init__(self, status_code: int = 500, detail: str = ""):
                self.status_code = status_code
                self.detail = detail
        class MockFastAPI:
            def __init__(self, *args, **kwargs):
                pass
            def include_router(self, *args, **kwargs):
                pass
            def middleware(self, *args, **kwargs):
                return lambda f: f
            def get(self, *args, **kwargs):
                return lambda f: f
            def post(self, *args, **kwargs):
                return lambda f: f
            def add_middleware(self, *args, **kwargs):
                pass

        class MockAPIRouter:
            def __init__(self, *args, **kwargs):
                pass
            def get(self, *args, **kwargs):
                return lambda f: f
            def post(self, *args, **kwargs):
                return lambda f: f
            def head(self, *args, **kwargs):
                return lambda f: f
            def put(self, *args, **kwargs):
                return lambda f: f
            def delete(self, *args, **kwargs):
                return lambda f: f
            def patch(self, *args, **kwargs):
                return lambda f: f
            def options(self, *args, **kwargs):
                return lambda f: f
            def include_router(self, *args, **kwargs):
                pass

        fastapi_mod.FastAPI = MockFastAPI
        fastapi_mod.APIRouter = MockAPIRouter
        fastapi_mod.HTTPException = HTTPException
        fastapi_mod.Header = lambda default=None, **kw: default
        fastapi_mod.Request = type("Request", (), {})
        sys.modules["fastapi"] = fastapi_mod
        sys.modules["fastapi.responses"] = fastapi_responses

from app.services.downloader.tiktok import (
    _normalize_tiktok_data,
    cache_tiktok,
    clear_tiktok_cache,
    cmd_tiktok,
    download_media_to_file,
    extract_tiktok_url,
    fetch_tiktok_data,
    get_cached_tiktok,
    get_tiktok_video_kb,
    handle_tiktok_ai_summary,
    handle_tiktok_download,
    handle_tiktok_file_download,
    handle_tiktok_mp3_download,
    handle_tiktok_stats,
    is_tiktok_url,
    tiktok_callback,
)


class TestTikTokDownloader(unittest.IsolatedAsyncioTestCase):
    """Test suite for TikTok URL matching, metadata fetching, delivery, and callbacks."""

    def setUp(self) -> None:
        clear_tiktok_cache()

    def test_is_tiktok_url(self) -> None:
        valid_urls = [
            "https://vt.tiktok.com/ZSjR12345/",
            "http://vm.tiktok.com/abcde/",
            "https://www.tiktok.com/@user.name/video/7123456789012345678",
            "https://www.tiktok.com/@user.name/photo/7123456789012345678",
            "https://m.tiktok.com/v/7123456789012345678.html",
            "https://tiktok.com/@creator/video/987654321?is_from_webapp=1",
            "Check this out: https://vt.tiktok.com/ZSabcde/ so cool!",
        ]
        for url in valid_urls:
            self.assertTrue(is_tiktok_url(url), f"Failed for {url}")

        invalid_urls = [
            "https://facebook.com/watch/?v=123",
            "https://youtube.com/watch?v=123",
            "https://not-tiktok.com/video/123",
            "hello world",
            "",
        ]
        for url in invalid_urls:
            self.assertFalse(is_tiktok_url(url), f"Failed for {url}")

    def test_extract_tiktok_url(self) -> None:
        text = "មើលវីដេអូនេះ https://vt.tiktok.com/ZSjR12345/ ឡូយណាស់"
        extracted = extract_tiktok_url(text)
        self.assertEqual("https://vt.tiktok.com/ZSjR12345/", extracted)
        self.assertIsNone(extract_tiktok_url("គ្មានតំណភ្ជាប់ទេ"))

    def test_normalize_tiktok_data(self) -> None:
        raw = {
            "id": "7123456789",
            "title": "Fun Cambodian Street Food",
            "play": "https://cdn.tikwm.com/video.mp4",
            "hdplay": "https://cdn.tikwm.com/hd_video.mp4",
            "music": "https://cdn.tikwm.com/music.mp3",
            "music_info": {"title": "Khmer Folk Music", "author": "Traditional"},
            "author": {"nickname": "Heng Creator", "unique_id": "heng_khmer"},
            "duration": 45,
            "play_count": 50000,
            "digg_count": 3500,
            "images": ["https://cdn.tikwm.com/img1.jpg", "https://cdn.tikwm.com/img2.jpg"],
        }
        data = _normalize_tiktok_data(raw)
        self.assertEqual("7123456789", data["id"])
        self.assertEqual("Fun Cambodian Street Food", data["title"])
        self.assertEqual("https://cdn.tikwm.com/hd_video.mp4", data["hdplay"])
        self.assertEqual("https://cdn.tikwm.com/music.mp3", data["music"])
        self.assertEqual("Khmer Folk Music", data["music_title"])
        self.assertEqual("Traditional", data["music_author"])
        self.assertEqual("heng_khmer", data["author_username"])
        self.assertEqual(2, len(data["images"]))
        self.assertEqual(45, data["duration"])

    def test_tiktok_caching(self) -> None:
        self.assertIsNone(get_cached_tiktok("12345"))
        cache_tiktok("12345", {"title": "Test Video", "video_file_id": "vid_fid_999"})
        cached = get_cached_tiktok("12345")
        self.assertIsNotNone(cached)
        self.assertEqual("vid_fid_999", cached["video_file_id"])

        cleared = clear_tiktok_cache()
        self.assertEqual(1, cleared)
        self.assertIsNone(get_cached_tiktok("12345"))

    def test_get_tiktok_video_kb(self) -> None:
        kb = get_tiktok_video_kb("video123", has_music=True)
        all_callbacks = [btn.callback_data for row in kb.inline_keyboard for btn in row]
        self.assertIn("tt_mp3:video123", all_callbacks)
        self.assertIn("tt_file:video123", all_callbacks)
        self.assertIn("tt_ai:video123", all_callbacks)
        self.assertIn("tt_stats:video123", all_callbacks)
        self.assertIn("close_msg", all_callbacks)

        kb_no_music = get_tiktok_video_kb("video123", has_music=False)
        callbacks_no_music = [btn.callback_data for row in kb_no_music.inline_keyboard for btn in row]
        self.assertNotIn("tt_mp3:video123", callbacks_no_music)
        self.assertIn("tt_file:video123", callbacks_no_music)
        self.assertIn("tt_ai:video123", callbacks_no_music)
        self.assertIn("tt_stats:video123", callbacks_no_music)

    async def test_handle_tiktok_download_video_success(self) -> None:
        mock_update = MagicMock()
        mock_msg = MagicMock()
        mock_status_msg = MagicMock()
        mock_status_msg.delete = AsyncMock()
        mock_msg.reply_text = AsyncMock(return_value=mock_status_msg)

        mock_sent_video = MagicMock()
        mock_sent_video.video.file_id = "new_tg_file_id_777"
        mock_msg.reply_video = AsyncMock(return_value=mock_sent_video)
        mock_update.effective_message = mock_msg
        mock_context = MagicMock()

        sample_data = {
            "id": "7123456789",
            "title": "Great Video",
            "play": "https://example.com/video.mp4",
            "hdplay": "https://example.com/hd_video.mp4",
            "music": "https://example.com/music.mp3",
            "author_username": "test_creator",
            "duration": 30,
            "likes": 1200,
            "images": [],
        }

        with patch("app.services.downloader.tiktok.fetch_tiktok_data", new_callable=AsyncMock, return_value=sample_data):
            await handle_tiktok_download(mock_update, mock_context, "https://vt.tiktok.com/ZSjR12345/")

            mock_msg.reply_video.assert_awaited_once()
            mock_status_msg.delete.assert_awaited_once()

            # Verify file_id is cached
            cached = get_cached_tiktok("7123456789")
            self.assertIsNotNone(cached)
            self.assertEqual("new_tg_file_id_777", cached.get("video_file_id"))

    async def test_handle_tiktok_download_large_video_exceeds_50mb(self) -> None:
        mock_update = MagicMock()
        mock_msg = MagicMock()
        mock_status_msg = MagicMock()
        mock_status_msg.delete = AsyncMock()
        mock_msg.reply_text = AsyncMock(return_value=mock_status_msg)
        mock_update.effective_message = mock_msg
        mock_context = MagicMock()

        # Both HD and SD exceed 50MB (e.g. 10-minute long video)
        large_video_data = {
            "id": "large_vid_999",
            "title": "A Very Long 10-Minute TikTok Video",
            "play": "https://example.com/sd_long.mp4",
            "hdplay": "https://example.com/hd_long.mp4",
            "music": "https://example.com/music.mp3",
            "author_username": "long_creator",
            "duration": 600,
            "likes": 50000,
            "size": 95 * 1024 * 1024,
            "hd_size": 95 * 1024 * 1024,
            "sd_size": 60 * 1024 * 1024,
            "images": [],
        }

        with patch("app.services.downloader.tiktok.fetch_tiktok_data", new_callable=AsyncMock, return_value=large_video_data):
            await handle_tiktok_download(mock_update, mock_context, "https://vt.tiktok.com/ZSlongVideo/")

            # Status message should be deleted
            mock_status_msg.delete.assert_awaited_once()
            # Bot should send large video notification card with direct URL button
            self.assertEqual(2, mock_msg.reply_text.await_count)
            card_call = mock_msg.reply_text.call_args_list[1]
            caption_text = card_call[0][0]
            self.assertIn("វីដេអូ TikTok វែង (ទំហំលើសពី 50MB)", caption_text)
            self.assertIn("95.0 MB", caption_text)
            self.assertIn("10 នាទី", caption_text)
            # Check inline button has URL
            kb = card_call[1].get("reply_markup")
            self.assertIsNotNone(kb)
            all_urls = [btn.url for row in kb.inline_keyboard for btn in row if getattr(btn, "url", None)]
            self.assertIn("https://example.com/hd_long.mp4", all_urls)

    async def test_handle_tiktok_download_hd_large_sd_fallback(self) -> None:
        mock_update = MagicMock()
        mock_msg = MagicMock()
        mock_status_msg = MagicMock()
        mock_status_msg.delete = AsyncMock()
        mock_msg.reply_text = AsyncMock(return_value=mock_status_msg)
        mock_sent_video = MagicMock()
        mock_sent_video.video.file_id = "sd_video_fid_123"
        mock_msg.reply_video = AsyncMock(return_value=mock_sent_video)
        mock_update.effective_message = mock_msg
        mock_context = MagicMock()

        # HD is 85MB (>50MB), but SD is 35MB (<=50MB) -> falls back to SD
        hd_large_sd_fit = {
            "id": "vid_sd_fallback",
            "title": "HD is big but SD fits",
            "play": "https://example.com/sd_fits.mp4",
            "hdplay": "https://example.com/hd_large.mp4",
            "music": "https://example.com/sound.mp3",
            "author_username": "smart_creator",
            "duration": 180,
            "likes": 2500,
            "size": 85 * 1024 * 1024,
            "hd_size": 85 * 1024 * 1024,
            "sd_size": 35 * 1024 * 1024,
            "images": [],
        }

        with patch("app.services.downloader.tiktok.fetch_tiktok_data", new_callable=AsyncMock, return_value=hd_large_sd_fit), \
             patch("app.services.downloader.tiktok.download_media_bytes", new_callable=AsyncMock, return_value=b"fake_sd_video_bytes"):
            await handle_tiktok_download(mock_update, mock_context, "https://vt.tiktok.com/ZSfallback/")

            mock_msg.reply_video.assert_awaited_once()
            caption_sent = mock_msg.reply_video.call_args[1].get("caption", "")
            self.assertIn("Standard គ្មាន Watermark", caption_sent)
            self.assertIn("35.0 MB", caption_sent)

    async def test_handle_tiktok_download_photo_carousel(self) -> None:
        mock_update = MagicMock()
        mock_msg = MagicMock()
        mock_status_msg = MagicMock()
        mock_status_msg.delete = AsyncMock()
        mock_msg.reply_text = AsyncMock(return_value=mock_status_msg)
        mock_msg.reply_media_group = AsyncMock()
        mock_update.effective_message = mock_msg
        mock_context = MagicMock()

        sample_photo_data = {
            "id": "photo_carousel_123",
            "title": "Photo Slide Test",
            "author_username": "photographer",
            "images": ["https://example.com/img1.jpg", "https://example.com/img2.jpg"],
            "music": "https://example.com/slide_sound.mp3",
            "music_title": "Slide Sound",
        }

        with patch("app.services.downloader.tiktok.fetch_tiktok_data", new_callable=AsyncMock, return_value=sample_photo_data):
            await handle_tiktok_download(mock_update, mock_context, "https://vt.tiktok.com/ZSphoto123/")

            mock_msg.reply_media_group.assert_awaited_once()
            mock_status_msg.delete.assert_awaited_once()

    async def test_handle_tiktok_mp3_download(self) -> None:
        mock_query = MagicMock()
        mock_msg = MagicMock()
        mock_sent_audio = MagicMock()
        mock_sent_audio.audio.file_id = "tg_audio_fid_888"
        mock_msg.reply_audio = AsyncMock(return_value=mock_sent_audio)
        mock_query.message = mock_msg
        mock_query.answer = AsyncMock()
        mock_context = MagicMock()

        cache_tiktok("vid999", {
            "music": "https://example.com/audio.mp3",
            "music_title": "Song Title",
            "music_author": "Artist",
        })

        with patch("app.services.downloader.tiktok.download_media_bytes", new_callable=AsyncMock, return_value=b"fake_mp3_bytes"):
            await handle_tiktok_mp3_download(mock_query, mock_context, "vid999")
            mock_msg.reply_audio.assert_awaited_once()
            cached = get_cached_tiktok("vid999")
            self.assertEqual("tg_audio_fid_888", cached.get("audio_file_id"))

    async def test_handle_tiktok_ai_summary(self) -> None:
        mock_query = MagicMock()
        mock_msg = MagicMock()
        mock_msg.reply_text = AsyncMock()
        mock_query.message = mock_msg
        mock_query.answer = AsyncMock()
        mock_context = MagicMock()

        cache_tiktok("vid_summary", {
            "title": "How to stay productive with artificial intelligence",
            "author_username": "tech_creator",
        })

        mock_gemini = MagicMock()
        mock_ai_resp = MagicMock()
        import app.legacy

        with patch.object(app.legacy, "_gemini", mock_gemini), \
             patch("app.services.ai.gemini.generate_content_with_fallback", return_value=mock_ai_resp), \
             patch("app.services.ai.gemini.extract_gemini_text", return_value="ខ្លឹមសារសង្ខេបពី AI"):
            await handle_tiktok_ai_summary(mock_query, mock_context, "vid_summary")
            self.assertEqual(2, mock_msg.reply_text.await_count)
            status_text = mock_msg.reply_text.call_args_list[0][0][0]
            self.assertIn("AI TIKTOK SUMMARIZER", status_text)
            sent_text = mock_msg.reply_text.call_args_list[1][0][0]
            self.assertIn("AI សង្ខេបខ្លឹមសារ TikTok", sent_text)

    async def test_handle_tiktok_file_download_video(self) -> None:
        mock_query = MagicMock()
        mock_msg = MagicMock()
        mock_sent_doc = MagicMock()
        mock_sent_doc.document.file_id = "tg_doc_fid_555"
        mock_msg.reply_document = AsyncMock(return_value=mock_sent_doc)
        mock_query.message = mock_msg
        mock_query.answer = AsyncMock()
        mock_context = MagicMock()

        cache_tiktok("vid_file_1", {
            "title": "Document Video",
            "play": "https://example.com/video.mp4",
            "hdplay": "https://example.com/hd_video.mp4",
            "author_username": "test_creator",
        })

        with patch("app.services.downloader.tiktok.download_media_bytes", new_callable=AsyncMock, return_value=b"fake_video_bytes"):
            await handle_tiktok_file_download(mock_query, mock_context, "vid_file_1")
            mock_msg.reply_document.assert_awaited_once()
            cached = get_cached_tiktok("vid_file_1")
            self.assertEqual("tg_doc_fid_555", cached.get("doc_file_id"))

        # Test fast cached 0ms delivery
        mock_msg.reply_document.reset_mock()
        await handle_tiktok_file_download(mock_query, mock_context, "vid_file_1")
        mock_msg.reply_document.assert_awaited_once()
        self.assertEqual("tg_doc_fid_555", mock_msg.reply_document.call_args[1].get("document"))

    async def test_handle_tiktok_file_download_large_video_exceeds_50mb(self) -> None:
        mock_query = MagicMock()
        mock_msg = MagicMock()
        mock_msg.reply_text = AsyncMock()
        mock_msg.reply_document = AsyncMock()
        mock_query.message = mock_msg
        mock_query.answer = AsyncMock()
        mock_context = MagicMock()

        cache_tiktok("vid_file_large", {
            "title": "Large Document Video",
            "play": "https://example.com/huge.mp4",
            "size": 90 * 1024 * 1024,
            "author_username": "large_creator",
        })

        await handle_tiktok_file_download(mock_query, mock_context, "vid_file_large")
        mock_msg.reply_document.assert_not_awaited()
        mock_msg.reply_text.assert_awaited_once()
        sent_caption = mock_msg.reply_text.call_args[0][0]
        self.assertIn("ឯកសារ TikTok (ទំហំលើសពី 50MB)", sent_caption)
        self.assertIn("90.0 MB", sent_caption)
        markup = mock_msg.reply_text.call_args[1].get("reply_markup")
        self.assertIsNotNone(markup)
        all_urls = [btn.url for row in markup.inline_keyboard for btn in row if getattr(btn, "url", None)]
        self.assertIn("https://example.com/huge.mp4", all_urls)

    async def test_handle_tiktok_file_download_photos(self) -> None:
        mock_query = MagicMock()
        mock_msg = MagicMock()
        mock_msg.reply_document = AsyncMock()
        mock_query.message = mock_msg
        mock_query.answer = AsyncMock()
        mock_context = MagicMock()

        cache_tiktok("vid_photo_files", {
            "title": "Photo Slides",
            "images": ["https://example.com/img1.jpg", "https://example.com/img2.jpg"],
            "author_username": "photographer",
        })

        with patch("app.services.downloader.tiktok.download_media_bytes", new_callable=AsyncMock, return_value=b"fake_image_bytes"):
            await handle_tiktok_file_download(mock_query, mock_context, "vid_photo_files")
            self.assertEqual(2, mock_msg.reply_document.await_count)

    async def test_handle_tiktok_file_download_inflight_deduplication(self) -> None:
        mock_query = MagicMock()
        mock_query.from_user.id = 99999
        mock_msg = MagicMock()
        mock_msg.reply_document = AsyncMock()
        mock_query.message = mock_msg
        mock_query.answer = AsyncMock()
        mock_context = MagicMock()

        cache_tiktok("vid_inflight", {
            "title": "In-flight Video",
            "play": "https://example.com/video.mp4",
        })

        from app.services.downloader.tiktok import _IN_FLIGHT_TASKS
        _IN_FLIGHT_TASKS.add("file:99999:vid_inflight")
        try:
            await handle_tiktok_file_download(mock_query, mock_context, "vid_inflight")
            mock_msg.reply_document.assert_not_awaited()
            mock_query.answer.assert_awaited_once()
        finally:
            _IN_FLIGHT_TASKS.discard("file:99999:vid_inflight")

    async def test_cmd_tiktok_usage(self) -> None:
        mock_update = MagicMock()
        mock_msg = MagicMock()
        mock_msg.reply_text = AsyncMock()
        mock_update.effective_message = mock_msg
        mock_context = MagicMock()
        mock_context.args = []

        await cmd_tiktok(mock_update, mock_context)
        mock_msg.reply_text.assert_awaited_once()
        text_arg = mock_msg.reply_text.call_args[0][0]
        self.assertIn("របៀបប្រើប្រាស់ TikTok Downloader", text_arg)
        self.assertIn("reply_markup", mock_msg.reply_text.call_args[1])
        markup = mock_msg.reply_text.call_args[1]["reply_markup"]
        self.assertIsNotNone(markup)

    async def test_handle_tiktok_stats(self) -> None:
        mock_query = MagicMock()
        mock_query.answer = AsyncMock()
        mock_context = MagicMock()

        # 1. Non-cached stats
        await handle_tiktok_stats(mock_query, mock_context, "vid_missing")
        mock_query.answer.assert_awaited_once()
        self.assertIn("មិនមានទិន្នន័យស្ថិតិ", mock_query.answer.call_args[0][0])
        self.assertTrue(mock_query.answer.call_args[1].get("show_alert"))

        # 2. Cached stats with rich metrics
        mock_query.answer.reset_mock()
        cache_tiktok("vid_stats_1", {
            "author_username": "heng_creator",
            "author_name": "Heng",
            "views": 250000,
            "likes": 18500,
            "comments": 420,
            "shares": 135,
            "downloads": 88,
            "duration": 45,
            "music_title": "Popular Melody",
            "size": 15 * 1024 * 1024,
        })
        await handle_tiktok_stats(mock_query, mock_context, "vid_stats_1")
        mock_query.answer.assert_awaited_once()
        stats_text = mock_query.answer.call_args[0][0]
        self.assertIn("ស្ថិតិមេឌៀ TikTok (@heng_creator (Heng))", stats_text)
        self.assertIn("250,000 ដង", stats_text)
        self.assertIn("18,500 នាក់", stats_text)
        self.assertIn("420 មតិ", stats_text)
        self.assertIn("135 ដង", stats_text)
        self.assertIn("15.0 MB", stats_text)
        self.assertTrue(mock_query.answer.call_args[1].get("show_alert"))

    async def test_tiktok_callback_router(self) -> None:
        mock_update = MagicMock()
        mock_query = MagicMock()
        mock_update.callback_query = mock_query
        mock_context = MagicMock()

        with patch("app.services.downloader.tiktok.handle_tiktok_mp3_download", new_callable=AsyncMock) as mock_mp3, \
             patch("app.services.downloader.tiktok.handle_tiktok_file_download", new_callable=AsyncMock) as mock_file, \
             patch("app.services.downloader.tiktok.handle_tiktok_ai_summary", new_callable=AsyncMock) as mock_ai, \
             patch("app.services.downloader.tiktok.handle_tiktok_stats", new_callable=AsyncMock) as mock_stats:
            mock_query.data = "tt_mp3:12345"
            await tiktok_callback(mock_update, mock_context)
            mock_mp3.assert_awaited_once_with(mock_query, mock_context, "12345")

            mock_query.data = "tt_file:11223"
            await tiktok_callback(mock_update, mock_context)
            mock_file.assert_awaited_once_with(mock_query, mock_context, "11223")

            mock_query.data = "tt_ai:67890"
            await tiktok_callback(mock_update, mock_context)
            mock_ai.assert_awaited_once_with(mock_query, mock_context, "67890")

            mock_query.data = "tt_stats:54321"
            await tiktok_callback(mock_update, mock_context)
            mock_stats.assert_awaited_once_with(mock_query, mock_context, "54321")

    async def test_download_media_to_file_max_bytes_cleanup(self) -> None:
        import os
        import tempfile

        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp_path = tmp.name

        try:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.headers = {}

            async def fake_aiter(chunk_size=65536):
                yield b"A" * 600
                yield b"B" * 600

            mock_resp.aiter_bytes = fake_aiter

            mock_stream_ctx = MagicMock()
            mock_stream_ctx.__aenter__ = AsyncMock(return_value=mock_resp)
            mock_stream_ctx.__aexit__ = AsyncMock(return_value=None)

            mock_client = MagicMock()
            mock_client.stream.return_value = mock_stream_ctx
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)

            with patch("httpx.AsyncClient", return_value=mock_client):
                res = await download_media_to_file("https://example.com/test_stream.mp4", tmp_path, max_bytes=500)
                self.assertIsNone(res)
                # Ensure the partial file was cleaned up and deleted immediately
                self.assertFalse(os.path.exists(tmp_path))
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


if __name__ == "__main__":
    unittest.main()

