"""Unit tests for Facebook Video & Reels Downloader service."""

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

        class MockFastAPI:
            def __init__(self, *args, **kwargs):
                self.routes = []
            def include_router(self, *args, **kwargs):
                pass
            def middleware(self, *args, **kwargs):
                return lambda f: f

        class MockAPIRouter:
            def __init__(self, *args, **kwargs):
                self.routes = []
            def __getattr__(self, name: str):
                return lambda *args, **kwargs: (lambda f: f)
            def include_router(self, *args, **kwargs):
                pass

        class MockHTTPException(Exception):
            def __init__(self, status_code: int = 400, detail: str = "", *args, **kwargs):
                super().__init__(detail)
                self.status_code = status_code
                self.detail = detail

        fastapi_mod.FastAPI = MockFastAPI
        fastapi_mod.APIRouter = MockAPIRouter
        fastapi_mod.HTTPException = MockHTTPException
        fastapi_mod.Request = type("Request", (), {})
        fastapi_mod.Header = lambda default=None, **kw: default
        fastapi_mod.Query = lambda default=None, **kw: default
        sys.modules["fastapi"] = fastapi_mod
        sys.modules["fastapi.responses"] = fastapi_responses

from app.core.features import (
    FEATURE_FACEBOOK,
    is_facebook_enabled,
    reset_feature_overrides,
    set_feature_override,
)
from app.services.downloader.facebook import (
    _clean_json_escaped_url,
    _extract_video_id_from_url,
    _normalize_facebook_data,
    _parse_html_facebook_data,
    cache_facebook,
    clear_facebook_cache,
    cmd_facebook,
    download_fb_media_to_file,
    extract_facebook_url,
    facebook_callback,
    fetch_facebook_data,
    format_facebook_caption,
    get_cached_facebook,
    get_facebook_video_kb,
    handle_facebook_ai_summary,
    handle_facebook_download,
    handle_facebook_file_download,
    handle_facebook_mp3_download,
    handle_facebook_stats,
    is_facebook_url,
    resolve_facebook_redirect,
)


class TestFacebookDownloader(unittest.IsolatedAsyncioTestCase):
    """Test suite for Facebook Video & Reels matching, metadata fetching, and delivery."""

    def setUp(self) -> None:
        clear_facebook_cache()
        reset_feature_overrides()

    def test_is_facebook_url(self) -> None:
        valid_urls = [
            "https://www.facebook.com/reel/123456789012345",
            "https://facebook.com/reel/987654321098765/?s=ch_share",
            "https://fb.watch/abcdef1234/",
            "http://fb.watch/xyz987/",
            "https://www.facebook.com/watch/?v=112233445566778",
            "https://m.facebook.com/watch/?v=998877665544",
            "https://www.facebook.com/share/r/6vNfN5Q1abc/",
            "https://www.facebook.com/share/v/8kLmNp90xyz/",
            "https://www.facebook.com/username.page/videos/554433221100/",
            "https://web.facebook.com/reel/443322110099",
            "Check this out: https://www.facebook.com/reel/123456789/ so cool!",
        ]
        for url in valid_urls:
            self.assertTrue(is_facebook_url(url), f"Failed for {url}")

        invalid_urls = [
            "https://vt.tiktok.com/ZSjR12345/",
            "https://youtube.com/watch?v=123",
            "https://twitter.com/post/123",
            "https://notfacebook.com/reel/123",
            "hello world",
            "",
        ]
        for url in invalid_urls:
            self.assertFalse(is_facebook_url(url), f"Failed for {url}")

    def test_extract_facebook_url(self) -> None:
        text = "មើលវីដេអូនេះ https://www.facebook.com/reel/123456789012345 ឡូយណាស់"
        extracted = extract_facebook_url(text)
        self.assertEqual("https://www.facebook.com/reel/123456789012345", extracted)
        self.assertIsNone(extract_facebook_url("គ្មានតំណភ្ជាប់ Facebook ទេ"))

    def test_clean_json_escaped_url(self) -> None:
        raw = r"https:\/\/video.xx.fbcdn.net\/v\/t1.mp4?_nc_cat=1\u0026tag=test"
        cleaned = _clean_json_escaped_url(raw)
        self.assertEqual("https://video.xx.fbcdn.net/v/t1.mp4?_nc_cat=1&tag=test", cleaned)

    def test_extract_video_id(self) -> None:
        self.assertEqual("1234567890", _extract_video_id_from_url("https://www.facebook.com/reel/1234567890"))
        self.assertEqual("abcdef", _extract_video_id_from_url("https://fb.watch/abcdef/"))
        self.assertEqual("6vNfN5Q1abc", _extract_video_id_from_url("https://www.facebook.com/share/r/6vNfN5Q1abc/"))

    def test_normalize_facebook_data(self) -> None:
        raw = {
            "id": "fb_112233",
            "title": "Traditional Khmer Dance",
            "author_name": "Culture Page",
            "hd_url": "https://video.fbcdn.net/hd.mp4",
            "sd_url": "https://video.fbcdn.net/sd.mp4",
            "cover": "https://scontent.fbcdn.net/cover.jpg",
            "duration": 90,
            "views": 45000,
            "likes": 2300,
        }
        data = _normalize_facebook_data(raw)
        self.assertEqual("fb_112233", data["id"])
        self.assertEqual("Traditional Khmer Dance", data["title"])
        self.assertEqual("Culture Page", data["author_name"])
        self.assertEqual("https://video.fbcdn.net/hd.mp4", data["hd_url"])
        self.assertEqual("https://video.fbcdn.net/sd.mp4", data["sd_url"])
        self.assertEqual(90, data["duration"])

    def test_facebook_caching(self) -> None:
        self.assertIsNone(get_cached_facebook("fb_9988"))
        cache_facebook("fb_9988", {"title": "Test FB Video", "video_file_id": "vid_fid_111"})
        cached = get_cached_facebook("fb_9988")
        self.assertIsNotNone(cached)
        self.assertEqual("Test FB Video", cached["title"])
        self.assertEqual("vid_fid_111", cached["video_file_id"])

        # Update cache
        cache_facebook("fb_9988", {"mp3_file_id": "mp3_fid_222"})
        updated = get_cached_facebook("fb_9988")
        self.assertEqual("vid_fid_111", updated["video_file_id"])
        self.assertEqual("mp3_fid_222", updated["mp3_file_id"])

        clear_facebook_cache()
        self.assertIsNone(get_cached_facebook("fb_9988"))

    def test_facebook_video_kb(self) -> None:
        # 1. Dual HD & SD available
        kb = get_facebook_video_kb("vid_100", has_hd=True, has_sd=True)
        callbacks = [btn.callback_data for row in kb.inline_keyboard for btn in row]
        self.assertIn("fb_hd:vid_100", callbacks)
        self.assertIn("fb_sd:vid_100", callbacks)
        self.assertIn("fb_mp3:vid_100", callbacks)
        self.assertIn("fb_file:vid_100", callbacks)
        self.assertIn("fb_ai:vid_100", callbacks)
        self.assertIn("fb_stats:vid_100", callbacks)

        # 2. SD only available
        kb_sd = get_facebook_video_kb("vid_200", has_hd=False, has_sd=True)
        sd_cbs = [btn.callback_data for row in kb_sd.inline_keyboard for btn in row]
        self.assertNotIn("fb_hd:vid_200", sd_cbs)
        self.assertIn("fb_sd:vid_200", sd_cbs)

    def test_parse_html_facebook_data(self) -> None:
        sample_html = """
        <!DOCTYPE html>
        <html>
        <head>
        <meta property="og:title" content="Bokator Martial Arts Championship" />
        <meta property="og:image" content="https://scontent.xx.fbcdn.net/v/thumb.jpg" />
        <meta property="og:site_name" content="Khmer Sports TV" />
        <script>
        var info = {
            "playable_url": "https:\\/\\/video.xx.fbcdn.net\\/sd.mp4",
            "playable_url_quality_hd": "https:\\/\\/video.xx.fbcdn.net\\/hd.mp4"
        };
        </script>
        </head>
        <body></body>
        </html>
        """
        parsed = _parse_html_facebook_data(sample_html, "https://www.facebook.com/reel/778899")
        self.assertIsNotNone(parsed)
        self.assertEqual("Bokator Martial Arts Championship", parsed["title"])
        self.assertEqual("Khmer Sports TV", parsed["author_name"])
        self.assertEqual("https://video.xx.fbcdn.net/hd.mp4", parsed["hd_url"])
        self.assertEqual("https://video.xx.fbcdn.net/sd.mp4", parsed["sd_url"])
        self.assertEqual("https://scontent.xx.fbcdn.net/v/thumb.jpg", parsed["cover"])

    async def test_download_fb_media_to_file_max_bytes_cleanup(self) -> None:
        import os
        import tempfile

        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp_path = tmp.name

        try:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.headers = {}

            async def fake_aiter(chunk_size=65536):
                yield b"X" * 600
                yield b"Y" * 600

            mock_resp.aiter_bytes = fake_aiter

            mock_stream_ctx = MagicMock()
            mock_stream_ctx.__aenter__ = AsyncMock(return_value=mock_resp)
            mock_stream_ctx.__aexit__ = AsyncMock(return_value=None)

            mock_client = MagicMock()
            mock_client.stream.return_value = mock_stream_ctx
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)

            with patch("httpx.AsyncClient", return_value=mock_client):
                res = await download_fb_media_to_file("https://video.fbcdn.net/stream.mp4", tmp_path, max_bytes=500)
                self.assertIsNone(res)
                # Ensure the partial file was closed and unlinked
                self.assertFalse(os.path.exists(tmp_path))
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    async def test_handle_facebook_download_flow(self) -> None:
        mock_update = MagicMock()
        mock_message = MagicMock()
        mock_status_card = MagicMock()
        mock_message.reply_text = AsyncMock(return_value=mock_status_card)
        mock_status_card.edit_text = AsyncMock()
        mock_status_card.delete = AsyncMock()

        mock_sent_video = MagicMock()
        mock_sent_video.video.file_id = "fb_cached_fid_999"
        mock_message.reply_video = AsyncMock(return_value=mock_sent_video)
        mock_update.effective_message = mock_message
        mock_context = MagicMock()

        fake_fb_data = {
            "id": "vid_flow_1",
            "title": "Cambodian Street Food Review",
            "author_name": "Foodie KH",
            "hd_url": "https://video.fbcdn.net/hd_stream.mp4",
            "sd_url": "https://video.fbcdn.net/sd_stream.mp4",
            "source_url": "https://www.facebook.com/reel/vid_flow_1",
        }

        with patch("app.services.downloader.facebook.fetch_facebook_data", new_callable=AsyncMock) as mock_fetch, \
             patch("app.services.downloader.facebook.download_fb_media_to_file", new_callable=AsyncMock) as mock_dl:
            mock_fetch.return_value = fake_fb_data
            mock_dl.return_value = 1024 * 1024  # 1MB downloaded

            await handle_facebook_download(mock_update, mock_context, "https://www.facebook.com/reel/vid_flow_1")

            mock_fetch.assert_awaited_once_with("https://www.facebook.com/reel/vid_flow_1")
            mock_message.reply_video.assert_awaited_once()
            call_kwargs = mock_message.reply_video.call_args[1]
            self.assertIn("Cambodian Street Food Review", call_kwargs["caption"])
            self.assertIn("Foodie KH", call_kwargs["caption"])

            # Verify cached file_id
            cached = get_cached_facebook("vid_flow_1")
            self.assertEqual("fb_cached_fid_999", cached["video_file_id"])

    async def test_handle_facebook_download_cached_file_id(self) -> None:
        video_id = "123456789012"
        cache_facebook(video_id, {
            "id": video_id,
            "title": "Instant Cached FB Video",
            "author_name": "Fast Creator",
            "hd_url": "https://video.fbcdn.net/hd.mp4",
            "sd_url": "https://video.fbcdn.net/sd.mp4",
            "video_file_id": "instant_fid_888",
            "source_url": f"https://www.facebook.com/reel/{video_id}",
        })

        mock_update = MagicMock()
        mock_message = MagicMock()
        mock_status_card = MagicMock()
        mock_message.reply_text = AsyncMock(return_value=mock_status_card)
        mock_status_card.edit_text = AsyncMock()
        mock_status_card.delete = AsyncMock()
        mock_message.reply_video = AsyncMock()
        mock_update.effective_message = mock_message
        mock_context = MagicMock()

        await handle_facebook_download(mock_update, mock_context, f"https://www.facebook.com/reel/{video_id}")

        # Must reply video with cached file_id without downloading
        mock_message.reply_video.assert_awaited_once()
        self.assertEqual("instant_fid_888", mock_message.reply_video.call_args[1]["video"])

    async def test_handle_facebook_download_exceeds_50mb(self) -> None:
        mock_update = MagicMock()
        mock_message = MagicMock()
        mock_status_card = MagicMock()
        mock_message.reply_text = AsyncMock(return_value=mock_status_card)
        mock_status_card.edit_text = AsyncMock()
        mock_update.effective_message = mock_message
        mock_context = MagicMock()

        fake_fb_data = {
            "id": "vid_huge_1",
            "title": "2-Hour Football Match",
            "author_name": "Sports Live",
            "hd_url": "https://video.fbcdn.net/match_hd.mp4",
            "sd_url": "https://video.fbcdn.net/match_sd.mp4",
            "source_url": "https://www.facebook.com/watch/?v=vid_huge_1",
        }

        with patch("app.services.downloader.facebook.fetch_facebook_data", new_callable=AsyncMock) as mock_fetch, \
             patch("app.services.downloader.facebook.download_fb_media_to_file", new_callable=AsyncMock) as mock_dl:
            mock_fetch.return_value = fake_fb_data
            mock_dl.return_value = None  # None indicates file exceeds max_bytes limit

            await handle_facebook_download(mock_update, mock_context, "https://www.facebook.com/watch/?v=vid_huge_1")

            # Must display 50MB explanation with direct browser link
            mock_status_card.edit_text.assert_awaited()
            last_text = mock_status_card.edit_text.call_args[0][0]
            self.assertIn("ធំជាង 50MB", last_text)
            self.assertIn("Telegram Bot API", last_text)

    async def test_handle_facebook_mp3_download(self) -> None:
        mock_query = MagicMock()
        mock_query.answer = AsyncMock()
        mock_query.message.reply_audio = AsyncMock(return_value=MagicMock(audio=MagicMock(file_id="mp3_fid_555")))
        mock_context = MagicMock()

        cache_facebook("vid_audio_1", {
            "id": "vid_audio_1",
            "title": "Acoustic Live Session",
            "author_name": "Singer Name",
            "sd_url": "https://video.fbcdn.net/acoustic.mp4",
        })

        with patch("app.services.downloader.facebook.download_fb_media_to_file", new_callable=AsyncMock) as mock_dl:
            mock_dl.return_value = 500000

            await handle_facebook_mp3_download(mock_query, mock_context, "vid_audio_1")

            mock_query.message.reply_audio.assert_awaited_once()
            self.assertEqual("Acoustic Live Session", mock_query.message.reply_audio.call_args[1]["title"])

            # Verify cached mp3_file_id
            cached = get_cached_facebook("vid_audio_1")
            self.assertEqual("mp3_fid_555", cached["mp3_file_id"])

    async def test_handle_facebook_ai_summary(self) -> None:
        mock_query = MagicMock()
        mock_query.answer = AsyncMock()
        mock_query.message.reply_text = AsyncMock()
        mock_context = MagicMock()

        cache_facebook("vid_ai_1", {
            "id": "vid_ai_1",
            "title": "Tech Innovation Talk 2026",
            "author_name": "Tech KH",
        })

        with patch("app.services.ai.gemini.generate_content_with_fallback") as mock_gemini, \
             patch("app.services.ai.gemini.extract_gemini_text") as mock_extract:
            mock_gemini.return_value = "raw_res"
            mock_extract.return_value = "• ចំណុចទី១: បច្ចេកវិទ្យា AI ទំនើប\n• ចំណុចទី២: ការអភិវឌ្ឍជំនាញ\n• ចំណុចទី៣: ការអនុវត្តជាក់ស្ដែង"

            await handle_facebook_ai_summary(mock_query, mock_context, "vid_ai_1")

            mock_query.message.reply_text.assert_awaited_once()
            summary_msg = mock_query.message.reply_text.call_args[0][0]
            self.assertIn("សង្ខេបវីដេអូ Facebook ដោយ AI", summary_msg)
            self.assertIn("បច្ចេកវិទ្យា AI ទំនើប", summary_msg)

    async def test_handle_facebook_stats(self) -> None:
        mock_query = MagicMock()
        mock_query.answer = AsyncMock()
        mock_context = MagicMock()

        cache_facebook("vid_stat_1", {
            "id": "vid_stat_1",
            "title": "Siem Reap Travel Guide",
            "author_name": "Traveler KH",
            "hd_url": "https://video.fbcdn.net/hd.mp4",
            "sd_url": "https://video.fbcdn.net/sd.mp4",
            "video_file_id": "cached_stat_fid",
        })

        await handle_facebook_stats(mock_query, mock_context, "vid_stat_1")

        mock_query.answer.assert_awaited_once()
        stats_text = mock_query.answer.call_args[0][0]
        self.assertIn("ស្ថិតិវីដេអូ Facebook", stats_text)
        self.assertIn("Siem Reap Travel Guide", stats_text)
        self.assertIn("Traveler KH", stats_text)
        self.assertIn("Cached", stats_text)

    async def test_cmd_facebook_usage(self) -> None:
        mock_update = MagicMock()
        mock_message = MagicMock()
        mock_message.reply_text = AsyncMock()
        mock_update.effective_message = mock_message
        mock_context = MagicMock()

        # 1. Without args: display usage guide
        mock_context.args = []
        await cmd_facebook(mock_update, mock_context)
        mock_message.reply_text.assert_awaited_once()
        self.assertIn("Facebook Ultra-Downloader", mock_message.reply_text.call_args[0][0])

        # 2. With invalid URL
        mock_message.reply_text.reset_mock()
        mock_context.args = ["https://invalid-site.com/foo"]
        await cmd_facebook(mock_update, mock_context)
        self.assertIn("តំណភ្ជាប់មិនត្រឹមត្រូវ", mock_message.reply_text.call_args[0][0])

    async def test_facebook_callback_router(self) -> None:
        mock_update = MagicMock()
        mock_query = MagicMock()
        mock_update.callback_query = mock_query
        mock_context = MagicMock()

        with patch("app.services.downloader.facebook.handle_facebook_mp3_download", new_callable=AsyncMock) as mock_mp3, \
             patch("app.services.downloader.facebook.handle_facebook_file_download", new_callable=AsyncMock) as mock_file, \
             patch("app.services.downloader.facebook.handle_facebook_ai_summary", new_callable=AsyncMock) as mock_ai, \
             patch("app.services.downloader.facebook.handle_facebook_stats", new_callable=AsyncMock) as mock_stats:

            mock_query.data = "fb_mp3:12345"
            await facebook_callback(mock_update, mock_context)
            mock_mp3.assert_awaited_once_with(mock_query, mock_context, "12345")

            mock_query.data = "fb_file:22334"
            await facebook_callback(mock_update, mock_context)
            mock_file.assert_awaited_once_with(mock_query, mock_context, "22334")

            mock_query.data = "fb_ai:99887"
            await facebook_callback(mock_update, mock_context)
            mock_ai.assert_awaited_once_with(mock_query, mock_context, "99887")

            mock_query.data = "fb_stats:55443"
            await facebook_callback(mock_update, mock_context)
            mock_stats.assert_awaited_once_with(mock_query, mock_context, "55443")

    async def test_handle_facebook_file_download_hd_fallback_sd(self) -> None:
        mock_query = MagicMock()
        mock_query.answer = AsyncMock()
        mock_query.message.reply_document = AsyncMock()
        mock_context = MagicMock()

        cache_facebook("vid_file_1", {
            "id": "vid_file_1",
            "title": "HD Exceeds But SD Fits",
            "author_name": "Video Maker",
            "hd_url": "https://video.fbcdn.net/hd_huge.mp4",
            "sd_url": "https://video.fbcdn.net/sd_small.mp4",
        })

        with patch("app.services.downloader.facebook.download_fb_media_to_file", new_callable=AsyncMock) as mock_dl:
            # First call (HD) returns None (exceeds 50MB); second call (SD) succeeds
            mock_dl.side_effect = [None, 25 * 1024 * 1024]

            await handle_facebook_file_download(mock_query, mock_context, "vid_file_1")

            self.assertEqual(mock_dl.call_count, 2)
            mock_query.message.reply_document.assert_awaited_once()
            call_kwargs = mock_query.message.reply_document.call_args[1]
            self.assertIn("facebook_vid_file_1_sd.mp4", call_kwargs["filename"])

    async def test_handle_facebook_file_download_both_exceed_50mb(self) -> None:
        mock_query = MagicMock()
        mock_query.answer = AsyncMock()
        mock_query.message.reply_text = AsyncMock()
        mock_context = MagicMock()

        cache_facebook("vid_file_2", {
            "id": "vid_file_2",
            "title": "Massive Video File",
            "author_name": "Movie Studio",
            "hd_url": "https://video.fbcdn.net/hd_huge.mp4",
            "sd_url": "https://video.fbcdn.net/sd_huge.mp4",
        })

        with patch("app.services.downloader.facebook.download_fb_media_to_file", new_callable=AsyncMock) as mock_dl:
            mock_dl.return_value = None  # Both exceed 50MB

            await handle_facebook_file_download(mock_query, mock_context, "vid_file_2")

            mock_query.message.reply_text.assert_awaited_once()
            call_args = mock_query.message.reply_text.call_args
            self.assertIn("ធំជាង 50MB", call_args[0][0])
            self.assertIsNotNone(call_args[1].get("reply_markup"))

    def test_feature_toggle_facebook(self) -> None:
        self.assertTrue(is_facebook_enabled())
        set_feature_override(FEATURE_FACEBOOK, False)
        self.assertFalse(is_facebook_enabled())
        set_feature_override(FEATURE_FACEBOOK, True)
        self.assertTrue(is_facebook_enabled())


if __name__ == "__main__":
    unittest.main()
