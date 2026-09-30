"""Unit tests for Instagram Video, Reels & Posts Downloader service."""

from __future__ import annotations

import asyncio
from pathlib import Path
import sys
import tempfile
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
                pass
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
    FEATURE_INSTAGRAM,
    is_instagram_enabled,
    reset_feature_overrides,
    set_feature_override,
)
from app.services.downloader.instagram import (
    _IG_CACHE,
    _parse_html_instagram_data,
    cmd_instagram,
    download_ig_media_to_file,
    extract_instagram_shortcode,
    extract_instagram_url,
    fetch_instagram_video_info,
    get_instagram_video_kb,
    handle_instagram_ai_summary,
    handle_instagram_download,
    handle_instagram_file_download,
    handle_instagram_mp3_download,
    handle_instagram_stats,
    instagram_callback,
    is_instagram_url,
    resolve_instagram_redirect,
)


class InstagramDownloaderTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        reset_feature_overrides()
        _IG_CACHE.clear()

    def tearDown(self):
        reset_feature_overrides()
        _IG_CACHE.clear()

    def test_is_instagram_url(self):
        valid_urls = [
            "https://www.instagram.com/reel/C7xyz123/",
            "https://instagram.com/reel/C7xyz123",
            "https://m.instagram.com/reels/C7xyz123/",
            "https://www.instagram.com/p/C7xyz123/?utm_source=ig_web",
            "https://www.instagram.com/tv/C7xyz123/",
            "https://www.instagram.com/share/reel/C7xyz123/",
            "https://instagram.com/share/p/C7xyz123/",
        ]
        for u in valid_urls:
            self.assertTrue(is_instagram_url(u), f"Should match valid URL: {u}")

        invalid_urls = [
            "https://www.facebook.com/reel/12345",
            "https://www.tiktok.com/@user/video/12345",
            "https://youtube.com/watch?v=12345",
            "hello world",
            "",
            None,
        ]
        for u in invalid_urls:
            self.assertFalse(is_instagram_url(u), f"Should NOT match invalid URL: {u}")

    def test_extract_instagram_url(self):
        text = "Check this reel: https://www.instagram.com/reel/C7xyz123/?utm=test so cool!"
        extracted = extract_instagram_url(text)
        self.assertIsNotNone(extracted)
        self.assertIn("https://www.instagram.com/reel/C7xyz123/", extracted)

        self.assertIsNone(extract_instagram_url("No link here"))

    def test_extract_instagram_shortcode(self):
        self.assertEqual(extract_instagram_shortcode("https://www.instagram.com/reel/C7xyz123/"), "C7xyz123")
        self.assertEqual(extract_instagram_shortcode("https://www.instagram.com/p/Post999/"), "Post999")
        self.assertEqual(extract_instagram_shortcode("https://www.instagram.com/share/reel/Share456/"), "Share456")

    async def test_resolve_instagram_redirect(self):
        # Non-share link returns immediately
        url = "https://www.instagram.com/reel/C7xyz123/"
        resolved = await resolve_instagram_redirect(url)
        self.assertEqual(resolved, url)

    def test_parse_html_instagram_data_og_video(self):
        html_text = """
        <html>
            <head>
                <meta property="og:video" content="https://scontent.cdninstagram.com/v/t50.2886-16/test.mp4" />
                <meta property="og:title" content="Awesome Sunset on Instagram" />
                <meta property="og:image" content="https://scontent.cdninstagram.com/thumb.jpg" />
            </head>
        </html>
        """
        data = _parse_html_instagram_data(html_text, "https://www.instagram.com/reel/test/")
        self.assertIsNotNone(data)
        self.assertEqual(data["video_url"], "https://scontent.cdninstagram.com/v/t50.2886-16/test.mp4")
        self.assertIn("Awesome Sunset", data["title"])

    def test_parse_html_instagram_data_json(self):
        html_text = """
        <script>
            window.__additionalData = {"video_url": "https:\\/\\/video.fbcdn.net\\/v\\/stream.mp4"};
        </script>
        <meta property="og:title" content="John on Instagram: &quot;Summer vibes&quot;" />
        """
        data = _parse_html_instagram_data(html_text, "https://www.instagram.com/reel/test/")
        self.assertIsNotNone(data)
        self.assertEqual(data["video_url"], "https://video.fbcdn.net/v/stream.mp4")
        self.assertEqual(data["author"], "John")

    def test_parse_html_instagram_data_no_video(self):
        html_text = "<html><body>No video here</body></html>"
        data = _parse_html_instagram_data(html_text, "https://www.instagram.com/reel/test/")
        self.assertIsNone(data)

    async def test_download_ig_media_to_file_success(self):
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp_path = tmp.name

        mock_resp = MagicMock()
        mock_resp.headers = {"Content-Length": "100"}
        mock_resp.read.side_effect = [b"A" * 50, b"B" * 50, b""]
        mock_resp.__enter__.return_value = mock_resp
        mock_resp.__exit__.return_value = None

        with patch("urllib.request.urlopen", return_value=mock_resp):
            bytes_written = await download_ig_media_to_file("https://example.com/video.mp4", tmp_path)
            self.assertEqual(bytes_written, 100)

        with open(tmp_path, "rb") as f:
            content = f.read()
        self.assertEqual(len(content), 100)
        Path(tmp_path).unlink(missing_ok=True)

    async def test_download_ig_media_to_file_max_bytes_abort(self):
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp_path = tmp.name

        mock_resp = MagicMock()
        mock_resp.headers = {"Content-Length": "1000"}
        mock_resp.read.side_effect = [b"A" * 600, b"B" * 600]
        mock_resp.__enter__.return_value = mock_resp
        mock_resp.__exit__.return_value = None

        with patch("urllib.request.urlopen", return_value=mock_resp):
            with self.assertRaises(ValueError):
                await download_ig_media_to_file("https://example.com/video.mp4", tmp_path, max_bytes=500)

        Path(tmp_path).unlink(missing_ok=True)

    def test_get_instagram_video_kb(self):
        kb = get_instagram_video_kb("short123")
        self.assertEqual(len(kb.inline_keyboard), 3)
        self.assertEqual(kb.inline_keyboard[0][0].callback_data, "ig_video:short123")
        self.assertEqual(kb.inline_keyboard[0][1].callback_data, "ig_audio:short123")
        self.assertEqual(kb.inline_keyboard[1][0].callback_data, "ig_doc:short123")
        self.assertEqual(kb.inline_keyboard[1][1].callback_data, "ig_ai:short123")
        self.assertEqual(kb.inline_keyboard[2][0].callback_data, "ig_stats:short123")

    async def test_handle_instagram_download_feature_disabled(self):
        set_feature_override(FEATURE_INSTAGRAM, False)
        self.assertFalse(is_instagram_enabled())

        update = MagicMock()
        msg = AsyncMock()
        update.effective_message = msg

        await handle_instagram_download(update, MagicMock(), "https://www.instagram.com/reel/test/")
        msg.reply_text.assert_called_once()
        self.assertIn("ត្រូវបានបិទជាបណ្ដោះអាសន្ន", msg.reply_text.call_args[0][0])

    async def test_handle_instagram_download_no_url(self):
        update = MagicMock()
        msg = AsyncMock()
        update.effective_message = msg

        await handle_instagram_download(update, MagicMock(), "invalid text")
        msg.reply_text.assert_called_once()
        self.assertIn("រកមិនឃើញតំណភ្ជាប់ Instagram", msg.reply_text.call_args[0][0])

    async def test_handle_instagram_download_success(self):
        update = MagicMock()
        msg = AsyncMock()
        status_msg = AsyncMock()
        msg.reply_text.return_value = status_msg
        update.effective_message = msg

        mock_info = {
            "shortcode": "C7test",
            "video_url": "https://scontent.cdninstagram.com/test.mp4",
            "title": "Test Reel",
            "author": "Test Author",
        }

        with patch("app.services.downloader.instagram.fetch_instagram_video_info", new_callable=AsyncMock) as mock_fetch:
            mock_fetch.return_value = mock_info
            with patch("app.services.downloader.instagram.download_ig_media_to_file", new_callable=AsyncMock) as mock_down:
                mock_down.return_value = 1024

                async def fake_stream(url, dest, **kwargs):
                    with open(dest, "wb") as f:
                        f.write(b"video data" * 100)
                    return 1000

                mock_down.side_effect = fake_stream

                await handle_instagram_download(update, MagicMock(), "https://www.instagram.com/reel/C7test/")
                msg.reply_video.assert_called_once()
                self.assertIn("Instagram Video", msg.reply_video.call_args[1]["caption"])
                status_msg.delete.assert_called_once()

    async def test_handle_instagram_download_over_50mb(self):
        update = MagicMock()
        msg = AsyncMock()
        status_msg = AsyncMock()
        msg.reply_text.return_value = status_msg
        update.effective_message = msg

        mock_info = {
            "shortcode": "C7large",
            "video_url": "https://scontent.cdninstagram.com/large.mp4",
            "title": "Large Reel",
            "author": "Creator",
        }

        with patch("app.services.downloader.instagram.fetch_instagram_video_info", new_callable=AsyncMock) as mock_fetch:
            mock_fetch.return_value = mock_info
            with patch("app.services.downloader.instagram.download_ig_media_to_file", new_callable=AsyncMock) as mock_down:
                mock_down.side_effect = ValueError("Exceeded 50MB")

                await handle_instagram_download(update, MagicMock(), "https://www.instagram.com/reel/C7large/")
                self.assertTrue(status_msg.edit_text.called)
                last_call_text = status_msg.edit_text.call_args[0][0]
                self.assertIn("50MB", last_call_text)

    async def test_handle_instagram_mp3_download(self):
        query = MagicMock()
        query.message = AsyncMock()
        query.answer = AsyncMock()

        _IG_CACHE["mp3test"] = {
            "video_url": "https://scontent.cdninstagram.com/test.mp4",
            "title": "Song Reel",
            "author": "Singer",
        }

        with patch("app.services.downloader.instagram.download_ig_media_to_file", new_callable=AsyncMock):
            with patch("asyncio.create_subprocess_exec") as mock_proc:
                mock_sub = AsyncMock()
                mock_sub.communicate.return_value = (b"", b"")

                async def fake_ffmpeg(*args, **kwargs):
                    # Write fake mp3 to the output file argument
                    out_path = args[9]
                    with open(out_path, "wb") as f:
                        f.write(b"fake mp3 audio")
                    return mock_sub

                mock_proc.side_effect = fake_ffmpeg

                await handle_instagram_mp3_download(query, MagicMock(), "mp3test")
                query.message.reply_audio.assert_called_once()
                self.assertIn("Song Reel", query.message.reply_audio.call_args[1]["title"])

    async def test_handle_instagram_file_download(self):
        query = MagicMock()
        query.message = AsyncMock()
        query.answer = AsyncMock()

        _IG_CACHE["doctest"] = {
            "video_url": "https://scontent.cdninstagram.com/test.mp4",
            "author": "Creator",
        }

        with patch("app.services.downloader.instagram.download_ig_media_to_file", new_callable=AsyncMock) as mock_down:
            async def fake_write(url, dest, **kwargs):
                with open(dest, "wb") as f:
                    f.write(b"mp4 binary data")
                return 15

            mock_down.side_effect = fake_write

            await handle_instagram_file_download(query, MagicMock(), "doctest")
            query.message.reply_document.assert_called_once()
            self.assertEqual(query.message.reply_document.call_args[1]["filename"], "instagram_doctest.mp4")

    async def test_handle_instagram_ai_summary(self):
        query = MagicMock()
        query.message = AsyncMock()
        query.answer = AsyncMock()

        _IG_CACHE["aitest"] = {
            "title": "Tutorial on Cooking",
            "author": "Chef",
        }

        with patch("app.services.ai.gemini.generate_content_with_fallback", new_callable=AsyncMock) as mock_gemini:
            mock_gemini.return_value = "1. គ្រឿងផ្សំ\n2. វិធីធ្វើ"
            with patch("app.services.ai.gemini.extract_gemini_text", return_value="1. គ្រឿងផ្សំ\n2. វិធីធ្វើ"):
                await handle_instagram_ai_summary(query, MagicMock(), "aitest")
                query.message.reply_text.assert_called_once()
                self.assertIn("AI Summary", query.message.reply_text.call_args[0][0])
                self.assertIn("វិធីធ្វើ", query.message.reply_text.call_args[0][0])

    async def test_cmd_instagram(self):
        update = MagicMock()
        msg = AsyncMock()
        msg.text = "/ig"
        update.effective_message = msg

        await cmd_instagram(update, MagicMock())
        msg.reply_text.assert_called_once()
        self.assertIn("Instagram", msg.reply_text.call_args[0][0])

    async def test_instagram_callback(self):
        update = MagicMock()
        query = AsyncMock()
        update.callback_query = query

        query.data = "ig_stats:test_id"
        _IG_CACHE["test_id"] = {"author": "Author", "title": "Title"}

        await instagram_callback(update, MagicMock())
        query.answer.assert_called_once()
        self.assertIn("ស្ថិតិវីដេអូ Instagram", query.answer.call_args[0][0])


if __name__ == "__main__":
    unittest.main()
