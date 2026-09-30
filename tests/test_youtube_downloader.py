"""Unit tests for YouTube Video & Shorts Downloader service."""

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
    FEATURE_YOUTUBE,
    is_youtube_enabled,
    reset_feature_overrides,
    set_feature_override,
)
from app.services.downloader.youtube import (
    _YT_CACHE,
    cmd_youtube,
    download_yt_media_to_file,
    extract_youtube_id,
    extract_youtube_url,
    fetch_youtube_oembed_metadata,
    fetch_youtube_video_info,
    get_youtube_video_kb,
    handle_youtube_ai_summary,
    handle_youtube_download,
    handle_youtube_file_download,
    handle_youtube_mp3_download,
    handle_youtube_stats,
    is_youtube_url,
    youtube_callback,
)


class YouTubeDownloaderTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        reset_feature_overrides()
        _YT_CACHE.clear()

    def tearDown(self):
        reset_feature_overrides()
        _YT_CACHE.clear()

    def test_is_youtube_url(self):
        valid_urls = [
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            "https://youtube.com/watch?v=dQw4w9WgXcQ",
            "https://m.youtube.com/watch?v=dQw4w9WgXcQ",
            "https://youtu.be/dQw4w9WgXcQ",
            "https://www.youtube.com/shorts/dQw4w9WgXcQ",
            "https://youtube.com/shorts/dQw4w9WgXcQ?feature=share",
            "https://www.youtube.com/embed/dQw4w9WgXcQ",
        ]
        for u in valid_urls:
            self.assertTrue(is_youtube_url(u), f"Should match valid URL: {u}")

        invalid_urls = [
            "https://www.facebook.com/reel/12345",
            "https://www.tiktok.com/@user/video/12345",
            "https://instagram.com/reel/C7test",
            "hello world",
            "",
            None,
        ]
        for u in invalid_urls:
            self.assertFalse(is_youtube_url(u), f"Should NOT match invalid URL: {u}")

    def test_extract_youtube_url(self):
        text = "Listen to this: https://www.youtube.com/watch?v=dQw4w9WgXcQ it's legendary"
        extracted = extract_youtube_url(text)
        self.assertIsNotNone(extracted)
        self.assertIn("https://www.youtube.com/watch?v=dQw4w9WgXcQ", extracted)

        self.assertIsNone(extract_youtube_url("No link here"))

    def test_extract_youtube_id(self):
        self.assertEqual(extract_youtube_id("https://www.youtube.com/watch?v=dQw4w9WgXcQ"), "dQw4w9WgXcQ")
        self.assertEqual(extract_youtube_id("https://youtu.be/dQw4w9WgXcQ"), "dQw4w9WgXcQ")
        self.assertEqual(extract_youtube_id("https://www.youtube.com/shorts/dQw4w9WgXcQ"), "dQw4w9WgXcQ")
        self.assertEqual(extract_youtube_id("https://www.youtube.com/embed/dQw4w9WgXcQ"), "dQw4w9WgXcQ")

    async def test_fetch_youtube_oembed_metadata(self):
        mock_resp = MagicMock()
        mock_resp.read.return_value = b'{"title": "Never Gonna Give You Up", "author_name": "Rick Astley", "thumbnail_url": "https://i.ytimg.com/thumb.jpg"}'
        mock_resp.__enter__.return_value = mock_resp
        mock_resp.__exit__.return_value = None

        with patch("urllib.request.urlopen", return_value=mock_resp):
            meta = await fetch_youtube_oembed_metadata("dQw4w9WgXcQ")
            self.assertEqual(meta["title"], "Never Gonna Give You Up")
            self.assertEqual(meta["author"], "Rick Astley")
            self.assertEqual(meta["thumbnail"], "https://i.ytimg.com/thumb.jpg")

    async def test_fetch_youtube_video_info(self):
        with patch("app.services.downloader.youtube.fetch_youtube_oembed_metadata", new_callable=AsyncMock) as mock_oembed:
            mock_oembed.return_value = {
                "title": "Rick Roll",
                "author": "Rick Astley",
                "thumbnail": "https://i.ytimg.com/thumb.jpg",
            }
            info = await fetch_youtube_video_info("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
            self.assertIsNotNone(info)
            self.assertEqual(info["video_id"], "dQw4w9WgXcQ")
            self.assertEqual(info["title"], "Rick Roll")
            self.assertEqual(info["author"], "Rick Astley")

    async def test_download_yt_media_to_file_success(self):
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp_path = tmp.name

        mock_resp = MagicMock()
        mock_resp.headers = {"Content-Length": "100"}
        mock_resp.read.side_effect = [b"X" * 50, b"Y" * 50, b""]
        mock_resp.__enter__.return_value = mock_resp
        mock_resp.__exit__.return_value = None

        with patch("urllib.request.urlopen", return_value=mock_resp):
            bytes_written = await download_yt_media_to_file("https://example.com/yt.mp4", tmp_path)
            self.assertEqual(bytes_written, 100)

        with open(tmp_path, "rb") as f:
            content = f.read()
        self.assertEqual(len(content), 100)
        Path(tmp_path).unlink(missing_ok=True)

    async def test_download_yt_media_to_file_max_bytes_abort(self):
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp_path = tmp.name

        mock_resp = MagicMock()
        mock_resp.headers = {"Content-Length": "1000"}
        mock_resp.read.side_effect = [b"X" * 600, b"Y" * 600]
        mock_resp.__enter__.return_value = mock_resp
        mock_resp.__exit__.return_value = None

        with patch("urllib.request.urlopen", return_value=mock_resp):
            with self.assertRaises(ValueError):
                await download_yt_media_to_file("https://example.com/yt.mp4", tmp_path, max_bytes=500)

        Path(tmp_path).unlink(missing_ok=True)

    def test_get_youtube_video_kb(self):
        kb = get_youtube_video_kb("dQw4w9WgXcQ")
        self.assertEqual(len(kb.inline_keyboard), 3)
        self.assertEqual(kb.inline_keyboard[0][0].callback_data, "yt_video:dQw4w9WgXcQ")
        self.assertEqual(kb.inline_keyboard[0][1].callback_data, "yt_audio:dQw4w9WgXcQ")
        self.assertEqual(kb.inline_keyboard[1][0].callback_data, "yt_doc:dQw4w9WgXcQ")
        self.assertEqual(kb.inline_keyboard[1][1].callback_data, "yt_ai:dQw4w9WgXcQ")
        self.assertEqual(kb.inline_keyboard[2][0].callback_data, "yt_stats:dQw4w9WgXcQ")

    async def test_handle_youtube_download_feature_disabled(self):
        set_feature_override(FEATURE_YOUTUBE, False)
        self.assertFalse(is_youtube_enabled())

        update = MagicMock()
        msg = AsyncMock()
        update.effective_message = msg

        await handle_youtube_download(update, MagicMock(), "https://www.youtube.com/watch?v=dQw4w9WgXcQ")
        msg.reply_text.assert_called_once()
        self.assertIn("ត្រូវបានបិទជាបណ្ដោះអាសន្ន", msg.reply_text.call_args[0][0])

    async def test_handle_youtube_download_no_url(self):
        update = MagicMock()
        msg = AsyncMock()
        update.effective_message = msg

        await handle_youtube_download(update, MagicMock(), "invalid text")
        msg.reply_text.assert_called_once()
        self.assertIn("រកមិនឃើញតំណភ្ជាប់ YouTube", msg.reply_text.call_args[0][0])

    async def test_handle_youtube_download_direct_stream(self):
        update = MagicMock()
        msg = AsyncMock()
        status_msg = AsyncMock()
        msg.reply_text.return_value = status_msg
        update.effective_message = msg

        mock_info = {
            "video_id": "dQw4w9WgXcQ",
            "video_url": "https://stream.googlevideo.com/videoplayback?id=123",
            "title": "Rick Roll Video",
            "author": "Rick Astley",
        }

        with patch("app.services.downloader.youtube.fetch_youtube_video_info", new_callable=AsyncMock) as mock_fetch:
            mock_fetch.return_value = mock_info
            with patch("app.services.downloader.youtube.download_yt_media_to_file", new_callable=AsyncMock) as mock_down:
                async def fake_stream(url, dest, **kwargs):
                    with open(dest, "wb") as f:
                        f.write(b"yt video data" * 100)
                    return 1300

                mock_down.side_effect = fake_stream

                await handle_youtube_download(update, MagicMock(), "https://www.youtube.com/watch?v=dQw4w9WgXcQ")
                msg.reply_video.assert_called_once()
                self.assertIn("YouTube Video", msg.reply_video.call_args[1]["caption"])
                status_msg.delete.assert_called_once()

    async def test_handle_youtube_download_web_watch_fallback(self):
        update = MagicMock()
        msg = AsyncMock()
        status_msg = AsyncMock()
        msg.reply_text.return_value = status_msg
        update.effective_message = msg

        mock_info = {
            "video_id": "dQw4w9WgXcQ",
            "video_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            "title": "Full Concert",
            "author": "Artist",
        }

        with patch("app.services.downloader.youtube.fetch_youtube_video_info", new_callable=AsyncMock) as mock_fetch:
            mock_fetch.return_value = mock_info

            await handle_youtube_download(update, MagicMock(), "https://www.youtube.com/watch?v=dQw4w9WgXcQ")
            self.assertTrue(status_msg.edit_text.called)
            self.assertIn("YouTube Video & Shorts", status_msg.edit_text.call_args[0][0])

    async def test_handle_youtube_mp3_download(self):
        query = MagicMock()
        query.message = AsyncMock()
        query.answer = AsyncMock()

        _YT_CACHE["ytmp3"] = {
            "video_url": "https://stream.googlevideo.com/playback.mp4",
            "title": "Song Title",
            "author": "Band",
        }

        with patch("app.services.downloader.youtube.download_yt_media_to_file", new_callable=AsyncMock):
            with patch("asyncio.create_subprocess_exec") as mock_proc:
                mock_sub = AsyncMock()
                mock_sub.communicate.return_value = (b"", b"")

                async def fake_ffmpeg(*args, **kwargs):
                    out_path = args[9]
                    with open(out_path, "wb") as f:
                        f.write(b"fake mp3 audio")
                    return mock_sub

                mock_proc.side_effect = fake_ffmpeg

                await handle_youtube_mp3_download(query, MagicMock(), "ytmp3")
                query.message.reply_audio.assert_called_once()
                self.assertIn("Song Title", query.message.reply_audio.call_args[1]["title"])

    async def test_handle_youtube_file_download(self):
        query = MagicMock()
        query.message = AsyncMock()
        query.answer = AsyncMock()

        _YT_CACHE["ytdoc"] = {
            "video_url": "https://stream.googlevideo.com/playback.mp4",
            "author": "Creator",
        }

        with patch("app.services.downloader.youtube.download_yt_media_to_file", new_callable=AsyncMock) as mock_down:
            async def fake_write(url, dest, **kwargs):
                with open(dest, "wb") as f:
                    f.write(b"yt mp4 binary data")
                return 18

            mock_down.side_effect = fake_write

            await handle_youtube_file_download(query, MagicMock(), "ytdoc")
            query.message.reply_document.assert_called_once()
            self.assertEqual(query.message.reply_document.call_args[1]["filename"], "youtube_ytdoc.mp4")

    async def test_handle_youtube_ai_summary(self):
        query = MagicMock()
        query.message = AsyncMock()
        query.answer = AsyncMock()

        _YT_CACHE["ytai"] = {
            "title": "How to Learn Python",
            "author": "Code Academy",
        }

        with patch("app.services.ai.gemini.generate_content_with_fallback", new_callable=AsyncMock) as mock_gemini:
            mock_gemini.return_value = "1. មូលដ្ឋានគ្រឹះ\n2. លំហាត់ជាក់ស្តែង"
            with patch("app.services.ai.gemini.extract_gemini_text", return_value="1. មូលដ្ឋានគ្រឹះ\n2. លំហាត់ជាក់ស្តែង"):
                await handle_youtube_ai_summary(query, MagicMock(), "ytai")
                query.message.reply_text.assert_called_once()
                self.assertIn("AI Summary", query.message.reply_text.call_args[0][0])
                self.assertIn("មូលដ្ឋានគ្រឹះ", query.message.reply_text.call_args[0][0])

    async def test_cmd_youtube(self):
        update = MagicMock()
        msg = AsyncMock()
        msg.text = "/yt"
        update.effective_message = msg

        await cmd_youtube(update, MagicMock())
        msg.reply_text.assert_called_once()
        self.assertIn("YouTube", msg.reply_text.call_args[0][0])

    async def test_youtube_callback(self):
        update = MagicMock()
        query = AsyncMock()
        update.callback_query = query

        query.data = "yt_stats:test_yt_id"
        _YT_CACHE["test_yt_id"] = {"author": "Channel", "title": "Great Video"}

        await youtube_callback(update, MagicMock())
        query.answer.assert_called_once()
        self.assertIn("ស្ថិតិវីដេអូ YouTube", query.answer.call_args[0][0])


if __name__ == "__main__":
    unittest.main()
