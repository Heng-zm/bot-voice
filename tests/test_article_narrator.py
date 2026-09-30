"""Unit tests for the Web Link Article Narrator service."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock

from app.services.ai.article_reader import (
    extract_article_content,
    fetch_article_html,
    is_safe_public_url,
    summarize_article_with_ai,
)


class TestSafePublicUrlValidation(unittest.TestCase):
    """Test SSRF protections on input URLs."""

    def test_valid_public_urls(self) -> None:
        # Using domain that resolves to public IP
        safe, reason = is_safe_public_url("https://example.com/news/article")
        self.assertTrue(safe)
        self.assertEqual("", reason)

    def test_blocks_loopback_and_localhost(self) -> None:
        safe, reason = is_safe_public_url("http://127.0.0.1:8080/admin")
        self.assertFalse(safe)
        self.assertIn("forbidden", reason.lower())

        safe, reason = is_safe_public_url("http://localhost:3000/")
        self.assertFalse(safe)
        self.assertIn("forbidden", reason.lower())

    def test_blocks_cloud_metadata(self) -> None:
        safe, reason = is_safe_public_url("http://169.254.169.254/latest/meta-data/")
        self.assertFalse(safe)
        self.assertIn("forbidden", reason.lower())

    def test_blocks_ipv4_mapped_ipv6(self) -> None:
        safe, reason = is_safe_public_url("http://[::ffff:127.0.0.1]/admin")
        self.assertFalse(safe)
        self.assertIn("forbidden", reason.lower())

        safe, reason = is_safe_public_url("http://[::ffff:169.254.169.254]/latest/meta-data/")
        self.assertFalse(safe)
        self.assertIn("forbidden", reason.lower())

    def test_blocks_non_http_schemes(self) -> None:
        safe, reason = is_safe_public_url("ftp://example.com/file.txt")
        self.assertFalse(safe)
        self.assertIn("only http and https", reason.lower())

        safe, reason = is_safe_public_url("file:///etc/passwd")
        self.assertFalse(safe)
        self.assertIn("only http and https", reason.lower())

    def test_blocks_empty_url(self) -> None:
        safe, reason = is_safe_public_url("")
        self.assertFalse(safe)


class TestArticleHtmlExtraction(unittest.TestCase):
    """Test HTML parsing and clean article text extraction."""

    def test_extracts_title_and_clean_paragraphs(self) -> None:
        sample_html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Breaking News: Major Tech Discovery</title>
            <meta property="og:title" content="Breaking News: Major Tech Discovery" />
            <script>var x = 100; alert('spam');</script>
            <style>body { background: red; }</style>
        </head>
        <body>
            <nav><a href="/">Home</a> <a href="/news">News</a></nav>
            <header><h1>Header Banner</h1></header>
            <main>
                <article>
                    <h1>Breaking News: Major Tech Discovery</h1>
                    <p>Scientists and software engineers have announced a breakthrough in speech synthesis today.</p>
                    <p>The new technology allows computers to read complex multilingual documents with zero latency.</p>
                </article>
            </main>
            <footer>Copyright 2026 Example Corp</footer>
        </body>
        </html>
        """
        title, body = extract_article_content(sample_html)
        self.assertEqual("Breaking News: Major Tech Discovery", title)
        self.assertIn("Scientists and software engineers", body)
        self.assertIn("breakthrough in speech synthesis", body)
        self.assertNotIn("alert('spam')", body)
        self.assertNotIn("Copyright 2026", body)
        self.assertNotIn("Home", body)

    def test_og_description_fallback_when_body_sparse(self) -> None:
        sparse_html = """
        <html>
        <head>
            <meta property="og:title" content="Short Update" />
            <meta property="og:description" content="This is an informative summary from social meta tags." />
        </head>
        <body><div>Hi</div></body>
        </html>
        """
        title, body = extract_article_content(sparse_html)
        self.assertEqual("Short Update", title)
        self.assertIn("informative summary from social meta tags", body)


class TestArticleSummarization(unittest.TestCase):
    """Test AI summarization logic."""

    def test_fallback_excerpt_when_client_none(self) -> None:
        body = (
            "Paragraph one describing important background details.\n\n"
            "Paragraph two describing the latest developments today.\n\n"
            "Paragraph three with concluding remarks."
        )
        summary = summarize_article_with_ai("Sample Title", body, gemini_client=None)
        self.assertIn("Paragraph one", summary)
        self.assertIn("Paragraph two", summary)

    def test_ai_summarization_with_mock_client(self) -> None:
        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.text = "• Summary point 1\n• Summary point 2"
        mock_client.models.generate_content.return_value = mock_resp

        summary = summarize_article_with_ai("Test Title", "Long article body text", gemini_client=mock_client)
        self.assertIn("Summary point 1", summary)


class TestKhmerArticleTranslationAndSessions(unittest.TestCase):
    """Test dedicated Khmer translation, spoken bulletin preparation, and session storage."""

    def test_foreign_article_translated_to_khmer(self) -> None:
        from app.services.ai.article_reader import summarize_and_translate_for_khmer

        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.text = (
            "KHMER_TITLE: ការរកឃើញបច្ចេកវិទ្យា AI ថ្មីនៅកម្ពុជា\n"
            "KHMER_SUMMARY:\n"
            "• អ្នកស្រាវជ្រាវបានប្រកាសបច្ចេកវិទ្យាសំឡេង AI ថ្មី។\n"
            "• ប្រព័ន្ធនេះដំណើរការដោយផ្ទាល់ជាភាសាខ្មែរ។\n"
            "ORIGINAL_SUMMARY:\n"
            "Scientists have announced a new AI speech model that operates directly in Khmer."
        )
        mock_client.models.generate_content.return_value = mock_resp

        foreign_body = (
            "Artificial intelligence researchers have announced a new breakthrough in Phnom Penh. "
            "The new system enables real-time high-fidelity voice synthesis with low latency."
        )
        result = summarize_and_translate_for_khmer(
            title="Major AI Breakthrough Announced",
            body_text=foreign_body,
            gemini_client=mock_client,
        )

        self.assertEqual("en", result["orig_lang"])
        self.assertFalse(result["is_khmer"])
        self.assertEqual("ការរកឃើញបច្ចេកវិទ្យា AI ថ្មីនៅកម្ពុជា", result["khmer_title"])
        self.assertIn("អ្នកស្រាវជ្រាវបានប្រកាសបច្ចេកវិទ្យាសំឡេង AI ថ្មី", result["khmer_summary"])
        self.assertIn("Scientists have announced", result["original_summary"])
        self.assertIn("ការរកឃើញបច្ចេកវិទ្យា AI", result["khmer_tts_script"])

    def test_khmer_article_polished_spoken_summary(self) -> None:
        from app.services.ai.article_reader import summarize_and_translate_for_khmer

        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.text = (
            "KHMER_TITLE: ពិធីសម្ពោធស្ពានអាកាសថ្មី\n"
            "KHMER_SUMMARY:\n"
            "• ស្ពានអាកាសថ្មីត្រូវបានដាក់សម្ពោធឱ្យប្រើប្រាស់ជាផ្លូវការនៅថ្ងៃនេះ។"
        )
        mock_client.models.generate_content.return_value = mock_resp

        khmer_body = (
            "សម្តេចតេជោបានអញ្ជើញជាអធិបតីក្នុងពិធីសម្ពោធស្ពានអាកាសថ្មីក្នុងរាជធានីភ្នំពេញ "
            "ដើម្បីកាត់បន្ថយការកកស្ទះចរាចរណ៍សម្រាប់បងប្អូនប្រជាពលរដ្ឋ។"
        )
        result = summarize_and_translate_for_khmer(
            title="ពិធីសម្ពោធស្ពានអាកាស",
            body_text=khmer_body,
            gemini_client=mock_client,
        )

        self.assertEqual("km", result["orig_lang"])
        self.assertTrue(result["is_khmer"])
        self.assertEqual("ពិធីសម្ពោធស្ពានអាកាសថ្មី", result["khmer_title"])
        self.assertIn("ស្ពានអាកាសថ្មីត្រូវបានដាក់សម្ពោធ", result["khmer_summary"])

    def test_session_cache_lifecycle(self) -> None:
        from app.services.ai.article_reader import (
            get_article_session,
            make_article_session_id,
            store_article_session,
        )

        sid = make_article_session_id(12345, "https://example.com/khmer-news")
        self.assertTrue(len(sid) >= 8)

        data = {
            "title": "ព័ត៌មានជាតិ",
            "khmer_title": "ព័ត៌មានជាតិ",
            "khmer_summary": "សេចក្ដីសង្ខេប",
            "khmer_tts_script": "ព័ត៌មានជាតិ។ សេចក្ដីសង្ខេប",
            "body_text": "ខ្លឹមសារពេញលេញនៃអត្ថបទ",
        }
        store_article_session(sid, data)

        cached = get_article_session(sid)
        self.assertIsNotNone(cached)
        self.assertEqual("ព័ត៌មានជាតិ", cached["title"])
        self.assertEqual("ខ្លឹមសារពេញលេញនៃអត្ថបទ", cached["body_text"])

        # Non-existent session returns None
        self.assertIsNone(get_article_session("nonexistent_sid"))


class TestCambodianNewsParsing(unittest.TestCase):
    """Test HTML parsing with Cambodian news boilerplate filters."""

    def test_strips_boilerplate_containers_and_prefix_junk(self) -> None:
        from app.services.ai.article_reader import extract_article_content_with_image

        html_doc = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>រាជរដ្ឋាភិបាលប្រកាសគម្រោងថ្មី</title>
            <meta property="og:image" content="https://news.com.kh/images/cover.jpg">
        </head>
        <body>
            <div class="sidebar">ព័ត៌មានពេញនិយមប្រចាំថ្ងៃ</div>
            <div class="social-share">Share on Facebook Twitter Telegram</div>
            <div class="fn-news-related">អត្ថបទពាក់ព័ន្ធជាច្រើនទៀត</div>
            <article class="post-content">
                <p>រាជរដ្ឋាភិបាលកម្ពុជាបានប្រកាសដាក់ដំណើរការគម្រោងអភិវឌ្ឍន៍សេដ្ឋកិច្ចថ្មីនាព្រឹកនេះ។</p>
                <p>គម្រោងនេះនឹងបង្កើតការងាររាប់ម៉ឺនកន្លែងជូនប្រជាពលរដ្ឋនៅទូទាំងប្រទេស។</p>
                <p>ចុច Like ទំព័រហ្វេសប៊ុករបស់យើងដើម្បីទទួលបានព័ត៌មានថ្មីៗ</p>
                <p>Join Telegram Channel ផ្លូវការដើម្បីតាមដានព័ត៌មានទាន់ហេតុការណ៍</p>
            </article>
        </body>
        </html>
        """
        title, body, img = extract_article_content_with_image(html_doc)
        self.assertEqual("រាជរដ្ឋាភិបាលប្រកាសគម្រោងថ្មី", title)
        self.assertIn("គម្រោងអភិវឌ្ឍន៍សេដ្ឋកិច្ចថ្មី", body)
        self.assertIn("បង្កើតការងាររាប់ម៉ឺនកន្លែង", body)
        # Boilerplate should be filtered out
        self.assertNotIn("Share on Facebook", body)
        self.assertNotIn("ចុច Like ទំព័រ", body)
        self.assertNotIn("Join Telegram", body)
        self.assertNotIn("ព័ត៌មានពេញនិយម", body)
        self.assertEqual("https://news.com.kh/images/cover.jpg", img)


class TestFetchArticleHtmlRedirectValidation(unittest.IsolatedAsyncioTestCase):
    """Test SSRF redirect protection in fetch_article_html."""

    async def test_rejects_unsafe_initial_url(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            await fetch_article_html("http://127.0.0.1:8080/secret")
        self.assertIn("security validation error", str(ctx.exception).lower())

    async def test_rejects_redirect_to_private_ip(self) -> None:
        try:
            import httpx
            if isinstance(httpx, MagicMock) or not hasattr(httpx, "__file__"):
                self.skipTest("httpx is not real module in this environment")
        except ImportError:
            self.skipTest("httpx is not installed in this environment")

        from unittest.mock import AsyncMock, patch

        mock_redirect_resp = MagicMock()
        mock_redirect_resp.is_redirect = True
        mock_redirect_resp.headers = {"location": "http://169.254.169.254/latest/meta-data/"}

        with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_redirect_resp
            with self.assertRaises(ValueError) as ctx:
                await fetch_article_html("https://example.com/redirect")
            self.assertIn("security validation error on redirect", str(ctx.exception).lower())


class TestArticleCallbacks(unittest.IsolatedAsyncioTestCase):
    """Test interactive callbacks for article reader."""

    async def test_expired_session(self) -> None:
        from unittest.mock import AsyncMock
        from app.services.telegram.callbacks import article_callback

        update = MagicMock()
        update.callback_query.data = "art_voice:km_female:nonexistent"
        update.callback_query.from_user.id = 999
        update.callback_query.answer = AsyncMock()
        context = MagicMock()

        await article_callback(update, context)
        update.callback_query.answer.assert_called_once()
        self.assertIn("ផុតកំណត់", update.callback_query.answer.call_args[0][0])

    async def test_voice_switch_callback(self) -> None:
        from unittest.mock import AsyncMock, patch
        from app.services.ai.article_reader import store_article_session
        from app.services.telegram.callbacks import article_callback

        sid = "test_sid_123"
        store_article_session(sid, {
            "khmer_tts_script": "សេចក្ដីសង្ខេបជាសំឡេង",
            "khmer_summary": "សេចក្ដីសង្ខេប",
        })

        update = MagicMock()
        update.callback_query.data = f"art_voice:km_male:{sid}"
        update.callback_query.from_user.id = 1234
        update.callback_query.answer = AsyncMock()
        update.callback_query.message.reply_markup = MagicMock()
        context = MagicMock()

        with patch("app.services.telegram.media.process_tts_for_text", new_callable=AsyncMock) as mock_tts:
            await article_callback(update, context)
            mock_tts.assert_called_once()
            self.assertEqual("male", mock_tts.call_args[1]["gender_override"])
            self.assertEqual("សេចក្ដីសង្ខេបជាសំឡេង", mock_tts.call_args[0][2])

    async def test_orig_lang_callback(self) -> None:
        from unittest.mock import AsyncMock, patch
        from app.services.ai.article_reader import store_article_session
        from app.services.telegram.callbacks import article_callback

        sid = "test_sid_orig_456"
        store_article_session(sid, {
            "title": "US Tech Breakthrough",
            "original_title": "US Tech Breakthrough",
            "original_summary": "Scientists announced breakthrough.",
            "orig_lang_name": "English",
        })

        update = MagicMock()
        update.callback_query.data = f"art_orig:{sid}"
        update.callback_query.from_user.id = 1234
        update.callback_query.answer = AsyncMock()
        update.callback_query.message.reply_markup = MagicMock()
        context = MagicMock()

        with patch("app.services.telegram.media.process_tts_for_text", new_callable=AsyncMock) as mock_tts:
            await article_callback(update, context)
            mock_tts.assert_called_once()
            self.assertIn("Scientists announced breakthrough", mock_tts.call_args[0][2])

    async def test_full_text_callback(self) -> None:
        from unittest.mock import AsyncMock, patch
        from app.services.ai.article_reader import store_article_session
        from app.services.telegram.callbacks import article_callback

        sid = "test_sid_full_789"
        store_article_session(sid, {
            "title": "ព័ត៌មានជាតិពេញលេញ",
            "khmer_title": "ព័ត៌មានជាតិពេញលេញ",
            "body_text": "ខ្លឹមសារកថាខណ្ឌទីមួយ... ខ្លឹមសារកថាខណ្ឌទីពីរ...",
            "url": "https://news.com.kh/123",
        })

        update = MagicMock()
        update.callback_query.data = f"art_full:{sid}"
        update.callback_query.from_user.id = 1234
        update.callback_query.answer = AsyncMock()
        update.callback_query.message.reply_text = AsyncMock()
        context = MagicMock()

        with patch("app.services.telegram.formatters.send_split_html", new_callable=AsyncMock) as mock_send:
            await article_callback(update, context)
            mock_send.assert_called_once()
            self.assertIn("ព័ត៌មានជាតិពេញលេញ", mock_send.call_args[0][1])
            self.assertIn("ខ្លឹមសារកថាខណ្ឌទីមួយ", mock_send.call_args[0][1])

    async def test_gtts_voice_callback(self) -> None:
        from unittest.mock import AsyncMock, patch
        from app.services.ai.article_reader import store_article_session
        from app.services.telegram.callbacks import article_callback

        sid = "test_sid_gtts_999"
        store_article_session(sid, {
            "title": "ព័ត៌មាន gTTS",
            "khmer_title": "ព័ត៌មាន gTTS",
            "khmer_tts_script": "នេះជាសំឡេង gTTS",
            "khmer_summary": "នេះជាសំឡេង gTTS",
        })

        update = MagicMock()
        update.callback_query.data = f"art_voice:km_gtts:{sid}"
        update.callback_query.from_user.id = 1234
        update.callback_query.answer = AsyncMock()
        update.callback_query.message.reply_audio = AsyncMock()
        update.callback_query.message.reply_markup = MagicMock()
        context = MagicMock()

        with patch("app.services.ai.gtts_narrator.generate_khmer_gtts_audio_async", new_callable=AsyncMock) as mock_gtts:
            mock_gtts.return_value = b"\xff\xfb\x90\x44" + b"\x00" * 200
            await article_callback(update, context)
            mock_gtts.assert_called_once()
            update.callback_query.message.reply_audio.assert_called_once()
            self.assertIn("Google gTTS", update.callback_query.message.reply_audio.call_args[1]["caption"])


class TestArticleStorage(unittest.IsolatedAsyncioTestCase):
    """Test SQLite article idempotency storage."""

    async def test_idempotency_lifecycle(self) -> None:
        import uuid
        from app.services.ai.article_storage import (
            get_sent_article,
            is_article_sent,
            mark_article_sent,
        )

        test_hash = f"test_hash_{uuid.uuid4().hex[:12]}"
        # Initially not sent
        self.assertFalse(await is_article_sent(test_hash))
        self.assertIsNone(await get_sent_article(test_hash))

        # Mark sent
        await mark_article_sent(
            url_hash=test_hash,
            url="https://example.com/breaking-news",
            title="ព័ត៌មានទាន់ហេតុការណ៍",
            user_id=12345,
        )

        # Now is sent
        self.assertTrue(await is_article_sent(test_hash))
        record = await get_sent_article(test_hash)
        self.assertIsNotNone(record)
        self.assertEqual("https://example.com/breaking-news", record["url"])
        self.assertEqual("ព័ត៌មានទាន់ហេតុការណ៍", record["title"])
        self.assertEqual(12345, record["user_id"])


class TestArticleTranslator(unittest.TestCase):
    """Test Khmer translation engine and helpers."""

    def test_khmer_detection(self) -> None:
        from app.services.ai.article_translator import is_khmer

        self.assertTrue(is_khmer("សួស្តីកម្ពុជា"))
        self.assertTrue(is_khmer("News in Khmer: ព័ត៌មាន"))
        self.assertFalse(is_khmer("Breaking News: Tech Discovery in Phnom Penh"))
        self.assertFalse(is_khmer(""))

    def test_translate_text_cache(self) -> None:
        from unittest.mock import patch
        from app.services.ai.article_translator import translate_text

        # Khmer text is returned as-is without API call
        self.assertEqual("សួស្តី", translate_text("សួស្តី", target_lang="km"))

        # Foreign text translation with mocked chunk
        with patch("app.services.ai.article_translator.translate_chunk", return_value="បច្ចេកវិទ្យាថ្មី") as mock_chunk:
            res1 = translate_text("New Technology", target_lang="km")
            self.assertEqual("បច្ចេកវិទ្យាថ្មី", res1)
            # Second call hits cache
            res2 = translate_text("New Technology", target_lang="km")
            self.assertEqual("បច្ចេកវិទ្យាថ្មី", res2)
            mock_chunk.assert_called_once()


class TestGttsNarrator(unittest.TestCase):
    """Test gTTS Khmer voice narration."""

    def test_gtts_audio_generation(self) -> None:
        from app.services.ai.gtts_narrator import generate_khmer_gtts_audio

        # Empty returns empty
        self.assertEqual(b"", generate_khmer_gtts_audio(""))

        # Real synthesis with Khmer text
        audio = generate_khmer_gtts_audio("សួស្តី")
        self.assertTrue(len(audio) > 0)
        self.assertIsInstance(audio, bytes)


class TestTelegraphService(unittest.IsolatedAsyncioTestCase):
    """Test Telegra.ph Instant View generation."""

    def test_build_html_content(self) -> None:
        from app.services.ai.telegraph_service import _build_html_content

        html_out = _build_html_content(
            content_text="កថាខណ្ឌទីមួយ\nកថាខណ្ឌទីពីរ",
            image_url="https://news.com/img.jpg",
            source_url="https://news.com/article",
        )
        self.assertIn('<img src="https://news.com/img.jpg"/>', html_out)
        self.assertIn("<p>កថាខណ្ឌទីមួយ</p>", html_out)
        self.assertIn("<p>កថាខណ្ឌទីពីរ</p>", html_out)
        self.assertIn('href="https://news.com/article"', html_out)

    async def test_get_telegraph_url_with_mock(self) -> None:
        from unittest.mock import AsyncMock, patch
        from app.services.ai.telegraph_service import get_telegraph_url

        with patch("app.services.ai.telegraph_service._ensure_account", new_callable=AsyncMock) as mock_ensure:
            mock_ensure.return_value = True
            with patch("app.services.ai.telegraph_service._get_telegraph") as mock_tg:
                client = MagicMock()
                client.create_page.return_value = {"url": "https://telegra.ph/Khmer-News-10-01"}
                mock_tg.return_value = client

                url = await get_telegraph_url("ព័ត៌មានជាតិ", "ខ្លឹមសារ", "https://img.com/1.jpg", "https://news.com/1")
                self.assertEqual("https://telegra.ph/Khmer-News-10-01", url)


class TestSmartExtractorAndListing(unittest.TestCase):
    """Test smart extraction, listing page detection, and feed extraction."""

    def test_generate_article_hash(self) -> None:
        from app.services.ai.article_reader import generate_article_hash

        h1 = generate_article_hash("https://news.com/story/123?utm_source=tg#share")
        h2 = generate_article_hash("https://news.com/story/123")
        self.assertEqual(h1, h2)
        self.assertEqual(64, len(h1))

    def test_sanitize_html(self) -> None:
        from app.services.ai.article_reader import sanitize_html

        dirty = "Normal text\x00with null\x08and backspace"
        clean = sanitize_html(dirty)
        self.assertEqual("Normal textwith nulland backspace", clean)

    def test_is_listing_page_detection(self) -> None:
        from app.services.ai.article_reader import is_listing_page

        html_listing = """
        <html><body>
            <a href="/news/tech-breakthrough-2026">Major Technology Breakthrough in AI</a>
            <a href="/news/cambodia-economic-growth">Cambodia Economic Growth Surges</a>
            <a href="/about">About Us</a>
            <a href="/privacy">Privacy Policy</a>
            <a href="https://external.com/ad">External Ad Link</a>
        </body></html>
        """
        links = is_listing_page(html_listing, "https://example.com")
        self.assertEqual(2, len(links))
        self.assertTrue(any("tech-breakthrough" in l for l in links))
        self.assertTrue(any("cambodia-economic" in l for l in links))
        self.assertFalse(any("about" in l for l in links))
        self.assertFalse(any("privacy" in l for l in links))

    def test_extract_feed_links(self) -> None:
        from app.services.ai.article_reader import extract_feed_links

        html_feed = """
        <html><head>
            <link rel="alternate" type="application/rss+xml" href="/custom-rss.xml" />
        </head><body></body></html>
        """
        feeds = extract_feed_links(html_feed, "https://news.com")
        self.assertTrue(any("custom-rss.xml" in f for f in feeds))
        self.assertTrue(any("/rss" in f for f in feeds))


if __name__ == "__main__":
    unittest.main()
