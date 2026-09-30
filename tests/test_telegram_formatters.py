"""Unit tests for Telegram HTML formatters, Markdown conversion, tag balancing, and speech cleaning."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

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
        from unittest.mock import MagicMock
        sys.modules["httpx"] = MagicMock()

if "fastapi" not in sys.modules:
    try:
        import fastapi  # noqa: F401
    except ImportError:
        import types
        fastapi_mod = types.ModuleType("fastapi")
        fastapi_responses = types.ModuleType("fastapi.responses")
        fastapi_responses.JSONResponse = type("JSONResponse", (), {"media_type": "application/json"})
        sys.modules["fastapi"] = fastapi_mod
        sys.modules["fastapi.responses"] = fastapi_responses

from app.services.telegram.formatters import (
    balance_telegram_html,
    clean_speech_text,
    escape_html,
    markdown_to_telegram_html,
    split_telegram_html,
    strip_html_tags,
)


class TelegramFormattersTests(unittest.TestCase):
    def test_escape_html(self) -> None:
        raw = '<b>Hello & "World" <test></b>'
        escaped = escape_html(raw)
        self.assertEqual(escaped, '&lt;b&gt;Hello &amp; "World" &lt;test&gt;&lt;/b&gt;')

    def test_strip_html_tags(self) -> None:
        html_input = "<b>Bold</b> and <i>Italic</i> with <a href='https://example.com'>Link</a>"
        self.assertEqual(strip_html_tags(html_input), "Bold and Italic with Link")

    def test_markdown_bold_and_italic(self) -> None:
        md = "This is **bold** and *italic* and ***bold-italic***."
        res = markdown_to_telegram_html(md)
        self.assertIn("<b>bold</b>", res)
        self.assertIn("<i>italic</i>", res)
        self.assertIn("<b><i>bold-italic</i></b>", res)

    def test_markdown_strikethrough_and_spoiler(self) -> None:
        md = "~~wrong~~ and ||secret spoiler||"
        res = markdown_to_telegram_html(md)
        self.assertIn("<s>wrong</s>", res)
        self.assertIn("<tg-spoiler>secret spoiler</tg-spoiler>", res)

    def test_markdown_headers(self) -> None:
        md = "# Heading 1\n## Heading 2\n### Heading 3"
        res = markdown_to_telegram_html(md)
        self.assertIn("<b>Heading 1</b>", res)
        self.assertIn("<b>Heading 2</b>", res)
        self.assertIn("<b>Heading 3</b>", res)

    def test_markdown_code_block_escaping(self) -> None:
        bt = chr(96)
        md = f"{bt*3}python\nx = 10 < 20 and y > 5\n{bt*3}"
        res = markdown_to_telegram_html(md)
        self.assertIn('<pre><code class="language-python">', res)
        self.assertIn("x = 10 &lt; 20 and y &gt; 5", res)
        self.assertIn("</code></pre>", res)

    def test_markdown_inline_code_escaping(self) -> None:
        bt = chr(96)
        md = f"Run {bt}<script>alert(1)</script>{bt} command"
        res = markdown_to_telegram_html(md)
        self.assertIn("<code>&lt;script&gt;alert(1)&lt;/script&gt;</code>", res)

    def test_markdown_links(self) -> None:
        md = "Check [Bot Voice](https://t.me/voicekhaibot) for updates."
        res = markdown_to_telegram_html(md)
        self.assertIn('<a href="https://t.me/voicekhaibot">Bot Voice</a>', res)

    def test_markdown_bullet_list(self) -> None:
        md = "* Item 1\n* Item 2\n- Item 3"
        res = markdown_to_telegram_html(md)
        self.assertIn("• Item 1", res)
        self.assertIn("• Item 2", res)
        self.assertIn("• Item 3", res)

    def test_markdown_blockquotes(self) -> None:
        md = "> Line 1 of quote\n> Line 2 of quote\nNormal text"
        res = markdown_to_telegram_html(md)
        self.assertIn("<blockquote>Line 1 of quote\nLine 2 of quote</blockquote>", res)
        self.assertIn("Normal text", res)

    def test_tag_balancing_unclosed_tags(self) -> None:
        unclosed = "<b>Bold start <i>italic start"
        balanced = balance_telegram_html(unclosed)
        self.assertEqual(balanced, "<b>Bold start <i>italic start</i></b>")

    def test_tag_balancing_disallowed_tags(self) -> None:
        disallowed = "<script>alert(1)</script><b>Valid</b>"
        balanced = balance_telegram_html(disallowed)
        self.assertNotIn("<script>", balanced)
        self.assertIn("<b>Valid</b>", balanced)

    def test_clean_speech_text(self) -> None:
        bt = chr(96)
        md = (
            f"# Title\n**Hello** [Channel](https://t.me/test)\n"
            f"{bt*3}python\nprint('ignore')\n{bt*3}\n"
            f"Read this {bt}code{bt} out loud!"
        )
        speech = clean_speech_text(md)
        self.assertNotIn("#", speech)
        self.assertNotIn("**", speech)
        self.assertNotIn("https://", speech)
        self.assertNotIn("print('ignore')", speech)
        self.assertIn("Title", speech)
        self.assertIn("Hello Channel", speech)
        self.assertIn("Read this out loud!", speech)

    def test_split_telegram_html(self) -> None:
        long_text = "<b>Paragraph 1</b>\n\n<i>Paragraph 2</i>\n\n<code>Paragraph 3</code>"
        chunks = split_telegram_html(long_text, max_length=30)
        self.assertTrue(len(chunks) > 1)
        for c in chunks:
            # Each chunk must have balanced tags
            self.assertEqual(c.count("<b>"), c.count("</b>"))
            self.assertEqual(c.count("<i>"), c.count("</i>"))
            self.assertEqual(c.count("<code>"), c.count("</code>"))

    def test_split_telegram_html_large_continuous_text(self) -> None:
        # Create a single huge line > 10,000 chars
        words = ["ព័ត៌មានជាតិនិងអន្តរជាតិ", "បច្ចេកវិទ្យា", "បញ្ញាសិប្បនិម្មិត", "កម្ពុជា", "សេដ្ឋកិច្ច"] * 250
        huge_text = " ".join(words)
        self.assertGreater(len(huge_text), 8000)

        chunks = split_telegram_html(huge_text, max_length=3800)
        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertLessEqual(len(chunk), 4000)

    def test_send_split_html_on_message(self) -> None:
        import asyncio
        from unittest.mock import AsyncMock, MagicMock
        from app.services.telegram.formatters import send_split_html

        mock_msg = MagicMock()
        mock_msg.reply_text = AsyncMock(return_value="msg_sent")

        text = "Hello world! " * 400  # ~5200 chars
        markup = MagicMock()

        res = asyncio.run(send_split_html(mock_msg, text, reply_markup=markup, max_length=3000))
        self.assertGreater(len(res), 1)
        self.assertEqual(mock_msg.reply_text.call_count, len(res))

        # Check that markup is only on the final chunk
        first_call_kwargs = mock_msg.reply_text.call_args_list[0][1]
        last_call_kwargs = mock_msg.reply_text.call_args_list[-1][1]
        self.assertIsNone(first_call_kwargs.get("reply_markup"))
        self.assertEqual(last_call_kwargs.get("reply_markup"), markup)

    def test_send_split_html_on_callback_query(self) -> None:
        import asyncio
        from unittest.mock import AsyncMock, MagicMock
        from app.services.telegram.formatters import send_split_html

        mock_query = MagicMock(spec=["edit_message_text", "message"])
        mock_query.edit_message_text = AsyncMock(return_value="edit_sent")
        mock_sub_msg = MagicMock(spec=["reply_text"])
        mock_sub_msg.reply_text = AsyncMock(return_value="reply_sent")
        mock_query.message = mock_sub_msg

        text = "Paragraph 1: " + ("A" * 2500) + "\n\nParagraph 2: " + ("B" * 2500)
        markup = MagicMock()

        res = asyncio.run(send_split_html(mock_query, text, reply_markup=markup, max_length=3000, edit_initial=True))
        self.assertGreater(len(res), 1)
        self.assertEqual(mock_query.edit_message_text.call_count, 1)
        self.assertEqual(mock_sub_msg.reply_text.call_count, len(res) - 1)

    def test_send_split_bot_message(self) -> None:
        import asyncio
        from unittest.mock import AsyncMock, MagicMock
        from app.services.telegram.formatters import send_split_bot_message

        mock_bot = MagicMock()
        mock_bot.send_message = AsyncMock(return_value="bot_sent")

        text = "Broadcast line " * 300  # ~4500 chars
        markup = MagicMock()

        res = asyncio.run(send_split_bot_message(mock_bot, 12345, text, reply_markup=markup, max_length=3000))
        self.assertGreater(len(res), 1)
        self.assertEqual(mock_bot.send_message.call_count, len(res))
        for call_args in mock_bot.send_message.call_args_list:
            self.assertEqual(call_args[1].get("chat_id"), 12345)

    def test_status_card_animator_render(self) -> None:
        from unittest.mock import MagicMock
        from app.services.telegram.formatters import StatusCardAnimator

        mock_msg = MagicMock()
        animator = StatusCardAnimator(
            mock_msg,
            title="TIKTOK ULTRA-DOWNLOADER",
            stage="វិភាគតំណភ្ជាប់",
            percent=30,
            detail="កំពុងស្វែងរកទិន្នន័យ...",
        )
        rendered = animator.render()
        self.assertIn("TIKTOK ULTRA-DOWNLOADER", rendered)
        self.assertIn("▰▰▰▱▱▱▱▱▱▱", rendered)
        self.assertIn("30%", rendered)
        self.assertIn("វិភាគតំណភ្ជាប់", rendered)
        self.assertIn("⠋", rendered)
        self.assertIn("កំពុងស្វែងរកទិន្នន័យ...", rendered)

    def test_status_card_animator_lifecycle(self) -> None:
        import asyncio
        from unittest.mock import AsyncMock, MagicMock
        from app.services.telegram.formatters import StatusCardAnimator

        async def _test() -> None:
            mock_msg = MagicMock()
            mock_msg.edit_text = AsyncMock()

            animator = StatusCardAnimator(
                mock_msg,
                title="AI Thinking",
                stage="Analyzing",
                percent=10,
                interval=0.05,
            )
            async with animator:
                self.assertIsNotNone(animator._task)
                self.assertFalse(animator._task.done())
                await animator.push_state(stage="Generating", percent=60)
                self.assertEqual(animator.stage, "Generating")
                self.assertEqual(animator.percent, 60)
                self.assertTrue(mock_msg.edit_text.called)

            self.assertTrue(animator._closed)
            self.assertIsNone(animator._task)

        asyncio.run(_test())

    def test_status_card_animator_none_message(self) -> None:
        import asyncio
        from app.services.telegram.formatters import StatusCardAnimator

        async def _test() -> None:
            animator = StatusCardAnimator(None, title="Test None")
            animator.start()
            self.assertIsNone(animator._task)
            await animator.push_state(stage="Next", percent=50)
            await animator.stop()

        asyncio.run(_test())

    def test_status_card_animator_stale_message_termination(self) -> None:
        import asyncio
        from unittest.mock import AsyncMock, MagicMock
        from app.services.telegram.formatters import StatusCardAnimator

        async def _test() -> None:
            mock_msg = MagicMock()
            mock_msg.edit_text = AsyncMock(side_effect=Exception("BadRequest: message to edit not found"))

            animator = StatusCardAnimator(
                mock_msg,
                title="Stale Test",
                interval=0.01,
            )
            animator.start()
            for _ in range(50):
                if animator._closed:
                    break
                await asyncio.sleep(0.02)

            self.assertTrue(animator._closed)
            await animator.stop()

        asyncio.run(_test())


if __name__ == "__main__":
    unittest.main()
