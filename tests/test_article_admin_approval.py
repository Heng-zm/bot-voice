"""Unit tests for Admin Article Approval Workflow, Source URL Management, and Broadcasting."""

from __future__ import annotations

import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock, patch
import uuid

from app.services.ai.article_storage import (
    add_article_source,
    get_article_sources,
    get_pending_article,
    is_article_pending,
    is_article_sent,
    mark_article_sent,
    remove_article_source,
    save_pending_article,
    toggle_article_source,
    update_pending_status,
)


class TestArticleSourcesManagement(unittest.IsolatedAsyncioTestCase):
    """Test adding, listing, toggling, and removing base article sources."""

    async def test_add_and_remove_source(self) -> None:
        unique_url = f"https://news-{uuid.uuid4().hex[:8]}.com.kh"
        # 1. Add valid source
        ok, msg, sid = await add_article_source(unique_url, name="Test Cambodia News", added_by=999)
        self.assertTrue(ok)
        self.assertTrue(sid > 0)
        self.assertIn("Added source", msg)

        # 2. Duplicate URL rejected
        dup_ok, dup_msg, _ = await add_article_source(unique_url, name="Duplicate")
        self.assertFalse(dup_ok)
        self.assertIn("already exists", dup_msg.lower())

        # 3. Appears in sources list
        sources = await get_article_sources(active_only=True)
        found = [s for s in sources if s["url"] == unique_url]
        self.assertEqual(1, len(found))
        self.assertEqual("Test Cambodia News", found[0]["name"])

        # 4. Remove source
        rem_ok = await remove_article_source(sid)
        self.assertTrue(rem_ok)

        # 5. Verify gone
        after_sources = await get_article_sources(active_only=False)
        self.assertFalse(any(s["url"] == unique_url for s in after_sources))

    async def test_rejects_invalid_scheme(self) -> None:
        ok, msg, _ = await add_article_source("ftp://invalidscheme.com")
        self.assertFalse(ok)
        self.assertIn("invalid url schema", msg.lower())

    async def test_toggle_source_active(self) -> None:
        unique_url = f"https://toggle-{uuid.uuid4().hex[:8]}.com"
        ok, _, sid = await add_article_source(unique_url, name="Toggle News")
        self.assertTrue(ok)

        # Toggle inactive
        await toggle_article_source(sid, False)
        active_list = await get_article_sources(active_only=True)
        self.assertFalse(any(s["id"] == sid for s in active_list))

        all_list = await get_article_sources(active_only=False)
        self.assertTrue(any(s["id"] == sid for s in all_list))

        # Cleanup
        await remove_article_source(sid)


class TestPendingArticlesQueue(unittest.IsolatedAsyncioTestCase):
    """Test saving pending articles, checking states, and updating approval status."""

    async def test_pending_article_lifecycle(self) -> None:
        test_hash = f"pend_hash_{uuid.uuid4().hex[:12]}"
        test_url = "https://freshnewsasia.com/article/999"

        # Initially not pending
        self.assertFalse(await is_article_pending(test_hash))

        payload = {
            "hash": test_hash,
            "url": test_url,
            "title": "Breaking News Headline",
            "khmer_title": "ព័ត៌មានទាន់ហេតុការណ៍",
            "khmer_summary": "សេចក្ដីសង្ខេបព័ត៌មាន",
            "khmer_tts_script": "ព័ត៌មានទាន់ហេតុការណ៍។ សេចក្ដីសង្ខេប",
            "original_summary": "English summary text",
            "body_text": "Full article text content...",
            "image_url": "https://img.com/cover.jpg",
            "telegraph_url": "https://telegra.ph/News-01",
            "audio_bytes": b"MP3_BYTES_SAMPLE",
            "status": "pending",
        }
        await save_pending_article(payload)

        # Now is pending
        self.assertTrue(await is_article_pending(test_hash))
        record = await get_pending_article(test_hash)
        self.assertIsNotNone(record)
        self.assertEqual("pending", record["status"])
        self.assertEqual("ព័ត៌មានទាន់ហេតុការណ៍", record["khmer_title"])
        self.assertEqual("https://telegra.ph/News-01", record["telegraph_url"])

        # Update to approved
        updated = await update_pending_status(test_hash, "approved", reviewed_by=12345)
        self.assertTrue(updated)
        after_appr = await get_pending_article(test_hash)
        self.assertEqual("approved", after_appr["status"])
        self.assertEqual(12345, after_appr["reviewed_by"])
        self.assertIsNotNone(after_appr["reviewed_at"])


class TestAdminApprovalCallbacks(unittest.IsolatedAsyncioTestCase):
    """Test Telegram callback queries for admin approval, rejection, and source deletion."""

    async def test_unauthorized_user_blocked(self) -> None:
        from app.services.telegram.callbacks import article_callback

        update = MagicMock()
        update.callback_query.data = "art_adm:appr:test_hash_123"
        update.callback_query.from_user.id = 99999
        update.callback_query.answer = AsyncMock()
        context = MagicMock()

        with patch("app.legacy._is_admin", return_value=False):
            await article_callback(update, context)
            update.callback_query.answer.assert_called_once()
            self.assertIn("Admin", update.callback_query.answer.call_args[0][0])

    async def test_admin_approves_and_broadcasts(self) -> None:
        from app.services.telegram.callbacks import article_callback

        test_hash = f"appr_hash_{uuid.uuid4().hex[:12]}"
        await save_pending_article({
            "hash": test_hash,
            "url": "https://news.com/1",
            "title": "Approval Test",
            "khmer_title": "ព័ត៌មានអនុម័ត",
            "khmer_summary": "សេចក្ដីសង្ខេប",
            "status": "pending",
        })

        update = MagicMock()
        update.callback_query.data = f"art_adm:appr:{test_hash}"
        update.callback_query.from_user.id = 11111
        update.callback_query.answer = AsyncMock()
        update.callback_query.message.edit_reply_markup = AsyncMock()
        update.callback_query.message.reply_text = AsyncMock()
        context = MagicMock()

        with patch("app.legacy._is_admin", return_value=True), \
             patch("app.services.ai.article_monitor.broadcast_approved_article", new_callable=AsyncMock) as mock_bcast:
            mock_bcast.return_value = (5, 0)
            await article_callback(update, context)

            # Marked sent and approved
            record = await get_pending_article(test_hash)
            self.assertEqual("approved", record["status"])
            self.assertTrue(await is_article_sent(test_hash))
            mock_bcast.assert_called_once()
            self.assertIn("បានអនុម័ត", update.callback_query.answer.call_args[0][0])

    async def test_admin_rejects_article(self) -> None:
        from app.services.telegram.callbacks import article_callback

        test_hash = f"rej_hash_{uuid.uuid4().hex[:12]}"
        await save_pending_article({
            "hash": test_hash,
            "url": "https://news.com/2",
            "title": "Reject Test",
            "khmer_title": "ព័ត៌មានបដិសេធ",
            "khmer_summary": "សេចក្ដីសង្ខេប",
            "status": "pending",
        })

        update = MagicMock()
        update.callback_query.data = f"art_adm:rej:{test_hash}"
        update.callback_query.from_user.id = 11111
        update.callback_query.answer = AsyncMock()
        update.callback_query.message.edit_reply_markup = AsyncMock()
        update.callback_query.message.reply_text = AsyncMock()
        context = MagicMock()

        with patch("app.legacy._is_admin", return_value=True):
            await article_callback(update, context)

            record = await get_pending_article(test_hash)
            self.assertEqual("rejected", record["status"])
            self.assertTrue(await is_article_sent(test_hash))
            self.assertIn("បដិសេធ", update.callback_query.answer.call_args[0][0])

    async def test_admin_deletes_source_callback(self) -> None:
        from app.services.telegram.callbacks import article_callback

        ok, _, sid = await add_article_source("https://temp-del-news.com", name="Temp")
        self.assertTrue(ok)

        update = MagicMock()
        update.callback_query.data = f"art_src:del:{sid}"
        update.callback_query.from_user.id = 11111
        update.callback_query.answer = AsyncMock()
        update.callback_query.message.reply_text = AsyncMock()
        context = MagicMock()

        with patch("app.legacy._is_admin", return_value=True):
            await article_callback(update, context)
            self.assertIn("បានលុប", update.callback_query.answer.call_args[0][0])
            sources = await get_article_sources(active_only=False)
            self.assertFalse(any(s["id"] == sid for s in sources))


class TestArticleBroadcasting(unittest.IsolatedAsyncioTestCase):
    """Test broadcasting approved article cards and voice notes to users."""

    async def test_broadcast_approved_article(self) -> None:
        from app.services.ai.article_monitor import broadcast_approved_article

        test_hash = f"bcast_hash_{uuid.uuid4().hex[:12]}"
        await save_pending_article({
            "hash": test_hash,
            "url": "https://freshnews.com/777",
            "title": "Broadcast Headline",
            "khmer_title": "ព័ត៌មានផ្សាយជាសាធារណៈ",
            "khmer_summary": "ចំណុចសំខាន់ៗនៃព័ត៌មាន",
            "telegraph_url": "https://telegra.ph/Bcast-01",
            "audio_bytes": b"\xff\xfb\x90\x44" + b"\x00" * 100,
            "status": "approved",
        })

        mock_bot = MagicMock()
        mock_bot.send_message = AsyncMock()
        mock_bot.send_photo = AsyncMock()
        mock_bot.send_audio = AsyncMock()
        mock_bot.send_voice = AsyncMock()

        with patch("app.legacy.get_all_user_ids", return_value=[1001, 1002]), \
             patch("app.services.podcast.store.podcast_store.get_all_subscribers", return_value=[1002, 1003]):
            success, failed = await broadcast_approved_article(mock_bot, test_hash)

            self.assertEqual(3, success)
            self.assertEqual(0, failed)
            # Verified sent to 3 unique users (1001, 1002, 1003)
            self.assertEqual(3, mock_bot.send_message.call_count)


class TestAdminCommands(unittest.IsolatedAsyncioTestCase):
    """Test /article_sources and /article_scan commands."""

    async def test_cmd_article_sources_guard(self) -> None:
        from app.services.telegram.commands import cmd_article_sources

        update = MagicMock()
        update.effective_message.text = "/article_sources"
        update.effective_user.id = 99999
        update.effective_message.reply_text = AsyncMock()
        context = MagicMock()

        with patch("app.legacy._is_admin", return_value=False):
            await cmd_article_sources(update, context)
            update.effective_message.reply_text.assert_called_once()
            self.assertIn("Admin", update.effective_message.reply_text.call_args[0][0])

    async def test_cmd_article_sources_add(self) -> None:
        from app.services.telegram.commands import cmd_article_sources

        test_url = f"https://newsource-{uuid.uuid4().hex[:6]}.com"
        update = MagicMock()
        update.effective_message.text = f"/article_source add {test_url} Test Source"
        update.effective_user.id = 11111
        update.effective_message.reply_text = AsyncMock()
        context = MagicMock()

        with patch("app.legacy._is_admin", return_value=True), \
             patch("app.services.ai.article_reader.is_safe_public_url", return_value=(True, "")):
            await cmd_article_sources(update, context)
            update.effective_message.reply_text.assert_called_once()
            self.assertIn("បានបញ្ចូលប្រភពព័ត៌មានជោគជ័យ", update.effective_message.reply_text.call_args[0][0])

        # Cleanup
        await remove_article_source(test_url)

    async def test_cmd_article_scan(self) -> None:
        from app.services.telegram.commands import cmd_article_scan

        update = MagicMock()
        update.effective_message.text = "/article_scan"
        update.effective_user.id = 11111
        status_msg = MagicMock()
        status_msg.edit_text = AsyncMock()
        update.effective_message.reply_text = AsyncMock(return_value=status_msg)
        context = MagicMock()

        with patch("app.legacy._is_admin", return_value=True), \
             patch("app.services.ai.article_monitor.scan_sources_and_notify_admin", new_callable=AsyncMock) as mock_scan:
            mock_scan.return_value = [{"hash": "h1"}, {"hash": "h2"}]
            await cmd_article_scan(update, context)
            status_msg.edit_text.assert_called_once()
            self.assertIn("រកឃើញ <b>2</b> ព័ត៌មានថ្មី", status_msg.edit_text.call_args[0][0])


if __name__ == "__main__":
    unittest.main()
