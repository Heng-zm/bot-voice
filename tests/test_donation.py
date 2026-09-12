"""Unit tests for donation system, KHQR generation, blessing scripts, and store."""

from __future__ import annotations

import os
import shutil
import tempfile
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from app.services.ai.gemini import normalize_gemini_model
from app.services.donation.blessing import generate_blessing_script
from app.services.donation.khqr import (
    BakongKHQR,
    crc16_ccitt,
    generate_khqr_string,
)
from app.services.donation.store import DonationStore


class DonationKHQRTests(unittest.TestCase):
    def test_crc16_ccitt(self) -> None:
        test_str = "00020101021229300010bakong@nbc0114chuo_kimheng@bkrt52045999530384054041.005802KH5912CHUO KIMHENG6010Phnom Penh62180108COFF00010708BOTVOICE6304"
        crc = crc16_ccitt(test_str)
        self.assertEqual(4, len(crc))
        self.assertTrue(all(c in "0123456789ABCDEF" for c in crc))

    def test_generate_khqr_string_dynamic(self) -> None:
        khqr = generate_khqr_string(
            account_id="chuo_kimheng@bkrt",
            merchant_name="CHUO KIMHENG",
            merchant_city="Phnom Penh",
            amount=1.0,
            currency="USD",
            bill_number="COFF01",
        )
        self.assertTrue(khqr.startswith("000201"))
        self.assertIn("chuo_kimheng@bkrt", khqr)
        self.assertIn("CHUO KIMHENG", khqr)
        self.assertIn("1.00", khqr)
        self.assertIn("6304", khqr)

    def test_bakong_khqr_wrapper(self) -> None:
        khqr_text, bill_no = BakongKHQR.generate(amount=2.0, currency="USD", user_id=123456, tier="milktea")
        self.assertTrue(bill_no.startswith("MILK"))
        self.assertTrue(len(khqr_text) > 50)
        self.assertIn("2.00", khqr_text)

    def test_dynamic_khqr_detection(self) -> None:
        dynamic_khqr, _ = BakongKHQR.generate(amount=5.0, currency="USD", user_id=123, tier="server")
        self.assertIn("010212", dynamic_khqr)
        self.assertIn("54045.00", dynamic_khqr)

        static_khqr = generate_khqr_string(amount=None)
        self.assertIn("010211", static_khqr)
        self.assertNotIn("54", static_khqr)


class BlessingScriptTests(unittest.TestCase):
    def test_coffee_tier_script(self) -> None:
        script = generate_blessing_script("Dara", tier="coffee", amount=1.0)
        self.assertIn("Dara", script)
        self.assertIn("កាហ្វេ ១ កែវ", script)

    def test_milktea_tier_script(self) -> None:
        script = generate_blessing_script("Sokha", tier="milktea", amount=2.0)
        self.assertIn("Sokha", script)
        self.assertIn("តែទឹកដោះគោ", script)

    def test_server_tier_script(self) -> None:
        script = generate_blessing_script("Bona", tier="server", amount=5.0)
        self.assertIn("Bona", script)
        self.assertIn("Server", script)

    def test_patron_tier_script(self) -> None:
        script = generate_blessing_script("Vireak", tier="patron", amount=10.0)
        self.assertIn("Vireak", script)
        self.assertIn("សប្បុរសធម៌", script)


class DonationStoreTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.test_dir = tempfile.mkdtemp()
        self.store_file = os.path.join(self.test_dir, "donations_test.json")
        self.store = DonationStore(file_path=self.store_file)

    async def asyncTearDown(self) -> None:
        shutil.rmtree(self.test_dir, ignore_errors=True)

    async def test_record_and_aggregate_donations(self) -> None:
        d1 = await self.store.record_donation(
            user_id=1001,
            full_name="User Alpha",
            amount=1.0,
            tier="coffee",
            blessing_sent=True,
        )
        self.assertEqual(1001, d1["user_id"])
        self.assertEqual(1.0, d1["amount"])

        await self.store.record_donation(
            user_id=1002,
            full_name="User Beta",
            amount=5.0,
            tier="server",
            blessing_sent=True,
        )

        await self.store.record_donation(
            user_id=1001,
            full_name="User Alpha",
            amount=2.0,
            tier="milktea",
            blessing_sent=True,
        )

        stats = await self.store.get_donation_stats()
        self.assertEqual(8.0, stats["total_usd"])
        self.assertEqual(2, stats["total_donors"])

        top = await self.store.get_top_supporters(10)
        self.assertEqual(2, len(top))
        self.assertEqual(1002, top[0]["user_id"])
        self.assertEqual(5.0, top[0]["total_amount"])
        self.assertEqual("🥇", top[0]["badge"])

        self.assertEqual(1001, top[1]["user_id"])
        self.assertEqual(3.0, top[1]["total_amount"])
        self.assertEqual("🥈", top[1]["badge"])

        recent = await self.store.get_recent_donations(5)
        self.assertEqual(3, len(recent))
        self.assertEqual("milktea", recent[0]["tier"])


class GeminiModelNormalizationTests(unittest.TestCase):
    def test_normalizes_fictitious_models(self) -> None:
        self.assertEqual("gemini-2.5-flash", normalize_gemini_model("gemini-3.6-flash"))
        self.assertEqual("gemini-2.5-flash", normalize_gemini_model("gemini-3.0-flash"))
        self.assertEqual("gemini-2.5-flash", normalize_gemini_model("gemini-3.5-flash"))
        self.assertEqual("gemini-2.5-flash", normalize_gemini_model("gemini-3.6"))
        self.assertEqual("gemini-2.5-pro", normalize_gemini_model("gemini-3.1-pro-preview"))

    def test_preserves_valid_models(self) -> None:
        self.assertEqual("gemini-2.5-flash", normalize_gemini_model("gemini-2.5-flash"))
        self.assertEqual("gemini-2.0-flash", normalize_gemini_model("gemini-2.0-flash"))
        self.assertEqual("gemini-1.5-flash", normalize_gemini_model("gemini-1.5-flash"))
        self.assertEqual("gemini-2.5-pro", normalize_gemini_model("gemini-2.5-pro"))


class AddDonorCommandTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.test_dir = tempfile.mkdtemp()
        self.store_file = os.path.join(self.test_dir, "test_donations.json")
        from app.services.donation import handlers
        handlers.donation_store = DonationStore(file_path=self.store_file)

    async def asyncTearDown(self) -> None:
        shutil.rmtree(self.test_dir, ignore_errors=True)

    @patch("app.services.donation.handlers.deliver_voice_blessing", new_callable=AsyncMock)
    async def test_adddonor_full_args(self, mock_blessing: AsyncMock) -> None:
        from app.services.donation.handlers import cmd_adddonor, donation_store

        mock_blessing.return_value = True
        mock_update = MagicMock()
        mock_update.effective_user.id = 9999
        mock_msg = MagicMock()
        mock_msg.reply_text = AsyncMock()
        mock_update.effective_message = mock_msg
        mock_context = MagicMock()
        mock_context.args = ["1272791365", "1.0", "coffee", "Dara"]

        with patch("app.services.donation.handlers.is_admin_user", return_value=True):
            await cmd_adddonor(mock_update, mock_context)

        mock_msg.reply_text.assert_awaited_once()
        reply = mock_msg.reply_text.call_args[0][0]
        self.assertIn("បានកត់ត្រាការឧបត្ថម្ភជោគជ័យ", reply)
        self.assertIn("Dara", reply)
        self.assertIn("1272791365", reply)

        stats = await donation_store.get_donation_stats()
        self.assertEqual(1.0, stats["total_usd"])
        self.assertEqual(1, stats["total_donors"])

    @patch("app.services.donation.handlers.deliver_voice_blessing", new_callable=AsyncMock)
    async def test_adddonor_omitted_tier(self, mock_blessing: AsyncMock) -> None:
        from app.services.donation.handlers import cmd_adddonor

        mock_blessing.return_value = True
        mock_update = MagicMock()
        mock_update.effective_user.id = 9999
        mock_msg = MagicMock()
        mock_msg.reply_text = AsyncMock()
        mock_update.effective_message = mock_msg
        mock_context = MagicMock()
        mock_context.args = ["1272791365", "2.0", "Sokha"]

        with patch("app.services.donation.handlers.is_admin_user", return_value=True):
            await cmd_adddonor(mock_update, mock_context)

        mock_msg.reply_text.assert_awaited_once()
        reply = mock_msg.reply_text.call_args[0][0]
        self.assertIn("Sokha", reply)
        self.assertIn("milktea", reply)

    async def test_adddonor_usage_help(self) -> None:
        from app.services.donation.handlers import cmd_adddonor

        mock_update = MagicMock()
        mock_update.effective_user.id = 9999
        mock_msg = MagicMock()
        mock_msg.reply_text = AsyncMock()
        mock_update.effective_message = mock_msg
        mock_context = MagicMock()
        mock_context.args = ["help"]

        with patch("app.services.donation.handlers.is_admin_user", return_value=True):
            await cmd_adddonor(mock_update, mock_context)

        mock_msg.reply_text.assert_awaited_once()
        reply = mock_msg.reply_text.call_args[0][0]
        self.assertIn("របៀបប្រើប្រាស់ពាក្យបញ្ជា /adddonor", reply)
        self.assertIn("/adddonor &lt;user_id&gt; &lt;amount&gt; [tier] [name]", reply)
        self.assertIn("coffee", reply)
        self.assertIn("patron", reply)


class AddDonorWizardFlowTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.test_dir = tempfile.mkdtemp()
        self.store_file = os.path.join(self.test_dir, "test_donations_wizard.json")
        from app.services.donation import handlers
        handlers.donation_store = DonationStore(file_path=self.store_file)

    async def asyncTearDown(self) -> None:
        shutil.rmtree(self.test_dir, ignore_errors=True)

    async def test_step1_launch_wizard_empty_args(self) -> None:
        from app.services.donation.handlers import cmd_adddonor

        mock_update = MagicMock()
        mock_update.effective_user.id = 9999
        mock_msg = MagicMock()
        mock_msg.reply_text = AsyncMock()
        mock_update.effective_message = mock_msg
        mock_context = MagicMock()
        mock_context.args = []
        mock_context.user_data = {}

        with patch("app.services.donation.handlers.is_admin_user", return_value=True):
            await cmd_adddonor(mock_update, mock_context)

        self.assertEqual("wait_user_id", mock_context.user_data.get("adddonor_state"))
        mock_msg.reply_text.assert_awaited_once()
        reply = mock_msg.reply_text.call_args[0][0]
        self.assertIn("ជំហានទី ១/៤", reply)
        self.assertIn("Telegram User ID", reply)

    async def test_step1_launch_wizard_one_arg_userid(self) -> None:
        from app.services.donation.handlers import cmd_adddonor

        mock_update = MagicMock()
        mock_update.effective_user.id = 9999
        mock_msg = MagicMock()
        mock_msg.reply_text = AsyncMock()
        mock_update.effective_message = mock_msg
        mock_context = MagicMock()
        mock_context.args = ["1272791365"]
        mock_context.user_data = {}
        mock_context.bot = None

        with patch("app.services.donation.handlers.is_admin_user", return_value=True):
            await cmd_adddonor(mock_update, mock_context)

        self.assertEqual("wait_amount", mock_context.user_data.get("adddonor_state"))
        self.assertEqual(1272791365, mock_context.user_data["adddonor_data"]["user_id"])
        mock_msg.reply_text.assert_awaited_once()
        reply = mock_msg.reply_text.call_args[0][0]
        self.assertIn("ជំហានទី ២/៤", reply)

    async def test_step1_to_step2_numeric_id_text(self) -> None:
        from app.services.donation.handlers import handle_adddonor_text

        mock_update = MagicMock()
        mock_update.effective_user.id = 9999
        mock_msg = MagicMock()
        mock_msg.text = "1272791365"
        mock_msg.caption = None
        mock_msg.forward_from = None
        mock_msg.forward_from_chat = None
        mock_msg.reply_text = AsyncMock()
        mock_update.effective_message = mock_msg

        mock_context = MagicMock()
        mock_context.user_data = {"adddonor_state": "wait_user_id", "adddonor_data": {}}
        mock_context.bot = None

        with patch("app.services.donation.handlers.is_admin_user", return_value=True):
            handled = await handle_adddonor_text(mock_update, mock_context)

        self.assertTrue(handled)
        self.assertEqual("wait_amount", mock_context.user_data["adddonor_state"])
        self.assertEqual(1272791365, mock_context.user_data["adddonor_data"]["user_id"])
        mock_msg.reply_text.assert_awaited_once()
        self.assertIn("ជំហានទី ២/៤", mock_msg.reply_text.call_args[0][0])

    async def test_step1_to_step2_forwarded_message(self) -> None:
        from app.services.donation.handlers import handle_adddonor_text

        mock_update = MagicMock()
        mock_update.effective_user.id = 9999
        mock_msg = MagicMock()
        mock_msg.text = None
        mock_msg.caption = "Donation receipt"
        mock_fwd = MagicMock()
        mock_fwd.id = 777888999
        mock_fwd.first_name = "Bona"
        mock_fwd.full_name = "Bona Test"
        mock_msg.forward_from = mock_fwd
        mock_msg.forward_from_chat = None
        mock_msg.reply_text = AsyncMock()
        mock_update.effective_message = mock_msg

        mock_context = MagicMock()
        mock_context.user_data = {"adddonor_state": "wait_user_id", "adddonor_data": {}}
        mock_context.bot = None

        with patch("app.services.donation.handlers.is_admin_user", return_value=True):
            handled = await handle_adddonor_text(mock_update, mock_context)

        self.assertTrue(handled)
        self.assertEqual("wait_amount", mock_context.user_data["adddonor_state"])
        self.assertEqual(777888999, mock_context.user_data["adddonor_data"]["user_id"])
        self.assertEqual("Bona", mock_context.user_data["adddonor_data"]["auto_name"])

    async def test_step2_to_step3_callback_tier(self) -> None:
        from app.services.donation.handlers import donation_callback

        mock_update = MagicMock()
        mock_update.effective_user.id = 9999
        mock_query = MagicMock()
        mock_query.data = "donate_add_tier:server:5.0"
        mock_query.answer = AsyncMock()
        mock_query.message = MagicMock()
        mock_query.message.edit_text = AsyncMock()
        mock_update.callback_query = mock_query

        mock_context = MagicMock()
        mock_context.user_data = {
            "adddonor_state": "wait_amount",
            "adddonor_data": {"user_id": 1272791365, "auto_name": "Dara"},
        }

        with patch("app.services.donation.handlers.is_admin_user", return_value=True):
            await donation_callback(mock_update, mock_context)

        self.assertEqual("wait_name", mock_context.user_data["adddonor_state"])
        self.assertEqual(5.0, mock_context.user_data["adddonor_data"]["amount"])
        self.assertEqual("server", mock_context.user_data["adddonor_data"]["tier"])
        mock_query.message.edit_text.assert_awaited_once()
        self.assertIn("ជំហានទី ៣/៤", mock_query.message.edit_text.call_args[0][0])

    async def test_step2_to_step3_text_amount(self) -> None:
        from app.services.donation.handlers import handle_adddonor_text

        mock_update = MagicMock()
        mock_update.effective_user.id = 9999
        mock_msg = MagicMock()
        mock_msg.text = "$10.0"
        mock_msg.caption = None
        mock_msg.reply_text = AsyncMock()
        mock_update.effective_message = mock_msg

        mock_context = MagicMock()
        mock_context.user_data = {
            "adddonor_state": "wait_amount",
            "adddonor_data": {"user_id": 1272791365, "auto_name": "Dara"},
        }

        with patch("app.services.donation.handlers.is_admin_user", return_value=True):
            handled = await handle_adddonor_text(mock_update, mock_context)

        self.assertTrue(handled)
        self.assertEqual("wait_name", mock_context.user_data["adddonor_state"])
        self.assertEqual(10.0, mock_context.user_data["adddonor_data"]["amount"])
        self.assertEqual("patron", mock_context.user_data["adddonor_data"]["tier"])
        mock_msg.reply_text.assert_awaited_once()
        self.assertIn("ជំហានទី ៣/៤", mock_msg.reply_text.call_args[0][0])

    async def test_step3_to_step4_text_name(self) -> None:
        from app.services.donation.handlers import handle_adddonor_text

        mock_update = MagicMock()
        mock_update.effective_user.id = 9999
        mock_msg = MagicMock()
        mock_msg.text = "Brother Sok"
        mock_msg.caption = None
        mock_msg.reply_text = AsyncMock()
        mock_update.effective_message = mock_msg

        mock_context = MagicMock()
        mock_context.user_data = {
            "adddonor_state": "wait_name",
            "adddonor_data": {"user_id": 1272791365, "amount": 5.0, "tier": "server", "auto_name": "Dara"},
        }

        with patch("app.services.donation.handlers.is_admin_user", return_value=True):
            handled = await handle_adddonor_text(mock_update, mock_context)

        self.assertTrue(handled)
        self.assertEqual("confirm", mock_context.user_data["adddonor_state"])
        self.assertEqual("Brother Sok", mock_context.user_data["adddonor_data"]["custom_name"])
        mock_msg.reply_text.assert_awaited_once()
        self.assertIn("ជំហានទី ៤/៤", mock_msg.reply_text.call_args[0][0])

    async def test_step3_to_step4_callback_auto_name(self) -> None:
        from app.services.donation.handlers import donation_callback

        mock_update = MagicMock()
        mock_update.effective_user.id = 9999
        mock_query = MagicMock()
        mock_query.data = "donate_add_use_auto"
        mock_query.answer = AsyncMock()
        mock_query.message = MagicMock()
        mock_query.message.edit_text = AsyncMock()
        mock_update.callback_query = mock_query

        mock_context = MagicMock()
        mock_context.user_data = {
            "adddonor_state": "wait_name",
            "adddonor_data": {"user_id": 1272791365, "auto_name": "Dara Official", "amount": 1.0, "tier": "coffee"},
        }

        with patch("app.services.donation.handlers.is_admin_user", return_value=True):
            await donation_callback(mock_update, mock_context)

        self.assertEqual("confirm", mock_context.user_data["adddonor_state"])
        self.assertEqual("Dara Official", mock_context.user_data["adddonor_data"]["custom_name"])
        mock_query.message.edit_text.assert_awaited_once()
        self.assertIn("ជំហានទី ៤/៤", mock_query.message.edit_text.call_args[0][0])

    @patch("app.services.donation.handlers.deliver_voice_blessing", new_callable=AsyncMock)
    async def test_step4_confirm_execution(self, mock_blessing: AsyncMock) -> None:
        from app.services.donation.handlers import donation_callback, donation_store

        mock_blessing.return_value = True

        mock_update = MagicMock()
        mock_update.effective_user.id = 9999
        mock_query = MagicMock()
        mock_query.data = "donate_add_confirm"
        mock_query.answer = AsyncMock()
        mock_query.message = MagicMock()
        mock_query.message.edit_text = AsyncMock()
        mock_update.callback_query = mock_query

        mock_context = MagicMock()
        mock_context.bot = MagicMock()
        mock_context.user_data = {
            "adddonor_state": "confirm",
            "adddonor_data": {
                "user_id": 1272791365,
                "custom_name": "Lok Ta",
                "amount": 5.0,
                "tier": "server",
            },
        }

        with patch("app.services.donation.handlers.is_admin_user", return_value=True):
            await donation_callback(mock_update, mock_context)

        self.assertNotIn("adddonor_state", mock_context.user_data)
        self.assertNotIn("adddonor_data", mock_context.user_data)

        # Check recorded in store
        stats = await donation_store.get_donation_stats()
        self.assertEqual(5.0, stats["total_usd"])
        self.assertEqual(1, stats["total_donors"])

        mock_blessing.assert_awaited_once_with(
            mock_context.bot,
            user_id=1272791365,
            donor_name="Lok Ta",
            tier="server",
            amount=5.0,
        )

    async def test_cancel_via_text(self) -> None:
        from app.services.donation.handlers import handle_adddonor_text

        mock_update = MagicMock()
        mock_update.effective_user.id = 9999
        mock_msg = MagicMock()
        mock_msg.text = "/cancel"
        mock_msg.reply_text = AsyncMock()
        mock_update.effective_message = mock_msg

        mock_context = MagicMock()
        mock_context.user_data = {"adddonor_state": "wait_user_id", "adddonor_data": {}}

        with patch("app.services.donation.handlers.is_admin_user", return_value=True):
            handled = await handle_adddonor_text(mock_update, mock_context)

        self.assertTrue(handled)
        self.assertNotIn("adddonor_state", mock_context.user_data)
        self.assertNotIn("adddonor_data", mock_context.user_data)
        mock_msg.reply_text.assert_awaited_once()
        self.assertIn("បោះបង់", mock_msg.reply_text.call_args[0][0])

    async def test_cancel_via_callback(self) -> None:
        from app.services.donation.handlers import donation_callback

        mock_update = MagicMock()
        mock_update.effective_user.id = 9999
        mock_query = MagicMock()
        mock_query.data = "donate_add_cancel"
        mock_query.answer = AsyncMock()
        mock_query.message = MagicMock()
        mock_query.message.edit_text = AsyncMock()
        mock_update.callback_query = mock_query

        mock_context = MagicMock()
        mock_context.user_data = {"adddonor_state": "wait_amount", "adddonor_data": {}}

        await donation_callback(mock_update, mock_context)

        self.assertNotIn("adddonor_state", mock_context.user_data)
        self.assertNotIn("adddonor_data", mock_context.user_data)
        mock_query.message.edit_text.assert_awaited_once()
        self.assertIn("បោះបង់", mock_query.message.edit_text.call_args[0][0])


if __name__ == "__main__":
    unittest.main()

