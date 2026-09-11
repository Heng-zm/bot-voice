"""Unit tests for donation system, KHQR generation, blessing scripts, and store."""

from __future__ import annotations

import os
import shutil
import tempfile
import unittest

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


if __name__ == "__main__":
    unittest.main()
