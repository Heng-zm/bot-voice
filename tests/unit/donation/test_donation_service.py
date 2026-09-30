"""Unit tests for app.features.donation."""

from __future__ import annotations

import unittest
from unittest.mock import AsyncMock, patch

from app.features.donation.service import DonationService, donation_service


class DonationServiceUnitTests(unittest.TestCase):
    def test_service_initialization(self):
        self.assertIsNotNone(donation_service)
        svc = DonationService()
        self.assertIsNotNone(svc.store)

    def test_generate_khqr_payload(self):
        svc = DonationService()
        res = svc.generate_khqr(amount=5.0, currency="USD", bill_no="BILL001")
        self.assertIn("qr_string", res)
        self.assertIn("md5", res)
        self.assertEqual(res["bill_no"], "BILL001")


if __name__ == "__main__":
    unittest.main()
