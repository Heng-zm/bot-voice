from __future__ import annotations

import unittest
from unittest.mock import patch

from app.services.telegram.deduplication import WebhookReplayStore


class WebhookReplayStoreTests(unittest.TestCase):
    def test_claim_complete_and_duplicate(self) -> None:
        store = WebhookReplayStore()
        state, token = store.claim(123, include_token=True)
        self.assertEqual("claimed", state)
        self.assertTrue(token)
        self.assertEqual("processing", store.claim(123))
        self.assertTrue(store.complete(123, claim_token=token))
        self.assertEqual("completed", store.claim(123))

    def test_expired_processing_lease_is_reclaimed(self) -> None:
        store = WebhookReplayStore()
        with patch.dict("os.environ", {"WEBHOOK_PROCESSING_TTL_S": "15"}):
            with patch(
                "app.services.telegram.deduplication.time.monotonic",
                side_effect=[100.0, 116.0, 116.0, 116.0],
            ):
                first_state, first_token = store.claim(99, include_token=True)
                second_state, second_token = store.claim(99, include_token=True)
                self.assertEqual("claimed", first_state)
                self.assertEqual("claimed", second_state)
                self.assertNotEqual(first_token, second_token)
                self.assertFalse(store.release(99, claim_token=first_token))
                self.assertFalse(store.complete(99, claim_token=first_token))

    def test_old_owner_cannot_complete_reclaimed_lease(self) -> None:
        store = WebhookReplayStore()
        first_state, first_token = store.claim(7, include_token=True)
        self.assertEqual("claimed", first_state)
        self.assertTrue(store.release(7, claim_token=first_token))
        second_state, second_token = store.claim(7, include_token=True)
        self.assertEqual("claimed", second_state)
        self.assertNotEqual(first_token, second_token)
        self.assertFalse(store.complete(7, claim_token=first_token))
        self.assertTrue(store.complete(7, claim_token=second_token))


if __name__ == "__main__":
    unittest.main()
