from __future__ import annotations

import asyncio
import unittest

from app.services.telegram.deduplication import WebhookUpdateDeduplicator


class WebhookUpdateDeduplicatorTests(unittest.IsolatedAsyncioTestCase):
    def make_service(self) -> WebhookUpdateDeduplicator:
        return WebhookUpdateDeduplicator(
            environ={
                "WEBHOOK_REPLAY_TTL_S": "600",
                "WEBHOOK_PROCESSING_TTL_S": "120",
            },
            replay_key_builder=lambda update_id: f"tests:telegram:{update_id}",
        )

    async def test_memory_lifecycle_does_not_depend_on_legacy_runtime(self) -> None:
        service = self.make_service()

        state, token = await service.claim(101, include_token=True)

        self.assertEqual("claimed", state)
        self.assertIsNotNone(token)
        self.assertEqual(
            ("processing", None),
            await service.claim(101, include_token=True),
        )
        self.assertTrue(await service.complete(101, claim_token=token))
        self.assertEqual("completed", await service.claim(101))

    async def test_only_one_concurrent_memory_claim_wins(self) -> None:
        service = self.make_service()

        results = await asyncio.gather(
            *(service.claim(202, include_token=True) for _index in range(25))
        )

        self.assertEqual(1, sum(state == "claimed" for state, _token in results))
        self.assertEqual(24, sum(state == "processing" for state, _token in results))

    async def test_expired_owner_token_cannot_complete_or_release_new_claim(self) -> None:
        service = self.make_service()
        _state, first_token = await service.claim(303, include_token=True)
        self.assertTrue(await service.release(303, claim_token=first_token))

        _state, second_token = await service.claim(303, include_token=True)

        self.assertNotEqual(first_token, second_token)
        self.assertFalse(await service.complete(303, claim_token=first_token))
        self.assertFalse(await service.release(303, claim_token=first_token))
        self.assertTrue(await service.complete(303, claim_token=second_token))

    async def test_invalid_ttl_configuration_is_safely_bounded(self) -> None:
        service = WebhookUpdateDeduplicator(
            environ={
                "WEBHOOK_REPLAY_TTL_S": "invalid",
                "WEBHOOK_PROCESSING_TTL_S": "999999",
            }
        )

        self.assertEqual(600, service.replay_ttl_seconds())
        self.assertEqual(600, service.processing_ttl_seconds())
        self.assertEqual("invalid", await service.claim("not-an-update"))
