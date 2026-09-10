"""Unit tests for Audio Response Caching and Deterministic SHA-256 Deduplication."""

from __future__ import annotations

import asyncio
import time
import unicodedata
import unittest

from app.services.tts.cache import (
    TTSAudioCache,
    TTSFileIdCache,
    TTSSingleFlight,
    make_tts_audio_cache_key,
    normalize_tts_text_for_hash,
)


class TestTTSAudioDeduplication(unittest.TestCase):
    def test_unicode_nfc_normalization_determinism(self) -> None:
        # Decomposed vs Precomposed Unicode representation of Khmer & Latin text
        text_precomposed = "កម្ពុជា"
        text_decomposed = unicodedata.normalize("NFD", text_precomposed)

        # Both representations should normalize to identical text
        norm1 = normalize_tts_text_for_hash(text_precomposed)
        norm2 = normalize_tts_text_for_hash(text_decomposed)
        self.assertEqual(norm1, norm2)

        # Hash keys must be 100% deterministic and identical
        key1 = make_tts_audio_cache_key(text_precomposed, "female", 1.0, "auto")
        key2 = make_tts_audio_cache_key(text_decomposed, "female", 1.0, "auto")
        self.assertEqual(key1, key2)
        self.assertEqual(len(key1), 64)  # SHA-256 hex length

    def test_whitespace_and_newline_normalization(self) -> None:
        t1 = "ជំរាបសួរ   បងប្អូនទាំងអស់គ្នា \n\n\n  សូមស្វាគមន៍"
        t2 = "ជំរាបសួរ បងប្អូនទាំងអស់គ្នា \n\n សូមស្វាគមន៍"
        key1 = make_tts_audio_cache_key(t1, "male", 1.0, "edge")
        key2 = make_tts_audio_cache_key(t2, "male", 1.0, "edge")
        self.assertEqual(key1, key2)

    def test_different_parameters_produce_different_hashes(self) -> None:
        base_text = "ព័ត៌មានទាន់ហេតុការណ៍ថ្ងៃនេះ"
        key_base = make_tts_audio_cache_key(base_text, "female", 1.0, "auto")

        # Different speed
        key_faster = make_tts_audio_cache_key(base_text, "female", 1.5, "auto")
        self.assertNotEqual(key_base, key_faster)

        # Different gender
        key_male = make_tts_audio_cache_key(base_text, "male", 1.0, "auto")
        self.assertNotEqual(key_base, key_male)

        # Different model
        key_gemini = make_tts_audio_cache_key(base_text, "female", 1.0, "gemini")
        self.assertNotEqual(key_base, key_gemini)


class TestTTSFileIdCache(unittest.TestCase):
    def test_set_and_get(self) -> None:
        cache = TTSFileIdCache(max_items=10, ttl_seconds=3600.0)
        key = "test_sha256_hash_12345"
        file_id = "AwACAgIAAxkBAAICZ2X..."

        self.assertIsNone(cache.get(key))
        cache.set(key, file_id)
        self.assertEqual(cache.get(key), file_id)
        self.assertEqual(cache.entry_count, 1)
        self.assertEqual(cache.hits, 1)

    def test_lru_eviction(self) -> None:
        cache = TTSFileIdCache(max_items=3, ttl_seconds=3600.0)
        cache.set("k1", "fid1")
        cache.set("k2", "fid2")
        cache.set("k3", "fid3")
        self.assertEqual(cache.entry_count, 3)

        # Access k1 so k2 becomes the oldest
        self.assertEqual(cache.get("k1"), "fid1")

        # Adding k4 should evict k2
        cache.set("k4", "fid4")
        self.assertEqual(cache.entry_count, 3)
        self.assertIsNone(cache.get("k2"))
        self.assertEqual(cache.get("k1"), "fid1")
        self.assertEqual(cache.get("k3"), "fid3")
        self.assertEqual(cache.get("k4"), "fid4")

    def test_invalidation(self) -> None:
        cache = TTSFileIdCache(max_items=10, ttl_seconds=3600.0)
        cache.set("stale_key", "invalid_file_id")
        self.assertEqual(cache.get("stale_key"), "invalid_file_id")

        self.assertTrue(cache.invalidate("stale_key"))
        self.assertIsNone(cache.get("stale_key"))

    def test_ttl_expiration(self) -> None:
        cache = TTSFileIdCache(max_items=10, ttl_seconds=0.01)
        cache.set("expiring_key", "fid_exp")
        time.sleep(0.02)
        self.assertIsNone(cache.get("expiring_key"))


class TestTTSSingleFlight(unittest.IsolatedAsyncioTestCase):
    async def test_concurrent_coalescing(self) -> None:
        flight = TTSSingleFlight()
        execution_count = 0

        async def expensive_synthesis() -> str:
            nonlocal execution_count
            execution_count += 1
            await asyncio.sleep(0.05)
            return "generated_voice_file_id"

        # Launch 5 concurrent calls for the exact same key
        tasks = [
            flight.execute_or_wait("same_key", expensive_synthesis)
            for _ in range(5)
        ]
        results = await asyncio.gather(*tasks)

        # All 5 callers should receive the exact same result
        for res, _was_leader in results:
            self.assertEqual(res, "generated_voice_file_id")

        # The expensive synthesis should only execute ONCE!
        self.assertEqual(execution_count, 1)

        # Exactly 1 caller should be leader, 4 should be followers
        leaders = sum(1 for _res, was_leader in results if was_leader)
        self.assertEqual(leaders, 1)


class TestTTSAudioCache(unittest.TestCase):
    def test_bytes_cache_and_bounds(self) -> None:
        cache = TTSAudioCache(max_bytes=1000, item_max_bytes=400, ttl_seconds=3600.0)
        data1 = b"A" * 300
        data2 = b"B" * 300
        data3 = b"C" * 300
        oversized = b"X" * 500

        # Oversized item ignored
        cache.set("oversized", oversized)
        self.assertIsNone(cache.get("oversized"))

        # Add data1, data2, data3
        cache.set("d1", data1)
        cache.set("d2", data2)
        cache.set("d3", data3)
        self.assertEqual(cache.current_bytes, 900)

        # Adding data4 (300 bytes) should push over 1000 and evict oldest (d1)
        data4 = b"D" * 300
        cache.set("d4", data4)
        self.assertIsNone(cache.get("d1"))
        self.assertEqual(cache.get("d2"), data2)
        self.assertEqual(cache.get("d3"), data3)
        self.assertEqual(cache.get("d4"), data4)


if __name__ == "__main__":
    unittest.main()
