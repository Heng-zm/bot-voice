from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path

from app.services.artifacts.storage import (
    DEFAULT_BOT_ARTIFACT_LOCAL_DIRECTORY,
    DEFAULT_BOT_ARTIFACT_MAX_BYTES,
    DEFAULT_BOT_ARTIFACT_STORAGE_BUCKET,
    DEFAULT_BOT_ARTIFACT_STORAGE_MODE,
    ArtifactNotFound,
    ArtifactService,
    ArtifactStorageError,
    LocalArtifactStore,
    SupabaseArtifactStore,
)


class ExpiryRedis:
    def __init__(self) -> None:
        self.values: dict[str, dict[str, float]] = {}

    def zadd(self, key: str, mapping: dict[str, float]) -> int:
        self.values.setdefault(key, {}).update(mapping)
        return len(mapping)

    def zrangebyscore(
        self,
        key: str,
        minimum,
        maximum,
        *,
        start: int,
        num: int,
    ) -> list[str]:
        del minimum
        due = [
            member
            for member, score in sorted(
                self.values.get(key, {}).items(),
                key=lambda item: item[1],
            )
            if score <= float(maximum)
        ]
        return due[start : start + num]

    def zrem(self, key: str, *members: str) -> int:
        values = self.values.setdefault(key, {})
        removed = 0
        for member in members:
            removed += int(values.pop(member, None) is not None)
        return removed


class LocalArtifactStoreTests(unittest.IsolatedAsyncioTestCase):
    async def test_cleanup_discards_malformed_registry_entries(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            redis = ExpiryRedis()
            service = ArtifactService(
                LocalArtifactStore(directory, max_bytes=1024),
                redis_client=redis,
                redis_prefix="tests",
            )
            redis.values[service.expiry_key] = {"[]": 0.0, "not-json": 0.0}

            result = await service.cleanup_expired(now=1.0)

            self.assertEqual(2, result["checked"])
            self.assertEqual(0, result["errors"])
            self.assertEqual(0, result["remaining_due"])
            self.assertFalse(redis.values[service.expiry_key])

    async def test_cleanup_retains_entry_after_transient_storage_failure(self) -> None:
        class UnavailableLocalStore(LocalArtifactStore):
            async def get_bytes(self, artifact):
                del artifact
                raise ArtifactStorageError("temporary storage failure")

        with tempfile.TemporaryDirectory() as directory:
            redis = ExpiryRedis()
            service = ArtifactService(
                LocalArtifactStore(directory, max_bytes=1024),
                redis_client=redis,
                redis_prefix="tests",
            )
            artifact = await service.put_text(
                job_id="retry-cleanup",
                name="transcript.txt",
                text="keep me",
                ttl_seconds=-1,
            )
            target = Path(directory) / artifact.path
            service.store = UnavailableLocalStore(directory, max_bytes=1024)

            result = await service.cleanup_expired()

            self.assertEqual(1, result["errors"])
            self.assertEqual(1, result["remaining_due"])
            self.assertEqual(1, len(redis.values[service.expiry_key]))
            self.assertTrue(target.is_file())

    async def test_cleanup_does_not_delete_newer_content_at_same_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            redis = ExpiryRedis()
            service = ArtifactService(
                LocalArtifactStore(directory, max_bytes=1024),
                redis_client=redis,
                redis_prefix="tests",
            )
            await service.put_text(
                job_id="retried-job",
                name="transcript.txt",
                text="old",
                ttl_seconds=-1,
            )
            current = await service.put_text(
                job_id="retried-job",
                name="transcript.txt",
                text="new",
                ttl_seconds=3_600,
            )

            result = await service.cleanup_expired()

            self.assertEqual(0, result["deleted"])
            self.assertEqual("new", await service.get_text(current))
            self.assertEqual(1, len(redis.values[service.expiry_key]))

    async def test_expired_artifacts_are_deleted_from_registry_and_disk(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            redis = ExpiryRedis()
            service = ArtifactService(
                LocalArtifactStore(directory, max_bytes=1024),
                redis_client=redis,
                redis_prefix="tests",
            )
            artifact = await service.put_text(
                job_id="expired-job",
                name="transcript.txt",
                text="expired",
                ttl_seconds=-1,
            )
            target = Path(directory) / artifact.path
            self.assertTrue(target.is_file())

            result = await service.cleanup_expired()

            self.assertEqual(1, result["deleted"])
            self.assertFalse(target.exists())
            self.assertFalse(redis.values[service.expiry_key])

    async def test_retry_safe_put_and_integrity_checked_get(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = LocalArtifactStore(directory, max_bytes=1024)
            first = await store.put_bytes(
                "results/job-1/transcript.txt",
                b"hello",
                content_type="text/plain",
            )
            second = await store.put_bytes(
                "results/job-1/transcript.txt",
                b"hello",
                content_type="text/plain",
            )

            self.assertEqual(first.id, second.id)
            self.assertEqual(b"hello", await store.get_bytes(first))

    async def test_rejects_oversized_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = LocalArtifactStore(directory, max_bytes=1024)
            with self.assertRaises(ArtifactStorageError):
                await store.put_bytes(
                    "results/job-1/too-large.bin",
                    b"x" * 1025,
                    content_type="application/octet-stream",
                )

    async def test_path_cannot_escape_root(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = LocalArtifactStore(directory, max_bytes=1024)
            ref = await store.put_bytes(
                "../safe.txt",
                b"safe",
                content_type="text/plain",
            )
            self.assertTrue((Path(directory) / ref.path).is_file())


class ArtifactSourceDefaultTests(unittest.TestCase):
    def test_artifact_defaults_are_defined_in_source(self) -> None:
        self.assertEqual("auto", DEFAULT_BOT_ARTIFACT_STORAGE_MODE)
        self.assertEqual("bot-job-artifacts", DEFAULT_BOT_ARTIFACT_STORAGE_BUCKET)
        self.assertEqual("data/job-artifacts", DEFAULT_BOT_ARTIFACT_LOCAL_DIRECTORY)
        self.assertEqual(52_428_800, DEFAULT_BOT_ARTIFACT_MAX_BYTES)

    def test_supabase_transport_error_is_not_reported_as_missing(self) -> None:
        class Bucket:
            def download(self, path: str):
                del path
                raise RuntimeError("connection reset")

        class Storage:
            def from_(self, bucket: str):
                del bucket
                return Bucket()

        class Client:
            storage = Storage()

        store = SupabaseArtifactStore(Client(), bucket="artifacts", max_bytes=1024)
        artifact = type("Artifact", (), {"path": "results/job/file.txt"})()
        with self.assertRaises(ArtifactStorageError) as raised:
            asyncio.run(store.get_bytes(artifact))
        self.assertNotIsInstance(raised.exception, ArtifactNotFound)


if __name__ == "__main__":
    unittest.main()
