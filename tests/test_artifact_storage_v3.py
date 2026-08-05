from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from app.services.artifacts.storage import (
    DEFAULT_BOT_ARTIFACT_LOCAL_DIRECTORY,
    DEFAULT_BOT_ARTIFACT_MAX_BYTES,
    DEFAULT_BOT_ARTIFACT_STORAGE_BUCKET,
    DEFAULT_BOT_ARTIFACT_STORAGE_MODE,
    ArtifactStorageError,
    LocalArtifactStore,
)


class LocalArtifactStoreTests(unittest.IsolatedAsyncioTestCase):
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


if __name__ == "__main__":
    unittest.main()
