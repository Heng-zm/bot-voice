from __future__ import annotations

import logging
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from app import legacy
from app.utils.file_io import _write_file_bytes_sync


class AtomicFileWriteTests(unittest.TestCase):
    def test_binary_write_replaces_destination(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory, "audio.bin")
            destination.write_bytes(b"old")

            _write_file_bytes_sync(str(destination), b"new")

            self.assertEqual(b"new", destination.read_bytes())
            self.assertEqual([], list(Path(directory).glob("*.tmp")))

    def test_failed_replace_preserves_existing_file_and_cleans_temp(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory, "audio.bin")
            destination.write_bytes(b"old")

            with patch.object(legacy.os, "replace", side_effect=OSError("replace failed")):
                with self.assertRaisesRegex(OSError, "replace failed"):
                    _write_file_bytes_sync(str(destination), b"new")

            self.assertEqual(b"old", destination.read_bytes())
            leftovers = [path for path in Path(directory).iterdir() if path != destination]
            self.assertEqual([], leftovers)


class RetryTests(unittest.IsolatedAsyncioTestCase):
    async def test_timed_out_worker_is_not_started_again(self) -> None:
        calls = 0

        def slow_operation() -> str:
            nonlocal calls
            calls += 1
            time.sleep(0.05)
            return "late"

        with patch.object(legacy.logger, "log"):
            result = await legacy.retry_call(
                "test-timeout",
                slow_operation,
                default="fallback",
                attempts=3,
                timeout=0.005,
                breaker=None,
            )

        self.assertEqual("fallback", result)
        self.assertEqual(1, calls)


class BoundedLogDeduplicationTests(unittest.TestCase):
    def test_log_deduplication_cache_is_bounded(self) -> None:
        with legacy._LOG_ONCE_LOCK:
            original = dict(legacy._log_once_seen)
            legacy._log_once_seen.clear()
        try:
            with (
                patch.object(legacy, "_LOG_ONCE_MAX_ENTRIES", 3),
                patch.object(legacy, "_LOG_ONCE_TTL_S", 3600.0),
                patch.object(legacy.logger, "log"),
            ):
                for index in range(8):
                    legacy._log_once(logging.WARNING, f"unique-{index}", "message")

            self.assertLessEqual(len(legacy._log_once_seen), 3)
        finally:
            with legacy._LOG_ONCE_LOCK:
                legacy._log_once_seen.clear()
                legacy._log_once_seen.update(original)


class LazyOptionalDependencyTests(unittest.TestCase):
    def test_heavy_sdks_are_not_loaded_during_module_import(self) -> None:
        # Client initialization is explicit in app startup; importing ASGI
        # routes and pure helpers should not import multi-second SDK trees.
        self.assertFalse(legacy._GENAI_IMPORT_ATTEMPTED)
        self.assertFalse(legacy._SUPABASE_IMPORT_ATTEMPTED)
        self.assertFalse(legacy._HUGGINGFACE_IMPORT_ATTEMPTED)


class ContainerSecurityTests(unittest.TestCase):
    def test_environment_files_are_excluded_from_build_context(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        patterns = (project_root / ".dockerignore").read_text(encoding="utf-8").splitlines()
        self.assertIn(".env", patterns)
        self.assertIn(".env.*", patterns)

    def test_container_drops_root_privileges(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        dockerfile = (project_root / "Dockerfile").read_text(encoding="utf-8")
        self.assertIn("USER appuser", dockerfile)


if __name__ == "__main__":
    unittest.main()
