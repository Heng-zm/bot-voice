from __future__ import annotations

import tempfile
import time
import unittest
from datetime import datetime, timezone
from pathlib import Path

from app.services.ai.language import _detect_lang, _language_display
from app.services.jobs.queue import RedisJobQueue
from app.services.jobs.runtime import (
    job_worker_snapshot,
    job_workers_accepting,
    set_job_workers_accepting,
)
from app.utils.file_io import _read_file_bytes_async, _write_file_bytes_sync
from app.utils.time import _fmt_local_dt, _local_to_utc, _to_local_time


class ProgressRedis:
    def __init__(self) -> None:
        self.hashes: dict[str, dict[str, str]] = {}
        self.zsets: dict[str, dict[str, float]] = {}

    def hgetall(self, key: str):
        return dict(self.hashes.get(key, {}))

    def hset(self, key: str, *, mapping: dict[str, str]):
        self.hashes.setdefault(key, {}).update(mapping)
        return len(mapping)

    def zrevrange(self, key: str, start: int, end: int):
        ordered = [
            member
            for member, _score in sorted(
                self.zsets.get(key, {}).items(),
                key=lambda item: item[1],
                reverse=True,
            )
        ]
        return ordered[start : end + 1]

    def zremrangebyscore(self, key: str, minimum, maximum):
        del minimum
        cutoff = float(maximum)
        members = self.zsets.setdefault(key, {})
        removed = [member for member, score in members.items() if score <= cutoff]
        for member in removed:
            members.pop(member, None)
        return len(removed)

    def zcard(self, key: str) -> int:
        return len(self.zsets.get(key, {}))


class ExtractedUtilityTests(unittest.IsolatedAsyncioTestCase):
    async def test_atomic_file_helpers_and_limit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.bin"
            _write_file_bytes_sync(str(path), b"hello")
            self.assertEqual(b"hello", await _read_file_bytes_async(str(path)))
            with self.assertRaisesRegex(ValueError, "File too large"):
                await _read_file_bytes_async(str(path), max_bytes=4)

    def test_language_and_timezone_helpers_no_longer_need_legacy(self) -> None:
        self.assertEqual("km", _detect_lang("សួស្តី"))
        self.assertEqual("ar", _detect_lang("مرحبا"))
        self.assertEqual(("🇰🇭", "Khmer"), _language_display("km"))
        utc_value = datetime(2026, 8, 5, 4, 0, tzinfo=timezone.utc)
        local_value = _to_local_time(utc_value)
        self.assertEqual(11, local_value.hour)
        self.assertEqual(utc_value, _local_to_utc(local_value))
        self.assertIn("ICT", _fmt_local_dt(utc_value))


class DurableRuntimeV2Tests(unittest.IsolatedAsyncioTestCase):
    async def test_running_job_progress_is_persisted(self) -> None:
        redis = ProgressRedis()
        queue = RedisJobQueue(redis, redis_prefix="tests")
        job_id = "job-progress"
        redis.hashes[queue._job_key(job_id)] = {
            "id": job_id,
            "type": "tts",
            "payload": "{}",
            "state": "running",
            "priority": "0",
            "created_at": "1",
            "available_at": "1",
            "attempts": "1",
            "max_attempts": "3",
            "timeout_seconds": "60",
            "cancel_requested": "0",
            "lease_token": "lease",
        }

        changed = await queue.update_progress(
            job_id,
            "lease",
            percent=45,
            stage="generating_voice",
            detail="model=edge",
        )

        self.assertTrue(changed)
        job = await queue.get(job_id)
        self.assertEqual(45, job.progress_percent)
        self.assertEqual("generating_voice", job.progress_stage)
        self.assertEqual("model=edge", job.progress_detail)
        self.assertIsNotNone(job.updated_at)

    async def test_terminal_state_indexes_are_listable_and_counted(self) -> None:
        redis = ProgressRedis()
        queue = RedisJobQueue(redis, redis_prefix="tests")
        now = time.time()
        for state, key, score in (
            ("succeeded", queue.succeeded_key, now),
            ("cancelled", queue.cancelled_key, now + 1),
        ):
            job_id = f"{state}-job"
            redis.hashes[queue._job_key(job_id)] = {
                "id": job_id,
                "type": "ocr",
                "payload": "{}",
                "state": state,
                "priority": "0",
                "created_at": "1",
                "available_at": "1",
                "attempts": "1",
                "max_attempts": "3",
                "timeout_seconds": "60",
                "cancel_requested": "1" if state == "cancelled" else "0",
                "completed_at": str(score),
                "progress_percent": "100" if state == "succeeded" else "25",
                "progress_stage": state,
            }
            redis.zsets.setdefault(key, {})[job_id] = score

        succeeded, _ = await queue.list_jobs(state="succeeded")
        cancelled, _ = await queue.list_jobs(state="cancelled")
        self.assertEqual(["succeeded-job"], [job.id for job in succeeded])
        self.assertEqual(["cancelled-job"], [job.id for job in cancelled])
        stats = await queue.stats()
        self.assertEqual(1, stats["succeeded"])
        self.assertEqual(1, stats["cancelled"])

    def test_worker_drain_and_resume_are_visible(self) -> None:
        original = job_workers_accepting()
        try:
            self.assertFalse(set_job_workers_accepting(False))
            self.assertFalse(job_worker_snapshot()["accepting"])
            self.assertTrue(set_job_workers_accepting(True))
            self.assertTrue(job_worker_snapshot()["accepting"])
        finally:
            set_job_workers_accepting(original)


if __name__ == "__main__":
    unittest.main()
