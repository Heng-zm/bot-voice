from __future__ import annotations

import logging
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from app.api.dependencies import AdminPrincipal
from app.api.v1.admin_runtime import bot_monitor, download_monitor_logs
from app.services import monitoring
from app.services.incidents import reset_incident_state


def _job(job_id: str, state: str, job_type: str = "tts") -> SimpleNamespace:
    return SimpleNamespace(
        id=job_id,
        type=job_type,
        state=state,
        attempts=1,
        max_attempts=3,
        created_at=10.0,
        started_at=11.0 if state == "running" else None,
        worker_id="worker-1",
        progress_percent=50 if state == "running" else 0,
        progress_stage="generating" if state == "running" else "queued",
        progress_detail="voice=edge",
        last_error="",
        updated_at=12.0,
    )


class _Queue:
    async def stats(self) -> dict:
        return {
            "queued": 1,
            "running": 1,
            "queue_limit": 1000,
            "throughput_per_minute": 2.5,
        }

    async def list_jobs(self, *, state: str, limit: int):
        assert limit == (100 if state == "running" else 200)
        return [
            _job(f"tts-{state}", state),
            _job(f"ocr-{state}", state, "ocr"),
            _job(f"transcription-{state}", state, "transcription"),
        ], None


class _ProviderManager:
    def metadata(self) -> dict:
        return {"instance_id": "instance-test"}


class RuntimeMonitoringTests(unittest.TestCase):
    def setUp(self) -> None:
        with monitoring._RUNTIME_LOGS_LOCK:
            monitoring._RUNTIME_LOGS.clear()
        reset_incident_state()

    def test_monitor_log_redacts_credentials_and_supports_filters(self) -> None:
        logger = logging.getLogger("tests.monitoring.tts")
        logger.setLevel(logging.INFO)
        logger.info(
            "TTS started token=top-secret Bearer abcdefghijklmnop "
            "redis://user:password@example.invalid/0 "
            "content=private-user-sentence "
            "https://storage.example/file?X-Amz-Signature=signed-secret "
            "/tg-webhook-abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
        )
        logger.warning("TTS queue is delayed")

        snapshot = monitoring.runtime_log_snapshot(
            limit=10,
            level="WARNING",
            query="tts",
        )

        self.assertEqual(1, snapshot["count"])
        self.assertEqual("WARNING", snapshot["entries"][0]["level"])
        self.assertEqual(1, snapshot["level_counts"]["INFO"])
        self.assertEqual(1, snapshot["level_counts"]["WARNING"])
        all_text = str(monitoring.runtime_log_snapshot(limit=10))
        self.assertNotIn("top-secret", all_text)
        self.assertNotIn("abcdefghijklmnop", all_text)
        self.assertNotIn("user:password@", all_text)
        self.assertNotIn("private-user-sentence", all_text)
        self.assertNotIn("signed-secret", all_text)
        self.assertNotIn("abcdefghijklmnopqrstuvwxyzABC", all_text)
        self.assertIn("<hidden>", all_text)
        self.assertIn("<user-content>", all_text)

    def test_public_url_is_discovered_from_host_environment(self) -> None:
        with patch.dict(
            os.environ,
            {"RAILWAY_PUBLIC_DOMAIN": "voice-bot.example.com"},
            clear=True,
        ):
            public_url = monitoring.discover_public_url()

        self.assertEqual("https://voice-bot.example.com", public_url["url"])
        self.assertEqual("RAILWAY_PUBLIC_DOMAIN", public_url["source"])
        self.assertTrue(public_url["detected"])

    def test_process_snapshot_is_safe_and_local(self) -> None:
        snapshot = monitoring.process_snapshot()

        self.assertGreater(snapshot["pid"], 0)
        self.assertGreaterEqual(snapshot["uptime_seconds"], 0)
        self.assertGreaterEqual(snapshot["threads"], 1)
        self.assertGreaterEqual(snapshot["sampled_at"], snapshot["started_at"])
        self.assertGreaterEqual(snapshot["cpu_seconds"], 0)
        self.assertNotIn("environment", snapshot)
        self.assertNotIn("command_line", snapshot)


class MonitorEndpointTests(unittest.IsolatedAsyncioTestCase):
    async def test_monitor_exposes_tts_progress_without_payload_or_result(self) -> None:
        principal = AdminPrincipal(admin_id=42, auth_method="telegram_init_data")
        legacy_snapshot = {
            "uptime": "2m",
            "active_requests": 1,
            "db_queue_size": 0,
            "metrics": {"tts": 9},
            "tts_slots": {"configured": 2, "available": 1, "in_use": 1},
            "reserved_requests": 1,
        }
        with (
            patch("app.api.v1.admin_runtime.get_job_queue", return_value=_Queue()),
            patch(
                "app.api.v1.admin_runtime.get_provider_manager",
                return_value=_ProviderManager(),
            ),
            patch(
                "app.api.v1.admin_runtime._legacy_monitor_snapshot",
                return_value=legacy_snapshot,
            ),
        ):
            payload = await bot_monitor(principal, 20, "", "")

        self.assertTrue(payload["ok"])
        self.assertEqual("instance-test", payload["process"]["instance_id"])
        self.assertEqual("healthy", payload["health"]["state"])
        self.assertAlmostEqual(0.1, payload["health"]["queue_pressure_percent"])
        self.assertEqual(1, payload["tts"]["running_count"])
        self.assertEqual(1, payload["tts"]["queued_count"])
        self.assertEqual(3, payload["workloads"]["running_count"])
        self.assertEqual(3, payload["workloads"]["queued_count"])
        self.assertEqual(
            {"running": 1, "queued": 1},
            payload["workloads"]["counts_by_type"]["ocr"],
        )
        self.assertEqual("generating", payload["tts"]["running"][0]["progress_stage"])
        self.assertNotIn("payload", payload["tts"]["running"][0])
        self.assertNotIn("result", payload["tts"]["running"][0])

    async def test_log_download_is_bounded_and_redacted(self) -> None:
        principal = AdminPrincipal(admin_id=42, auth_method="telegram_init_data")
        logger = logging.getLogger("tests.monitoring.download")
        logger.setLevel(logging.INFO)
        logger.info("token=download-secret content=private-message")

        response = await download_monitor_logs(principal, 400, "", "")
        body = bytes(response.body).decode("utf-8")

        self.assertIn("attachment;", response.headers["content-disposition"])
        self.assertNotIn("download-secret", body)
        self.assertNotIn("private-message", body)
        self.assertIn("<hidden>", body)
        self.assertIn("<user-content>", body)


if __name__ == "__main__":
    unittest.main()
