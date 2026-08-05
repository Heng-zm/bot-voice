from __future__ import annotations

import unittest
from pathlib import Path


class LegacyQueueMigrationTests(unittest.TestCase):
    def test_ocr_and_transcription_call_sites_enqueue_durable_jobs(self) -> None:
        source = (
            Path(__file__).resolve().parents[1] / "app" / "legacy.py"
        ).read_text(encoding="utf-8")
        self.assertIn('DURABLE_OCR_ENABLED", True', source)
        self.assertIn("submit_ocr_job", source)
        self.assertIn('DURABLE_TRANSCRIPTION_ENABLED", True', source)
        self.assertGreaterEqual(source.count("submit_transcription_job"), 3)

    def test_dedicated_worker_entrypoint_exists(self) -> None:
        worker = Path(__file__).resolve().parents[1] / "app" / "worker.py"
        source = worker.read_text(encoding="utf-8")
        self.assertIn('role="worker"', source)
        self.assertIn("python -m app.worker", source)


if __name__ == "__main__":
    unittest.main()
