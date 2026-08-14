from __future__ import annotations

import unittest
from unittest.mock import AsyncMock, patch

from app.services.jobs.submission import (
    submit_ocr_job,
    submit_transcription_job,
    submit_tts_job,
)


class SubmissionTests(unittest.IsolatedAsyncioTestCase):
    async def test_removed_tts_model_falls_back_to_standard_tts_queue(self) -> None:
        enqueue = AsyncMock(return_value=(object(), True))
        with patch("app.services.jobs.submission.enqueue_bot_job", enqueue):
            await submit_tts_job(
                chat_id=10,
                user_id=20,
                text="hello",
                tts_model="voxcpm2",
                progress_message_id=30,
                idempotency_key="tts-key",
            )

        self.assertEqual("tts", enqueue.await_args.args[0])
        self.assertEqual("auto", enqueue.await_args.args[1]["tts_model"])
        self.assertEqual(30, enqueue.await_args.args[1]["progress_message_id"])

    async def test_ocr_payload_contains_delivery_target_and_user_context(self) -> None:
        enqueue = AsyncMock(return_value=(object(), True))
        with patch("app.services.jobs.submission.enqueue_bot_job", enqueue):
            await submit_ocr_job(
                chat_id=10,
                user_id=20,
                username="tester",
                file_id="file",
                mime_type="image/jpeg",
                progress_message_id=30,
                reply_to_message_id=40,
                idempotency_key="ocr-key",
            )
        payload = enqueue.await_args.args[1]
        self.assertEqual(30, payload["progress_message_id"])
        self.assertEqual(20, payload["user_id"])
        self.assertEqual("tester", payload["username"])

    async def test_transcription_payload_identifies_audio_file_source(self) -> None:
        enqueue = AsyncMock(return_value=(object(), True))
        with patch("app.services.jobs.submission.enqueue_bot_job", enqueue):
            await submit_transcription_job(
                chat_id=10,
                user_id=20,
                username="tester",
                file_id="file",
                mime_type="audio/mpeg",
                source_kind="audio_file",
                filename="sample.mp3",
                progress_message_id=30,
                idempotency_key="transcription-key",
            )
        payload = enqueue.await_args.args[1]
        self.assertEqual("audio_file", payload["source_kind"])
        self.assertEqual("sample.mp3", payload["filename"])


if __name__ == "__main__":
    unittest.main()
