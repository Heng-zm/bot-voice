from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from app.services.ai.language import _detect_lang, _language_display
from app.utils.file_io import _read_file_bytes_async, _write_file_bytes_sync
from app.utils.time import _fmt_local_dt, _local_to_utc, _to_local_time


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


if __name__ == "__main__":
    unittest.main()
