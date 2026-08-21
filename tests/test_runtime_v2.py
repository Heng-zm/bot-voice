from __future__ import annotations

import ast
import tempfile
import unittest
from datetime import UTC, datetime
from pathlib import Path

from app.services.ai.language import _detect_lang, _language_display
from app.utils.file_io import _read_file_bytes_async, _write_file_bytes_sync
from app.utils.time import _fmt_local_dt, _local_to_utc, _to_local_time


class ExtractedUtilityTests(unittest.IsolatedAsyncioTestCase):
    def test_python_sources_parse_with_deployment_python_311_grammar(self) -> None:
        root = Path(__file__).resolve().parents[1]
        sources = [
            *root.joinpath("app").rglob("*.py"),
            *root.joinpath("tests").rglob("*.py"),
        ]
        for source in sources:
            with self.subTest(source=source.relative_to(root)):
                ast.parse(
                    source.read_text(encoding="utf-8"),
                    filename=str(source),
                    feature_version=(3, 11),
                )

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
        utc_value = datetime(2026, 8, 5, 4, 0, tzinfo=UTC)
        local_value = _to_local_time(utc_value)
        self.assertEqual(11, local_value.hour)
        self.assertEqual(utc_value, _local_to_utc(local_value))
        self.assertIn("ICT", _fmt_local_dt(utc_value))


if __name__ == "__main__":
    unittest.main()
