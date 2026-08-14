from __future__ import annotations

import unittest
from pathlib import Path


class AdminMiniAppUiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        root = Path(__file__).resolve().parents[1]
        cls.html = (root / "static" / "admin" / "index.html").read_text(
            encoding="utf-8"
        )
        cls.css = (root / "static" / "admin" / "styles.css").read_text(
            encoding="utf-8"
        )
        cls.javascript = (root / "static" / "admin" / "app.js").read_text(
            encoding="utf-8"
        )
        cls.main = (root / "app" / "main.py").read_text(encoding="utf-8")

    def test_google_sans_and_khmer_fallback_are_loaded(self) -> None:
        self.assertIn("family=Google+Sans+Flex", self.html)
        self.assertIn("family=Noto+Sans+Khmer", self.html)
        self.assertIn('crossorigin', self.html)
        self.assertIn('styles.css?v=9', self.html)
        self.assertIn('app.js?v=10', self.html)
        self.assertIn('--font-sans: "Google Sans Flex"', self.css)
        self.assertIn('"Noto Sans Khmer"', self.css)

    def test_font_hosts_are_narrowly_allowed_by_csp(self) -> None:
        self.assertIn(
            '"style-src \'self\' https://fonts.googleapis.com; "',
            self.main,
        )
        self.assertIn(
            '"font-src \'self\' https://fonts.gstatic.com; "',
            self.main,
        )
        self.assertNotIn("font-src *", self.main)

    def test_small_telegram_viewports_have_dedicated_layouts(self) -> None:
        for breakpoint in (700, 420, 350):
            self.assertIn(f"@media (max-width: {breakpoint}px)", self.css)
        self.assertIn("env(safe-area-inset-bottom)", self.css)
        self.assertIn("touch-action: manipulation", self.css)
        self.assertIn("-webkit-text-size-adjust: 100%", self.css)

    def test_upgraded_monitor_controls_and_trends_are_present(self) -> None:
        for element_id in (
            "monitorInterval",
            "monitorAlert",
            "monitorCpuChart",
            "monitorPressureChart",
            "monitorRequestsChart",
            "monitorTtsTrendChart",
            "monitorQueueAgeChart",
            "monitorFailureChart",
            "monitorIncidentList",
            "monitorWorkloadType",
            "monitorCopyLogsButton",
            "monitorDownloadLogsButton",
        ):
            self.assertIn(f'id="{element_id}"', self.html)
        self.assertIn('data-tts-filter="running"', self.html)
        self.assertIn("scheduleMonitorPolling()", self.javascript)
        self.assertIn("updateMonitorTrends(payload)", self.javascript)
        self.assertIn("copyMonitorLogs", self.javascript)
        self.assertIn("downloadMonitorLogs", self.javascript)
        self.assertIn("renderMonitorIncidents", self.javascript)
        self.assertIn("navigationObserver", self.javascript)

    def test_redis_disabled_queue_mode_is_rendered_as_expected(self) -> None:
        self.assertIn('redis.status === "disabled"', self.javascript)
        self.assertIn("queueMode.durable === false", self.javascript)
        self.assertIn('t("processLocal")', self.javascript)


if __name__ == "__main__":
    unittest.main()
