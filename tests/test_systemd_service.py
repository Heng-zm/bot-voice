"""Unit tests for Linux Systemd Service & Server Deployment Configurations."""

from __future__ import annotations

import configparser
from pathlib import Path
import unittest

_ROOT = Path(__file__).resolve().parent.parent


class SystemdServiceTests(unittest.TestCase):
    def setUp(self):
        deploy_service = _ROOT / "deploy" / "systemd" / "bot-voice.service"
        self.service_path = deploy_service if deploy_service.exists() else (_ROOT / "bot-voice.service")
        self.start_sh_path = _ROOT / "start.sh"
        deploy_sh = _ROOT / "deploy" / "deploy.sh"
        self.deploy_sh_path = deploy_sh if deploy_sh.exists() else (_ROOT / "deploy.sh")

    def test_service_file_exists(self):
        self.assertTrue(self.service_path.exists(), "bot-voice.service must exist")

    def test_service_file_ini_structure(self):
        content = self.service_path.read_text(encoding="utf-8")
        parser = configparser.ConfigParser(strict=False, interpolation=None)
        # configparser requires section headers, which systemd files have
        parser.read_string(content)

        self.assertIn("Unit", parser.sections())
        self.assertIn("Service", parser.sections())
        self.assertIn("Install", parser.sections())

    def test_service_hardening_and_limits(self):
        content = self.service_path.read_text(encoding="utf-8")

        # Security hardening directives
        self.assertIn("NoNewPrivileges=true", content)
        self.assertIn("PrivateTmp=true", content)
        self.assertIn("ProtectSystem=full", content)
        self.assertIn("ProtectHome=read-only", content)

        # High-concurrency resource limits
        self.assertIn("LimitNOFILE=65536", content)
        self.assertIn("LimitNPROC=4096", content)

        # Fault tolerance and auto-restart
        self.assertIn("Restart=always", content)
        self.assertIn("RestartSec=3s", content)

        # Python environment flags
        self.assertIn("Environment=PYTHONUNBUFFERED=1", content)
        self.assertIn("Environment=PYTHONDONTWRITEBYTECODE=1", content)

        # Systemd Journal
        self.assertIn("StandardOutput=journal", content)
        self.assertIn("StandardError=journal", content)

    def test_start_script_unbuffered_and_mkdir(self):
        content = self.start_sh_path.read_text(encoding="utf-8")
        self.assertIn("export PYTHONUNBUFFERED=1", content)
        self.assertIn("mkdir -p data logs", content)
        self.assertIn("uvicorn app.main:app", content)

    def test_deploy_script_service_install_option(self):
        content = self.deploy_sh_path.read_text(encoding="utf-8")
        self.assertIn("--service", content)
        self.assertIn("bot-voice.service", content)
        self.assertIn("systemctl enable --now bot-voice", content)


if __name__ == "__main__":
    unittest.main()
