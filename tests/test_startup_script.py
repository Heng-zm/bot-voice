from __future__ import annotations

import os
import shutil
import subprocess
import unittest
from pathlib import Path


class StartupScriptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.script = Path(__file__).resolve().parents[1] / "start.sh"
        cls.text = cls.script.read_text(encoding="utf-8")

    def test_auto_update_is_wispbyte_scoped_and_fast_forward_only(self) -> None:
        self.assertIn('if [ -n "$SERVER_PORT" ] || [ -n "$WISPBYTE_PORT" ]', self.text)
        self.assertIn('git merge --ff-only --quiet "$REMOTE_REF"', self.text)
        self.assertIn("git status --porcelain --untracked-files=no", self.text)
        self.assertNotIn("git reset", self.text)

    def test_dependency_lock_changes_are_installed_before_start(self) -> None:
        install = "python3 -m pip install --disable-pip-version-check -r requirements.lock"
        self.assertIn(install, self.text)
        self.assertLess(self.text.index(install), self.text.index("python3 main.py"))

    def test_bash_syntax(self) -> None:
        if os.name == "nt":
            self.skipTest("CI validates Bash syntax on Linux")
        bash = shutil.which("bash")
        if not bash:
            self.skipTest("bash is not installed on this test host")
        result = subprocess.run(  # noqa: S603 - resolved system bash, fixed arguments
            [bash, "-n", str(self.script)],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
