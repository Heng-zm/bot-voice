import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_start_script_has_deployment_safeguards() -> None:
    script = (ROOT / "start.sh").read_text(encoding="utf-8")

    assert script.startswith("#!/usr/bin/env sh\nset -eu\n")
    assert 'cd "$SCRIPT_DIR"' in script
    assert "PYTHON_BIN" in script
    assert "Python 3.11 or newer is required" in script
    assert "INSTALL_REQUIREMENTS_ON_START" in script
    assert '"$PYTHON_CMD" -m compileall -q app' in script
    assert script.rstrip().endswith('exec "$PYTHON_CMD" -m app.main')


def test_root_launcher_imports_when_started_from_parent_directory() -> None:
    launcher = ROOT / "main.py"
    code = (
        "import importlib.util; "
        f"spec = importlib.util.spec_from_file_location('panel_launcher', {str(launcher)!r}); "
        "module = importlib.util.module_from_spec(spec); "
        "spec.loader.exec_module(module); "
        "assert module.app is not None"
    )
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)

    result = subprocess.run(  # noqa: S603 - fixed interpreter and test program
        [sys.executable, "-c", code],
        cwd=ROOT.parent,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, result.stderr
