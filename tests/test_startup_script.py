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
