#!/usr/bin/env sh
set -eu

# Always run from the repository root, even when a hosting panel invokes this
# script from another working directory.
SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$SCRIPT_DIR"

if [ -n "${PYTHON_BIN:-}" ]; then
    PYTHON_CMD=$PYTHON_BIN
elif command -v python3 >/dev/null 2>&1; then
    PYTHON_CMD=python3
elif command -v python >/dev/null 2>&1; then
    PYTHON_CMD=python
else
    echo "Error: Python 3.11 or newer is required." >&2
    exit 127
fi

export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export PYTHONDONTWRITEBYTECODE="${PYTHONDONTWRITEBYTECODE:-1}"
export PIP_DISABLE_PIP_VERSION_CHECK="${PIP_DISABLE_PIP_VERSION_CHECK:-1}"

"$PYTHON_CMD" -c 'import sys; minimum = (3, 11); current = sys.version_info[:2]; current >= minimum or sys.exit(f"Error: Python 3.11 or newer is required; found {sys.version.split()[0]}.")'

# Dependency installation is opt-in so automatic restarts do not repeatedly
# contact package indexes. Set INSTALL_REQUIREMENTS_ON_START=true when needed.
case "${INSTALL_REQUIREMENTS_ON_START:-false}" in
    1|true|TRUE|yes|YES)
        "$PYTHON_CMD" -m pip install -r requirements.txt
        ;;
esac

# Fail with a clear syntax error before starting the web server. This check can
# be disabled for read-only deployments with STARTUP_COMPILE_CHECK=false.
case "${STARTUP_COMPILE_CHECK:-true}" in
    0|false|FALSE|no|NO)
        ;;
    *)
        "$PYTHON_CMD" -m compileall -q app
        ;;
esac

exec "$PYTHON_CMD" -m app.main
