#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# Bot Voice - Automated Linux / VPS Deployment Script
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_DIR"

INSTALL_SERVICE=false
for arg in "$@"; do
    if [ "$arg" == "--service" ] || [ "$arg" == "--systemd" ]; then
        INSTALL_SERVICE=true
    fi
done

echo "====================================================="
echo " 🤖 Bot Voice - Automated Server Deployment"
echo "====================================================="

# 1. Check if .env exists
if [ ! -f ".env" ]; then
    if [ -f ".env.example" ]; then
        echo "⚠️  No .env file found. Creating from .env.example..."
        cp .env.example .env
        echo "📝 Please edit .env with your credentials and re-run ./deploy/deploy.sh:"
        echo "   - TELEGRAM_BOT_TOKEN"
        echo "   - ADMIN_IDS"
        echo "   - SUPABASE_URL"
        echo "   - SUPABASE_SERVICE_ROLE_KEY"
        echo "   - GEMINI_API_KEY"
        exit 1
    else
        echo "❌ Error: .env or .env.example not found."
        exit 1
    fi
fi

# Ensure runtime directories exist
mkdir -p data/cache data/exports data/temp logs

# 2. Prefer Docker Compose if Docker is installed (unless explicit systemd requested)
if [ "$INSTALL_SERVICE" = false ] && command -v docker >/dev/null 2>&1 && docker compose version >/dev/null 2>&1; then
    echo "🐳 Docker & Docker Compose detected."
    COMPOSE_FILE="docker/docker-compose.yml"
    if [ ! -f "$COMPOSE_FILE" ]; then
        COMPOSE_FILE="docker-compose.yml"
    fi
    docker compose -f "$COMPOSE_FILE" down --remove-orphans 2>/dev/null || true
    docker compose -f "$COMPOSE_FILE" up -d --build
    echo ""
    echo "✅ Bot is running in Docker!"
    echo "📊 View logs with:   docker compose -f $COMPOSE_FILE logs -f"
    echo "🛑 Stop bot with:    docker compose -f $COMPOSE_FILE down"
    exit 0
fi

# 3. Native Python / System Fallback
echo "🐍 Deploying via native Python environment..."

if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
else
    echo "❌ Error: Python 3.11 or newer is required."
    exit 1
fi

if ! command -v ffmpeg >/dev/null 2>&1; then
    echo "⚠️  FFmpeg is required for voice conversion."
    echo "   Install on Ubuntu/Debian: sudo apt-get update && sudo apt-get install -y ffmpeg libopus-dev"
    echo "   Install on CentOS/RHEL:   sudo dnf install -y ffmpeg"
fi

if [ ! -d ".venv" ]; then
    echo "📦 Creating virtual environment (.venv)..."
    "$PYTHON_BIN" -m venv .venv
fi

source .venv/bin/activate

echo "📦 Installing / updating dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# 4. Install Systemd Service if requested
if [ "$INSTALL_SERVICE" = true ]; then
    echo "🛡️ Configuring Linux Systemd Service (bot-voice.service)..."
    if [ "$EUID" -ne 0 ]; then
        SUDO_CMD="sudo"
    else
        SUDO_CMD=""
    fi

    TARGET_SERVICE="/etc/systemd/system/bot-voice.service"
    SERVICE_SOURCE="$SCRIPT_DIR/deploy/systemd/bot-voice.service"
    if [ ! -f "$SERVICE_SOURCE" ]; then
        SERVICE_SOURCE="$SCRIPT_DIR/bot-voice.service"
    fi

    $SUDO_CMD cp "$SERVICE_SOURCE" "$TARGET_SERVICE"
    if [ "$SCRIPT_DIR" != "/opt/bot-voice" ]; then
        $SUDO_CMD sed -i "s|/opt/bot-voice|$SCRIPT_DIR|g" "$TARGET_SERVICE"
    fi

    $SUDO_CMD systemctl daemon-reload
    $SUDO_CMD systemctl enable --now bot-voice
    echo "✅ Systemd service installed and started!"
    echo "📊 Status:  $SUDO_CMD systemctl status bot-voice"
    echo "📜 Logs:    $SUDO_CMD journalctl -u bot-voice -f"
    exit 0
fi

echo "🚀 Starting Bot Voice..."
chmod +x start.sh
exec ./start.sh
