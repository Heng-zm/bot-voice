#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# Bot Voice - Automated Linux / VPS Deployment Script
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "====================================================="
echo " 🤖 Bot Voice - Automated Server Deployment"
echo "====================================================="

# 1. Check if .env exists
if [ ! -f ".env" ]; then
    if [ -f ".env.example" ]; then
        echo "⚠️  No .env file found. Creating from .env.example..."
        cp .env.example .env
        echo "📝 Please edit .env with your credentials and re-run ./deploy.sh:"
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

# 2. Prefer Docker Compose if Docker is installed
if command -v docker >/dev/null 2>&1 && docker compose version >/dev/null 2>&1; then
    echo "🐳 Docker & Docker Compose detected."
    echo "🚀 Building and starting container in background..."
    docker compose down --remove-orphans 2>/dev/null || true
    docker compose up -d --build
    echo ""
    echo "✅ Bot is running in Docker!"
    echo "📊 View logs with:   docker compose logs -f"
    echo "🛑 Stop bot with:    docker compose down"
    exit 0
fi

# 3. Native Python / System Fallback
echo "🐍 Docker not found. Deploying via native Python environment..."

# Check Python 3.11+
if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
else
    echo "❌ Error: Python 3.11 or newer is required."
    exit 1
fi

# Check FFmpeg
if ! command -v ffmpeg >/dev/null 2>&1; then
    echo "⚠️  FFmpeg is required for voice conversion."
    echo "   Install on Ubuntu/Debian: sudo apt-get update && sudo apt-get install -y ffmpeg libopus-dev"
    echo "   Install on CentOS/RHEL:   sudo dnf install -y ffmpeg"
fi

# Setup virtual environment if missing
if [ ! -d ".venv" ]; then
    echo "📦 Creating virtual environment (.venv)..."
    "$PYTHON_BIN" -m venv .venv
fi

# Activate virtualenv
# shellcheck source=/dev/null
source .venv/bin/activate

echo "📦 Installing / updating dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo "🚀 Starting Bot Voice..."
chmod +x start.sh
exec ./start.sh
