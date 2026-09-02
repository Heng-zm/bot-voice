#!/usr/bin/env bash
set -e

# ==============================================================================
# 🚀 BOT VOICE — SERVER LAUNCH SCRIPT
# ==============================================================================

PORT="${PORT:-8080}"
HOST="${HOST:-0.0.0.0}"

echo "Starting Telegram Bot Voice & AI Suite on http://${HOST}:${PORT}..."

# Execute Uvicorn server running FastAPI + Python Telegram Bot
exec uvicorn app.main:app --host "${HOST}" --port "${PORT}" --workers 1
