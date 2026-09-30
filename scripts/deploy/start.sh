#!/usr/bin/env bash
set -e

# ==============================================================================
# 🚀 BOT VOICE — PRODUCTION SERVER LAUNCH SCRIPT
# ==============================================================================

PORT="${PORT:-8080}"
HOST="${HOST:-0.0.0.0}"

# Ensure immediate unbuffered console streaming for real-time logging
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1

# Ensure runtime directories exist
mkdir -p data logs

echo "🚀 Starting Telegram Bot Voice & AI Suite on http://${HOST}:${PORT}..."

# Execute Uvicorn server running FastAPI + Python Telegram Bot
exec uvicorn app.main:app --host "${HOST}" --port "${PORT}" --workers 1 --lifespan on --access-log
