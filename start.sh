#!/bin/bash

# Ensure we are in the correct directory
cd "$(dirname "$0")"

# Set default port if not provided by the environment
# Pterodactyl uses SERVER_PORT, while other hosts use PORT.
if [ -z "$PORT" ]; then
    if [ ! -z "$SERVER_PORT" ]; then
        export PORT="$SERVER_PORT"
    else
        export PORT=8080
    fi
fi

# Default to POLLING mode on Pterodactyl to avoid port 400 errors from Telegram
if [ -z "$BOT_MODE" ]; then
    export BOT_MODE="POLLING"
fi

echo "Starting Telegram Bot on port $PORT in $BOT_MODE mode..."

# Check local OCR support if enabled
if [ "$OCR_PROVIDER" = "local" ] || [ "$OCR_PROVIDER" = "tesseract" ]; then
    if command -v tesseract &> /dev/null; then
        echo "Local OCR (Tesseract) is available on this system."
    else
        echo "WARNING: OCR_PROVIDER is set to local/tesseract, but 'tesseract' binary is not found in PATH."
        echo "Please install tesseract-ocr and tesseract-ocr-khm on your server or use Gemini/HF API."
    fi
fi

# Run the application
python3 main.py
