#!/bin/bash

# Ensure we are in the correct directory
cd "$(dirname "$0")"

# Keep Wispbyte deployments current on every safe restart. The panel's native
# GitHub "Auto-update on startup" option remains the preferred first layer;
# this fallback is useful when the server egg exposes a normal Git checkout.
# Local/non-Wispbyte starts do not pull unless AUTO_UPDATE_ON_START is enabled.
AUTO_UPDATE_DEFAULT="false"
if [ -n "$SERVER_PORT" ] || [ -n "$WISPBYTE_PORT" ]; then
    AUTO_UPDATE_DEFAULT="true"
fi
AUTO_UPDATE_ON_START="${AUTO_UPDATE_ON_START:-$AUTO_UPDATE_DEFAULT}"
AUTO_UPDATE_BRANCH="${AUTO_UPDATE_BRANCH:-main}"

case "$AUTO_UPDATE_ON_START" in
    1|true|TRUE|yes|YES|on|ON)
        if ! [[ "$AUTO_UPDATE_BRANCH" =~ ^[A-Za-z0-9][A-Za-z0-9._/-]*$ ]]; then
            echo "WARNING: AUTO_UPDATE_BRANCH is invalid; automatic update skipped."
        elif [ ! -d .git ] || ! command -v git > /dev/null 2>&1; then
            echo "Automatic update skipped: this startup directory is not a Git checkout."
        elif [ -n "$(git status --porcelain --untracked-files=no 2>/dev/null)" ]; then
            echo "WARNING: Automatic update skipped because tracked server files have local changes."
        elif git fetch --quiet origin "$AUTO_UPDATE_BRANCH"; then
            REMOTE_REF="origin/$AUTO_UPDATE_BRANCH"
            if git merge-base --is-ancestor HEAD "$REMOTE_REF"; then
                OLD_REVISION="$(git rev-parse --short HEAD)"
                OLD_LOCK_REVISION="$(git rev-parse HEAD:requirements.lock 2>/dev/null || true)"
                if git merge --ff-only --quiet "$REMOTE_REF"; then
                    NEW_REVISION="$(git rev-parse --short HEAD)"
                    if [ "$OLD_REVISION" != "$NEW_REVISION" ]; then
                        echo "Automatic update applied: $OLD_REVISION -> $NEW_REVISION"
                        NEW_LOCK_REVISION="$(git rev-parse HEAD:requirements.lock 2>/dev/null || true)"
                        if [ "$OLD_LOCK_REVISION" != "$NEW_LOCK_REVISION" ] && [ -f requirements.lock ]; then
                            echo "Dependency lock changed; installing the tested runtime dependencies..."
                            if ! python3 -m pip install --disable-pip-version-check -r requirements.lock; then
                                echo "ERROR: Dependency installation failed; bot startup stopped."
                                exit 1
                            fi
                        fi
                    else
                        echo "Automatic update: already current ($NEW_REVISION)."
                    fi
                else
                    echo "WARNING: Automatic fast-forward update failed; starting existing version."
                fi
            else
                echo "WARNING: Local and origin/$AUTO_UPDATE_BRANCH histories diverged; automatic update skipped."
            fi
        else
            echo "WARNING: Could not check GitHub for updates; starting existing version."
        fi
        ;;
esac

# Pterodactyl/Wispbyte's allocated SERVER_PORT must override a generic PORT.
if [ -n "$SERVER_PORT" ]; then
    export PORT="$SERVER_PORT"
elif [ -n "$WISPBYTE_PORT" ]; then
    export PORT="$WISPBYTE_PORT"
elif [ -z "$PORT" ]; then
    export PORT=8080
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
