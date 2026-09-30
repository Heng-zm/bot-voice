#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# Bot Voice - Automated Healthcheck Script
# ==============================================================================

PORT="${PORT:-8080}"
HOST="${HOST:-127.0.0.1}"
HEALTH_ENDPOINT="http://${HOST}:${PORT}/health"

echo "🔍 Checking Bot Voice health on ${HEALTH_ENDPOINT}..."

if command -v curl >/dev/null 2>&1; then
    RESPONSE=$(curl -s -w "\n%{http_code}" --max-time 5 "${HEALTH_ENDPOINT}" 2>&1 || true)
    HTTP_CODE=$(echo "$RESPONSE" | tail -n1)
    BODY=$(echo "$RESPONSE" | head -n -1)
    
    if [ "$HTTP_CODE" = "200" ]; then
        echo "✅ Bot Voice is healthy! (HTTP 200)"
        echo "$BODY"
        exit 0
    else
        echo "❌ Bot Voice health check failed! (HTTP ${HTTP_CODE})"
        echo "$BODY"
        exit 1
    fi
else
    echo "⚠️ curl is not available; falling back to python check..."
    python3 -c "
import urllib.request, sys
try:
    with urllib.request.urlopen('${HEALTH_ENDPOINT}', timeout=5) as res:
        if res.status == 200:
            print('✅ Bot Voice is healthy!')
            sys.exit(0)
        sys.exit(1)
except Exception as e:
    print(f'❌ Health check failed: {e}')
    sys.exit(1)
"
fi
