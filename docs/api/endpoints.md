# REST & Telemetry API Documentation

## Endpoints

### 1. Health & Readiness
- `GET /health` / `GET /healthz`
  - Returns `200 OK` with system uptime, memory usage, and component statuses.
  - Response:
    ```json
    {
      "status": "ok",
      "uptime_seconds": 12450.5,
      "memory_mb": 142.1,
      "database": "connected"
    }
    ```

### 2. Telegram Webhook
- `POST /webhook` / `POST /`
  - Receives encrypted/signed updates from Telegram Bot API.
  - Secret token validation via `X-Telegram-Bot-Api-Secret-Token`.

### 3. Realtime Logs & SSE Telemetry
- `GET /logs`
  - Snapshot of recent buffered execution logs.
- `GET /logs/stream`
  - Server-Sent Events (SSE) streaming live application logs in realtime.

### 4. Text-to-Speech API
- `POST /api/tts`
  - Generate speech from text payload.
  - Parameters:
    - `text`: Text string to synthesize (Khmer or multilingual).
    - `gender`: `male` or `female`.
    - `speed`: Playback speed factor (default `1.0`).
    - `model`: TTS engine (`edge`, `hf`, `gemini`, `gtts`, or `auto`).
