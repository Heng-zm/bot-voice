# v4.2 — single-process optimization and resilience

## Scope

V4.2 optimizes the v4/v4.1 single-process bot after the live Telegram handlers were extracted from `app/legacy.py`. This release focuses on overload protection, webhook correctness, supervisor recovery, legacy-bridge overhead, and runtime visibility rather than introducing another large user-facing subsystem.

## 1. Native webhook replay protection

`app/services/telegram/deduplication.py` now owns webhook replay state directly. The previous Redis-era implementation embedded in `legacy.py` was removed from the supported webhook path.

The new store provides:

- bounded process-local replay storage
- a short processing lease (`WEBHOOK_PROCESSING_TTL_S`, default 120s)
- a longer completed-update replay window (`WEBHOOK_REPLAY_TTL_S`, default 600s)
- lease ownership tokens so a stale handler cannot release/complete a lease reclaimed by a newer handler
- expiration/reclaim metrics and bounded entry count

### Bug fixed

The old memory fallback effectively used the replay TTL for in-progress work. A cancelled/crashed handler could therefore keep a Telegram update in `processing` for roughly the full duplicate window. V4.2 separates the two TTLs so interrupted work becomes eligible for retry much sooner.

### Deployment boundary

Replay state is process-local by design because the supported topology is one combined application instance. Do not horizontally scale this version to multiple active replicas without first replacing replay/leader coordination with a shared database-backed mechanism.

## 2. Bounded expensive Telegram workloads

New `app/services/telegram/workloads.py` adds admission control for:

- image OCR
- voice/audio transcription
- uploaded-audio conversion

Defaults:

```env
TELEGRAM_OCR_MAX_CONCURRENT=2
TELEGRAM_TRANSCRIBE_MAX_CONCURRENT=2
TELEGRAM_AUDIO_MAX_CONCURRENT=2
TELEGRAM_WORKLOAD_QUEUE_TIMEOUT_S=6
```

A short bounded wait is used instead of unbounded task accumulation. When capacity remains saturated, the user receives a retry-later response and the `busy_rejected` metric increases.

Uploaded audio is handled resiliently: if voice conversion is saturated but transcription is enabled, transcription can still continue; likewise a successful voice conversion can still be returned if transcription is saturated.

TTS is not double-limited here because the existing runtime already has TTS reservations/semaphores and per-user locking.

## 3. Lower legacy compatibility overhead

`app/services/telegram/_legacy_runtime.py` previously built its dependency list from `code.co_names`. Python includes attribute/method names in that tuple, so names such as `strip`, `reply_text`, and `id` could trigger pointless `getattr(app.legacy, ...)` attempts on every handler call.

V4.2 inspects bytecode and records only actual `LOAD_GLOBAL` names, recursively including nested code objects. The transitional bridge stays compatible while doing less work per Telegram update.

## 4. Combined runtime supervisor fix

`app/main.py` now distinguishes critical services from auxiliary tasks. FastAPI and Telegram are supervised with `asyncio.FIRST_COMPLETED`.

Previously `FIRST_EXCEPTION` could leave the combined process half-alive if one critical service returned normally without raising. V4.2 treats completion of either critical service as an unexpected stop, cleans up the remaining tasks, and lets the existing process restart policy recover the whole runtime.

## 5. Admin/runtime visibility

The Admin Mini App now shows per-workload:

- configured capacity
- active work
- waiting work
- rejected requests

It also displays webhook replay state. The Telegram `/runtime` panel includes the same operational pressure summary.

A CSS regression was also corrected: a late `.metric-grid` override forced three columns even though the dashboard contains four overview metrics. The desktop override is now four columns and remains responsive on smaller screens.

## 6. WEB_KEY/admin migration bug fix

The Telegram Admin `WEB_KEY` action still contained Redis-era behavior after Redis removal. It attempted to store a generated session key in Redis and instructed administrators to configure `REDIS_URL`, so the action was effectively broken in v4/v4.1.

V4.2 changes the action to:

- report the currently active session-secret source/fingerprint
- generate a strong explicit `WEB_SECRET_KEY` candidate
- deliver the raw candidate only to the requesting administrator's private Telegram chat
- instruct the administrator to set `WEB_SECRET_KEY` in the backend environment and restart the one app service
- never claim the candidate is active or persisted before restart

Startup self-check and legacy environment guidance were also corrected so they no longer recommend Redis for session persistence.

## 7. Environment additions

See `.env.example` for the workload and webhook replay controls. Invalid/out-of-range values are clamped to safe bounds.

## Validation

In the available review container:

- focused/non-Telegram-SDK pytest: **37 passed + 5 subtests**
- workload/replay/extraction subset: passed
- Python compilation checks: passed
- Admin Mini App `node --check`: passed during validation

The review container does not currently have `python-telegram-bot` installed. Full tests importing `app.legacy` through the Telegram SDK therefore cannot be collected in this container. After applying to the real repository, install `requirements.txt` and run the complete suite.

## Next migration

The highest-value V4.3 target remains removing the transitional handler bridge by extracting helper/state ownership from `legacy.py`: preferences, text/history cache, media helper services, broadcast/schedule state, admin chat sessions, and Telegram lifecycle helpers.
