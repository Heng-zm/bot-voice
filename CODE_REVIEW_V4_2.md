# Code Review V4.2 — optimization and bug fixes

## Summary

V4.2 addresses four production risks that remained after the single-process migration: webhook retry semantics, unbounded expensive Telegram work, compatibility-bridge overhead, and partial runtime death. It also improves Admin/Telegram observability so overload is visible instead of appearing only as slow requests.

## Fixed issues

### P0/P1 — webhook processing lease could block retry too long

The old no-Redis fallback inherited Redis-era replay behavior and did not cleanly separate "currently processing" from "already completed" retention. Interrupted work could remain blocked for the longer duplicate window.

**Fix:** native `WebhookReplayStore` with distinct processing/completed TTLs and ownership tokens.

### P1 — expensive media operations could pile up in one process

OCR, transcription, and audio conversion were inline after worker removal, but there was no common admission boundary around these handler classes. Traffic spikes could increase memory pressure and provider concurrency simultaneously.

**Fix:** bounded semaphores, bounded wait time, explicit busy rejection, and visible pressure metrics.

### P1 — combined supervisor could remain half-alive

`asyncio.FIRST_EXCEPTION` does not return immediately when a critical task exits successfully while another task keeps running. That is the wrong failure policy for a combined FastAPI + Telegram process.

**Fix:** wait for `FIRST_COMPLETED` among the two critical tasks and restart the complete runtime when either exits unexpectedly.

### P2 — extracted handler bridge performed unnecessary global resolution

`code.co_names` contains attribute names in addition to globals. The compatibility bridge could perform repeated failed legacy lookups for ordinary object attributes/methods.

**Fix:** bytecode-based `LOAD_GLOBAL` dependency discovery.

### P2 — Admin metric layout regression

A later CSS override rendered a three-column desktop grid for four metric cards.

**Fix:** four-column desktop override plus existing responsive breakpoints.

### P1 — Telegram WEB_KEY action still depended on removed Redis

The admin callback generated a web session secret and attempted to persist it through Redis, then instructed the operator to configure `REDIS_URL`. That contradicted the supported v4 architecture and made the feature fail.

**Fix:** generate a private explicit `WEB_SECRET_KEY` candidate, report the active source/fingerprint, and give environment + restart instructions. No secret is falsely reported as persisted. Startup diagnostics were updated to stop treating missing Redis as a configuration problem.

## Architecture result

The supported runtime remains:

- one `python -m app.main` application instance
- FastAPI + Telegram in one process
- no dedicated Redis service
- no dedicated job worker
- Supabase for persistent control-plane/application data
- bounded process-local execution for expensive Telegram operations

## Remaining debt / risks

1. `app/legacy.py` remains a large compatibility monolith. V4.2 removes the webhook replay implementation from it, but many helper/state functions and old Redis-era compatibility branches still exist textually.
2. Extracted handlers still use `_legacy_runtime.py`; this is transitional and should be replaced by explicit services/dependencies.
3. Webhook replay is process-local. Multiple simultaneously active application replicas would not share duplicate state. Horizontal scaling requires a shared coordination design first.
4. Some legacy admin/browser surfaces still contain old architecture wording but are outside the supported Mini App path.
5. Live GitHub cloning was unavailable in the review environment; this updater continues from the V4.1 package and known Library baseline. Apply it to the actual repository and run the full project suite.

## Validation status

The available no-SDK/focused suite passes **37 tests plus 5 subtests**. Static compilation and Admin JavaScript syntax checks pass. The review container is missing `python-telegram-bot`, so Telegram-SDK-dependent tests must be rerun in the real development/deployment environment after installing `requirements.txt`.

## Recommended V4.3

Replace dynamic legacy handler dependencies with explicit native services in this order:

1. preference/history/text-cache service
2. media/OCR/transcription helper service
3. broadcast + schedule service
4. admin chat/session service
5. Telegram polling/webhook lifecycle service
6. remove handler compatibility wrappers and `_legacy_runtime.py`
