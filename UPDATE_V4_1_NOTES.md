# v4.1 — Telegram handler extraction

## Scope

V4.1 continues the staged removal of `app/legacy.py` after the v4 single-process/no-Redis migration. The goal is to move the **live Telegram request/interaction layer** into `app/services/telegram/` without changing user-visible bot behavior.

## Extracted live modules

- `app/services/telegram/commands.py`
  - start/help/preferences/TTS/security/privacy/admin/runtime/API/user-chat commands
  - broadcast/schedule command entry points
- `app/services/telegram/callbacks.py`
  - broadcast callbacks
  - user/history admin callbacks
  - schedule/runtime callbacks
  - generic TTS/media/admin callback dispatcher
- `app/services/telegram/media.py`
  - photo OCR
  - Telegram voice transcription
  - uploaded audio conversion/transcription
  - generic media handling
  - normal text/AI/TTS flow
- `app/services/telegram/guards.py`
  - Telegram rate limit guard
  - user security guard
  - stale update guard
  - global Telegram error handler
- `app/services/telegram/routing.py`
  - owns handler registration and ordering
- `app/services/telegram/_legacy_runtime.py`
  - temporary narrow compatibility bridge for helpers/state not extracted yet

## Migration result

- 38 live Telegram handlers moved out of `legacy.py`.
- `legacy.py` reduced from 28,824 lines in the v4 payload to 26,618 lines in v4.1.
- `_run_bot()` no longer contains the large CommandHandler/CallbackQueryHandler/MessageHandler registration block.
- Legacy handler names remain as small lazy wrappers for backward compatibility.
- Handler ordering is now explicit and testable in `routing.py`.
- Four unreachable Redis/job-queue branches were removed from media processing.
- Extracted media code no longer imports `app.services.jobs`.

## Compatibility boundary

The handler bodies are now native modules, but several shared helpers and mutable caches still live in `app.legacy`. `legacy_bound_handler` refreshes only the legacy names referenced by each extracted function immediately before execution. This avoids copying stale runtime settings while keeping the migration incremental.

This bridge is intentionally temporary. The next extraction should move helper/state ownership into native services so `_legacy_runtime.py` can be deleted.

## Recommended next extraction

1. Telegram preference/text-cache helpers.
2. Broadcast + schedule service objects.
3. Admin chat/session state.
4. OCR/audio/TTS helper dependencies used by `media.py`.
5. Telegram polling/webhook lifecycle and replay protection.
6. Delete compatibility wrappers after call sites/tests import native handlers directly.

## Validation

- `python -m compileall -q app tests` — passed.
- Focused pytest suite — 28 tests passed, plus 5 subtests.
- Extraction architecture tests verify:
  - live handler bodies exist in native modules,
  - legacy copies are wrappers only,
  - `_run_bot()` delegates registration to `routing.py`,
  - media handlers contain no removed job-queue imports,
  - nested runtime dependencies are refreshed by the compatibility bridge.
