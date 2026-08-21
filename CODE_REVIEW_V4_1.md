# Code Review V4.1 — Telegram extraction

## Result

The Telegram handler layer was a high-value remaining ownership problem in `app/legacy.py`: routing, commands, callbacks, media processing, guards, and error handling were colocated with database/cache/admin compatibility code. V4.1 extracts the live handler layer while retaining a narrow compatibility bridge for shared helpers that have not moved yet.

## Improvements

### 1. Handler ownership is explicit

`routing.py` is now the single registration table for Telegram handlers. Handler priority and callback regex routing are no longer buried near the end of the monolith.

### 2. User interaction logic is grouped by responsibility

Commands, callbacks, media, and guards are separated. This makes future changes safer because editing OCR/voice behavior no longer requires touching the command/admin dispatcher section of `legacy.py`.

### 3. Backward compatibility remains controlled

Old imports such as `app.legacy.on_photo` still work through four-line lazy wrappers. New runtime registration uses the extracted implementations directly.

### 4. Removed dead worker-era paths

The extracted media layer contained unreachable `if False` branches left behind when Redis/job workers were removed. Those branches referenced deleted `app.services.jobs` modules and are now removed entirely.

## Remaining debt

The extracted handlers still call many legacy helper functions. The temporary `_legacy_runtime.py` bridge is therefore dynamic, and the extracted modules carry a transitional Ruff `F821` suppression because those names are bound at runtime. This is acceptable as a migration seam, not as the desired final design.

The next code-quality milestone should replace dynamic helper binding with explicit injected services such as `TelegramContext`, `PreferenceService`, `BroadcastService`, `MediaService`, and `AdminChatService`. Once that is done, normal runtime code should have no import path back to `app.legacy`.

## Risk assessment

**Low-to-moderate migration risk.** Handler bodies were moved without algorithmic rewrites, registration order was preserved, legacy public names remain compatible, dead queue branches were removed, and focused architecture/regression tests pass.
