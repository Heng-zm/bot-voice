# Code Review v4

## Review result

The highest-risk issues were architectural coupling rather than syntax errors. v3.x assumed Redis existed across startup secrets, admin authorization, CORS, queueing, delivery state, and readiness. Removing only the worker would therefore have left multiple 503/crash paths.

## Fixed findings

### P0 — Redis was a hidden startup dependency

**Impact:** removing Redis could break stable session secrets/security bootstrap before normal FastAPI operation.

**Fix:** `app/core/security.py` now resolves explicit secrets first and otherwise derives stable, domain-separated secrets from the Telegram bot token. Webhook state can be persisted through the Supabase settings store.

### P0 — Worker removal would leave queue call sites

**Impact:** OCR/transcription could import deleted job modules at runtime.

**Fix:** supported OCR/transcription paths execute inline. The Redis queue/worker package and worker entry point are deleted by the update installer.

### P1 — CORS middleware could use a stale store

**Impact:** admin CORS changes could be persisted but not applied consistently after startup.

**Fix:** middleware resolves the configured policy store dynamically instead of capturing the import-time store.

### P1 — Telegram Mini App auth fallback was inconsistent

**Impact:** the UI could send `Authorization: Bearer <initData>` but native admin dependencies did not recognize it as Telegram init data, producing confusing 401/403 responses.

**Fix:** header, TMA, and Bearer init-data paths now converge on the same signature verification and allowlist check.

### P1 — Provider timeout policy was silently changed

**Impact:** a configured timeout below 100ms was raised to 100ms, delaying fallback and making timeout tests/controls inaccurate.

**Fix:** the provider manager now uses a 10ms floor.

### P2 — Admin dashboard failure isolation

**Impact:** a single failed endpoint could make the whole dashboard appear broken.

**Fix:** refresh uses `Promise.allSettled`, keeps successful sections visible, and shows degraded/offline state.

### P2 — Admin dashboard security/UX

**Fixes:** safer DOM construction, API timeouts, visibility-aware refresh, bilingual labels, Telegram theme support, haptics, responsive metric grid, and removal of obsolete queue/Redis controls.

## Remaining technical debt

The application still imports `app.legacy` as a compatibility shell. That is now the main architecture debt. A safe next extraction order is:

1. Telegram message/command handlers into `app/services/telegram/handlers.py`.
2. User preference/history persistence into dedicated repositories.
3. In-process broadcast scheduler/dispatcher into a service with explicit lifecycle ownership.
4. Remove old advanced-admin HTML/Redis compatibility helpers.
5. Replace the legacy FastAPI shell with a native `FastAPI()` application after route parity tests exist.

This update deliberately does not combine all five steps with the Redis/worker cutover, because doing so would greatly increase regression risk.

## Validation performed

- Python compileall: pass
- Admin JavaScript syntax (`node --check`): pass
- Focused unittest suite: 27/27 pass
- Static architecture scan: no active job-worker imports outside legacy compatibility code; Redis package removed from requirements

The live GitHub branch could not be cloned in this execution environment, so these changes are packaged as an apply-at-repository-root update based on the latest v3.1 full update bundle available in the user's Library.
