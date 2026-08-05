# Validation v3

Performed against the complete v2 update bundle plus v3 changes.

## Passed

- Python bytecode compilation for all `app/` and `tests/` files.
- 15 targeted unit tests:
  - queue backpressure and dead-job listing
  - queue progress and terminal transitions
  - synchronous provider timeout
  - atomic local artifact storage, size limits, and integrity verification
  - delivery lease contention and retry-safe single Telegram edit
  - OCR/transcription submission payloads
  - static verification of real legacy call-site migration
  - dedicated worker entry point verification
- Archive integrity check for generated ZIP bundles.
- Incremental patch dry-run against the v2 full bundle.

## Not completed in this environment

- Ruff was not installed in the execution environment, so `ruff check` could
  not be executed here.
- Full repository tests require source modules and third-party dependencies not
  included in the uploaded subset.
- Live Supabase Storage, Redis, Telegram, and provider integration tests require
  deployment credentials and were not attempted.

## Required production checks

- Create and permission the private Supabase artifact bucket.
- Start at least one `app.worker` instance before enabling durable call sites.
- Confirm `/readyz` is healthy on the web service.
- Submit one photo and one voice message, then replay the same Telegram update
  and confirm only one result message is edited.
