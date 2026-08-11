# Runtime Reliability Update v2

This update builds on the first durable-runtime patch and adds operational
visibility, deploy controls, and three low-risk legacy extractions.

## Added

- Per-job progress fields: percentage, stage, detail, and update timestamp.
- Progress reporting from TTS, OCR, transcription, and broadcast workers.
- Redis indexes for succeeded and cancelled jobs in addition to queued, running,
  and dead jobs.
- Retention pruning for terminal job indexes.
- Worker drain/resume controls. Draining stops new claims while in-flight jobs
  finish normally. New process starts accept jobs unless
  `BOT_JOB_START_DRAINED=true` is explicitly set.
- Admin endpoints and Mini App controls for drain/resume and terminal job views.
- Real implementations for `app/utils/file_io.py`, `app/utils/time.py`, and
  `app/services/ai/language.py`; these modules no longer proxy through
  `app._legacy_bridge`.
- GitHub Actions quality gate, Docker readiness health check, `pip check`, and a
  `.dockerignore` suitable for the current repository.
- Regression tests for extracted utilities, job progress, terminal indexes,
  and worker drain/resume.

## New admin operations

```text
POST /api/admin/runtime/workers/drain
POST /api/admin/runtime/workers/resume
GET  /api/admin/runtime/jobs/list?state=succeeded
GET  /api/admin/runtime/jobs/list?state=cancelled
```

## Deployment sequence

1. Apply the original runtime update first if it is not already merged.
2. Apply this v2 patch.
3. Run `python -m ruff check .`.
4. Run `python -m unittest discover -s tests -v`.
5. Deploy one instance and confirm `/readyz` is healthy.
6. Test drain mode: queued counts should remain stable while running jobs finish.
7. Resume workers and verify queued jobs begin processing.

## Remaining work

The existing Telegram handlers still perform their heavy work inline until each
call site is explicitly switched to the typed submission helpers. Provider
circuit-breaker state also remains process-local; Redis-backed cluster state is
recommended only after single-instance durable queue behavior is stable.
