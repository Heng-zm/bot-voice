# Runtime Reliability Update v3

V3 intentionally contains four production deliverables only.

## 1. Actual OCR and transcription call-site migration

The existing Telegram handlers now enqueue durable jobs by default:

- Photo messages use `submit_ocr_job()`.
- Telegram voice messages use `submit_transcription_job()`.
- Audio documents and Telegram audio messages use the transcription queue.
- When audio-to-voice conversion is also enabled, conversion finishes first and
  the same progress message is handed to the transcription worker.

Each request uses a deterministic idempotency key containing the Telegram
`update_id` and `file_unique_id`. A repeated webhook/update therefore resolves
to the existing job instead of creating another workload.

Emergency rollback flags:

```text
DURABLE_OCR_ENABLED=false
DURABLE_TRANSCRIPTION_ENABLED=false
```

Setting either flag to false restores the preserved inline legacy path for that
workload.

## 2. Idempotent Telegram result delivery

`app/services/telegram/delivery.py` stores delivery state in Redis. Result
workers claim a short lease and retain the successful Telegram message ID.
Retries return the stored result without calling Telegram again.

Migrated handlers pass the processing-message ID. Delivery edits that message
in place, which closes the most important crash window: a retry converges on
one known Telegram message instead of sending a new result message.

Delivery keys contain only a SHA-256 digest of the logical idempotency key and
expire after seven days by default. Raw user text and Telegram credentials are
not written to delivery keys.

## 3. Durable artifact storage

Full OCR/transcription text is stored outside Redis job hashes. Redis receives
only JSON-safe metadata:

- backend and private bucket/path
- content type and size
- SHA-256 digest
- creation and expiry timestamps

Supabase Storage is the production backend. Local atomic storage is retained
for tests and one-process development. Reads verify both byte length and
SHA-256 before returning content.

Create a private Supabase bucket named `bot-job-artifacts` before production
rollout. Do not make this bucket public. Configure a storage lifecycle policy
or scheduled cleanup for objects after the application retention window.

## 4. Separate `app.worker` process

Run the HTTP/Telegram-ingestion process and queue consumer independently:

```bash
PROCESS_ROLE=web uvicorn app.main:app --host 0.0.0.0 --port 8080
PROCESS_ROLE=worker python -m app.worker
```

The worker process initializes Redis, Supabase, runtime security, artifact
storage, delivery leases, and the registered job handlers. It does not bind an
HTTP port and does not start Telegram polling or webhook ingestion.

The Docker health check is role-aware. Web containers call `/readyz`; worker
containers ping Redis. The combined `python -m app.main` command remains
available for local development.

## Rollout order

1. Create the private Supabase artifact bucket.
2. Deploy the worker service with durable flags still disabled.
3. Verify worker logs show healthy workers and shared Supabase artifacts.
4. Deploy the web service from the same commit.
5. Enable `DURABLE_OCR_ENABLED=true`.
6. Test repeated delivery of the same Telegram update.
7. Enable `DURABLE_TRANSCRIPTION_ENABLED=true`.
8. Monitor queue depth, dead jobs, provider failures, and storage usage.

Do not enable durable call sites until at least one worker service is running.
