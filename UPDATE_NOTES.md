# Runtime Reliability Update

This bundle implements the P0 runtime foundation without adding new feature
logic to `app/legacy.py`.

## Included

- A single idempotent `RuntimeContext` for ASGI-only and combined-process boot.
- Durable Redis workers started and stopped with the application lifecycle.
- Worker heartbeat data included in `/readyz` and the admin API.
- Production job handlers for TTS, OCR, transcription, and broadcast.
- Typed submission helpers using durable Telegram file IDs and idempotency keys.
- Queue backpressure with `BOT_JOB_QUEUE_MAX`.
- Cursor listing for queued, running, and dead jobs.
- Bulk retry for selected dead jobs.
- Bounded synchronous provider execution with policy timeouts.
- Provider health explicitly labelled `process` scope with an instance ID.
- Admin Mini App sections for jobs, providers, administrators, and audit logs.
- Khmer/English dashboard switching.
- Additional queue and provider timeout regression tests.

## Merge paths

Copy the files in this bundle over the same paths in the complete repository.
New files are:

```text
app/runtime.py
app/services/jobs/handlers.py
app/services/jobs/submission.py
tests/test_job_queue_runtime.py
tests/test_provider_timeout.py
```

## Switching legacy inline work to the durable queue

The worker lifecycle is active after this update. Existing legacy Telegram
handlers still need to replace each inline heavy operation with one typed
submission call during extraction. Do not place raw files or temporary paths in
a job payload.

Example for text-to-speech:

```python
from app.services.jobs.submission import submit_tts_job

job, created = await submit_tts_job(
    chat_id=update.effective_chat.id,
    user_id=update.effective_user.id,
    text=resolved_text,
    gender=gender,
    speed=speed,
    tts_model=tts_model,
    reply_to_message_id=update.effective_message.message_id,
    idempotency_key=f"telegram:{update.update_id}:tts",
)
```

Use equivalent helpers for Telegram photo/audio file IDs and broadcasts.
Idempotency keys should derive from a stable source such as Telegram
`update_id`, an admin broadcast record ID, or an API request ID.

This bundle deliberately does not rewrite the 1.4 MB `legacy.py` monolith.
Changing all five large Telegram handlers in one patch would increase rollout
risk. Migrate their call sites one workflow at a time and keep the existing
unit tests green after each extraction.

## Optional environment controls

These are optional code overrides and do not need to be added to the minimal
`.env.example`:

```dotenv
BOT_JOB_WORKERS=2
BOT_JOB_QUEUE_MAX=1000
BOT_JOB_POLL_SECONDS=0.5
PROVIDER_SYNC_MAX_WORKERS=4
PROVIDER_SYNC_MAX_INFLIGHT=8
INSTANCE_ID=
```

## Deployment behavior

- `python -m app.main` acquires the runtime as the `combined` owner.
- Uvicorn's ASGI lifespan acquires it as the `asgi` owner.
- Services are released only after all owners stop, preventing double startup
  and premature shutdown in the combined process.
- `/readyz` returns HTTP 503 when Redis, security bootstrap, the queue, or
  durable workers are not ready.
- Provider circuit-breaker metrics are process-local. Multi-instance dashboards
  show the instance identifier so this state is not mistaken for cluster-wide
  health.

## Recommended rollout

1. Deploy one instance and verify `/readyz` reports all workers alive.
2. Enqueue a test TTS job with a unique idempotency key.
3. Verify Admin Mini App Jobs shows queued → running → succeeded behavior.
4. Test cancellation and one dead-job retry.
5. Extract one Telegram workflow at a time to call the submission helpers.
6. After stable single-instance operation, decide whether provider health must
   be moved into Redis for cluster-wide circuit breakers.
