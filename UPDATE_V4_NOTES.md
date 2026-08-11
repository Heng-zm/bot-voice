# Reliability and Observability Update V4

## Included

- Redis-backed idempotent delivery for durable TTS voice jobs.
- Automatic restart with bounded exponential backoff when a durable worker's
  outer run loop exits unexpectedly.
- Redis-indexed artifact expiration for both local and Supabase storage.
- Periodic artifact cleanup owned by worker and combined runtime roles.
- Queue age, hourly success/failure/cancellation counts, throughput, and final
  failure-rate metrics.
- Server-side job type filtering and bounded text search.
- Mini App queue sparklines, filters, and search controls.
- Focused TTS, OCR, broadcast, and runtime-settings validation modules with
  compatibility wrappers for staged legacy migration.

## Hardening follow-up

- Artifact cleanup now removes malformed registry records without stopping the
  batch, retains retryable records during storage outages, and distinguishes a
  missing Supabase object from a transient download failure.
- Queue statistics are collected in one Redis pipeline per dashboard refresh.
- Worker restart backoff resets after a stable run instead of retaining an old
  crash-loop penalty indefinitely.
- Job search and pagination surface request failures, prevent overlapping
  append requests, and reuse the worker snapshot already returned by the jobs
  endpoint.
- Runtime float settings reject non-finite values, and webhook base URLs now
  enforce the documented HTTPS-only format.

## Feature removal

- VoxCPM2 voice cloning was removed from Telegram commands, callbacks, TTS
  model choices, provider health, durable queues, workers, and the Admin Mini
  App. Stale model preferences and new submissions fall back to standard
  automatic TTS.

## Operations

The V4 queue/artifact changes require no database migration. A later lock-race
hardening update adds the recommended
`scripts/supabase_bot_locks.sql` migration for atomic scheduler/leader lease
acquisition. The runtime remains compatible during rollout through a
duplicate-safe fallback.

The cleanup loop defaults to 300 seconds. An optional
`BOT_ARTIFACT_CLEANUP_SECONDS` deployment variable can set a value from 30 to
86,400 seconds. Keep this override out of `.env.example`; that file intentionally
contains connection credentials only.

The existing Admin Runtime job endpoint accepts optional `job_type` and `query`
parameters. Search scans at most 1,000 indexed jobs per request.

## Rollback

The new Redis keys are additive. Rolling back code does not require deleting
them. Voice delivery and artifact expiration records expire or become inert
without V4 workers.
