# V4 Validation

Validated on Python 3.12 with the repository test and lint configuration.

```text
python -m pytest -q -p no:cacheprovider
139 passed

python -m ruff check .
All checks passed!

python -m compileall -q app main.py tests
passed

node --check static/admin/app.js
passed
```

Regression coverage includes duplicate voice delivery, worker-loop restart and
stable-run backoff reset, malformed and transient-failure artifact cleanup,
retry-safe artifact replacement, pipelined queue metrics, queue filtering,
runtime ownership, URL/finite-number validation, removed-model fallback, and
extracted services. Bot Monitor coverage verifies process snapshots, bounded
secret-redacted runtime logs, and safe TTS progress responses without job
payloads or results.

Supabase lock coverage verifies atomic RPC acquisition, duplicate-safe
pre-migration fallback, same-owner renewal, bounded inputs, and the checked-in
service-role-only SQL migration.

Admin Mini App UI coverage verifies the Google Sans Flex/Noto Sans Khmer font
pair, narrowly scoped font CSP hosts, cache-busted styles, safe-area support,
touch behavior, and responsive 700/420/350-pixel layouts.

The host's shared Python installation emits a Requests dependency warning that
is outside the project lock file. It does not fail this repository's checks.
