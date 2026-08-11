# V4 Validation

The codebase targets Python 3.11 and newer. Local validation was run on Python
3.12, while CI compiles and tests the deployment sources on both Python 3.11
and 3.12.

```text
python -m pytest -q -p no:cacheprovider
140 passed

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

Deployment compatibility additionally checks the complete source tree with the
Python 3.11 compiler in CI, preventing Python 3.12-only f-string grammar from
reaching a Python 3.11 host.

Supabase lock coverage verifies atomic RPC acquisition, duplicate-safe
pre-migration fallback, same-owner renewal, bounded inputs, and the checked-in
service-role-only SQL migration.

Admin Mini App UI coverage verifies the Google Sans Flex/Noto Sans Khmer font
pair, narrowly scoped font CSP hosts, cache-busted styles, safe-area support,
touch behavior, and responsive 700/420/350-pixel layouts.

The upgraded Bot Monitor validation covers process CPU sampling, server-side
queue-pressure health, log-level summaries, cache-busted command-center assets,
live trend/interval/filter/log controls, and scroll-aware mobile navigation.

The host's shared Python installation emits a Requests dependency warning that
is outside the project lock file. It does not fail this repository's checks.
