# V4 Validation

Validated on Python 3.12 with the repository test and lint configuration.

```text
python -m pytest -q -p no:cacheprovider
128 passed

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
extracted services.

The host's shared Python installation emits a Requests dependency warning that
is outside the project lock file. It does not fail this repository's checks.
