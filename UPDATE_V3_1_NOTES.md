# V3.1 Artifact Source Defaults

Artifact storage no longer depends on `.env` for its normal defaults. The
following values are explicit named constants in
`app/services/artifacts/storage.py`:

```python
DEFAULT_BOT_ARTIFACT_STORAGE_MODE = "auto"
DEFAULT_BOT_ARTIFACT_STORAGE_BUCKET = "bot-job-artifacts"
DEFAULT_BOT_ARTIFACT_LOCAL_DIRECTORY = "data/job-artifacts"
DEFAULT_BOT_ARTIFACT_MAX_BYTES = 52_428_800
```

Environment variables remain optional overrides. Empty variables fall back to
the constants. An invalid `BOT_ARTIFACT_MAX_BYTES` value also falls back safely
and writes a warning instead of stopping startup.

No `.env.example` changes are required.
