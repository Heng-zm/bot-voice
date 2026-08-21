# bot-voice V4.2.2 Final Full Source

This archive consolidates the V4 single-process migration, V4.1 Telegram handler
extraction, V4.2 workload/replay/runtime optimizations, and the V4.2.1/V4.2.2
startup compatibility fixes.

## Supported startup commands

Preferred:

```bash
python -m app.main
```

Compatibility modes:

```bash
python main.py
python app/main.py
```

The direct-file mode bootstraps the repository root before importing the `app`
package, fixing `ModuleNotFoundError: No module named 'app'` on deployment panels
that execute `app/main.py` directly.

## Install

```bash
python -m pip install -r requirements.txt
python -m compileall -q app tests
python -m pytest -q
python -m app.main
```

Configure environment variables from `.env.example`. Run `supabase_bot_setup.sql`
against the target Supabase project when the required tables have not yet been
created.

## Architecture

- One Python process owns FastAPI, Telegram, schedulers, and bounded heavy-media work.
- Redis and the dedicated worker service are removed from the supported runtime.
- Persistent runtime/admin settings use Supabase `bot_settings`.
- Webhook replay protection is process-local; deploy one active application instance.
- `legacy.py` remains only as the staged compatibility monolith for helpers not yet
  extracted. New Telegram handlers and runtime ownership live in native modules.
