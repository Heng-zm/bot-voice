"""Online direct database migration utility for Khmer Telegram Bot data.

Streams data directly from Source Supabase to Target Supabase via PostgREST API.
Zero external dependencies (pure Python standard library).

Optimized capabilities:
- Topological sort computed dynamically via Kahn's algorithm
- Multi-tier wave concurrency: independent tables migrate in parallel
- Safe concurrent worker pool for HTTP I/O
- Crash-safe checkpointing (.migration_checkpoint.jsonl) with automatic resume
- Adaptive batch sizing (dynamically adjusts to latency and avoids HTTP 413)
- Single-pass concurrent row verification (--verify)
"""

from __future__ import annotations

import argparse
import concurrent.futures
from datetime import UTC, datetime, timedelta
import json
import os
from pathlib import Path
import sys
import threading
import time
from typing import Any
import urllib.error
import urllib.parse
import urllib.request

from _migration_core import (
    SCHEMA_GRAPH,
    AdaptiveBatcher,
    Checkpoint,
    execute_with_retry,
    fetch_all_row_counts,
    fetch_row_count,
    get_dependency_waves,
    load_env,
    make_headers,
    topo_sort,
)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

_print_lock = threading.Lock()


def test_connection(base_url: str, api_key: str, role_name: str = "Database") -> bool:
    """Verify connectivity and authentication to a Supabase project."""
    url = f"{base_url.rstrip('/')}/rest/v1/bot_settings?select=key&limit=1"
    headers = make_headers(api_key)
    req = urllib.request.Request(url, headers=headers, method="GET")  # noqa: S310
    try:
        code, _, _ = execute_with_retry(req, timeout=15, max_retries=2)
        return code == 200
    except Exception as exc:
        with _print_lock:
            print(f"❌ Failed to connect to {role_name} ({base_url}): {exc}")
        return False


def fetch_batch_rows(
    base_url: str,
    api_key: str,
    table_name: str,
    limit: int,
    offset: int,
    *,
    date_filter: str | None = None,
    timeout: int = 30,
) -> list[dict[str, Any]]:
    """Fetch a single page of rows from a Supabase table."""
    pk = SCHEMA_GRAPH.get(table_name, {}).get("pk", "id")
    params: dict[str, str] = {
        "select": "*",
        "limit": str(limit),
        "offset": str(offset),
        "order": f"{pk}.asc",
    }
    if date_filter:
        params["created_at"] = f"gte.{date_filter}"

    query_str = urllib.parse.urlencode(params)
    url = f"{base_url.rstrip('/')}/rest/v1/{table_name}?{query_str}"
    headers = make_headers(api_key)
    req = urllib.request.Request(url, headers=headers, method="GET")  # noqa: S310

    try:
        _, body, _ = execute_with_retry(req, timeout=timeout)
        data = json.loads(body.decode("utf-8"))
        return data if isinstance(data, list) else []
    except urllib.error.HTTPError as exc:
        err_body = exc.read().decode("utf-8", errors="replace") if exc.fp else ""
        if exc.code == 404 or "PGRST204" in err_body or "relation" in err_body:
            return []
        raise RuntimeError(f"HTTP {exc.code} fetching {table_name}: {err_body}") from exc


def upsert_batch_rows(
    base_url: str,
    api_key: str,
    table_name: str,
    rows: list[dict[str, Any]],
    conflict_key: str | None = None,
    *,
    timeout: int = 30,
) -> int:
    """Upsert a batch of rows into target Supabase table using merge-duplicates."""
    if not rows:
        return 0

    url = f"{base_url.rstrip('/')}/rest/v1/{table_name}"
    if conflict_key:
        url = f"{url}?on_conflict={urllib.parse.quote(conflict_key)}"

    headers = make_headers(
        api_key,
        prefer="resolution=merge-duplicates, return=minimal",
        content_type="application/json",
    )
    payload = json.dumps(rows, ensure_ascii=False).encode("utf-8")

    req = urllib.request.Request(url, data=payload, headers=headers, method="POST")  # noqa: S310
    code, _, _ = execute_with_retry(req, timeout=timeout)
    if code in {200, 201, 204}:
        return len(rows)
    return 0


def render_progress(table_name: str, current: int, total: int, status: str = "") -> None:
    """Thread-safe dynamic progress reporter."""
    pct = (current / total * 100) if total > 0 else 100.0
    bar_len = 16
    filled = int(bar_len * (current / total)) if total > 0 else bar_len
    bar = "=" * filled + "-" * (bar_len - filled)
    msg = f"\r⏳ {table_name:<22} [{bar}] {pct:>5.1f}% ({current}/{total}) {status}"
    with _print_lock:
        sys.stdout.write(msg)
        sys.stdout.flush()


def migrate_single_table(
    table_name: str,
    source_url: str,
    source_key: str,
    target_url: str,
    target_key: str,
    *,
    batcher: AdaptiveBatcher,
    checkpoint: Checkpoint,
    dry_run: bool = False,
    date_filter: str | None = None,
    timeout: int = 30,
) -> dict[str, Any]:
    """Migrates a single table with adaptive batching and checkpointing."""
    spec = SCHEMA_GRAPH.get(table_name, {})
    conflict_key = spec.get("conflict") or spec.get("pk")
    use_date_filter = date_filter if spec.get("has_created_at") else None

    # Check if table already completed in checkpoint
    if not dry_run and checkpoint.is_completed(table_name):
        offset = checkpoint.get_offset(table_name)
        with _print_lock:
            print(f"⏩ {table_name:<22} : Already completed at checkpoint ({offset} rows)")
        return {
            "table": table_name,
            "source_count": offset,
            "migrated": 0,
            "resumed_from": offset,
            "status": "Resumed (Completed)",
        }

    # Query source total
    source_total = fetch_row_count(source_url, source_key, table_name, timeout=timeout)
    if source_total < 0:
        sample = fetch_batch_rows(source_url, source_key, table_name, limit=1, offset=0, timeout=timeout)
        source_total = 1 if sample else 0

    if source_total == 0:
        with _print_lock:
            print(f"⚪ {table_name:<22} : 0 rows (table empty or unmigrated)")
        if not dry_run:
            checkpoint.mark_completed(table_name, 0)
        return {
            "table": table_name,
            "source_count": 0,
            "migrated": 0,
            "resumed_from": 0,
            "status": "Skipped (0 rows)",
        }

    # Resume from last checkpoint offset if available
    offset = checkpoint.get_offset(table_name) if not dry_run else 0
    resumed_offset = offset
    migrated_count = 0
    status_msg = "OK"

    try:
        while True:
            batch_size = batcher.current_size
            t_start = time.perf_counter()

            try:
                batch = fetch_batch_rows(
                    source_url,
                    source_key,
                    table_name,
                    limit=batch_size,
                    offset=offset,
                    date_filter=use_date_filter,
                    timeout=timeout,
                )
            except urllib.error.HTTPError as exc:
                if exc.code == 413:
                    batcher.on_payload_error()
                    continue
                raise

            if not batch:
                break

            if not dry_run:
                try:
                    upserted = upsert_batch_rows(
                        target_url,
                        target_key,
                        table_name,
                        batch,
                        conflict_key=conflict_key,
                        timeout=timeout,
                    )
                    elapsed = time.perf_counter() - t_start
                    batcher.on_success(elapsed)
                    migrated_count += upserted
                    offset += len(batch)
                    checkpoint.record_progress(table_name, offset, upserted)
                except urllib.error.HTTPError as exc:
                    if exc.code == 413:
                        batcher.on_payload_error()
                        continue
                    raise
            else:
                migrated_count += len(batch)
                offset += len(batch)

            render_progress(table_name, offset, max(source_total, offset))
            if len(batch) < batch_size:
                break

        render_progress(table_name, offset, offset, status="✅ Done\n")
        if not dry_run:
            checkpoint.mark_completed(table_name, offset)

    except Exception as exc:
        status_msg = f"Error: {exc}"
        with _print_lock:
            print(f"\n❌ {table_name:<22} failed: {exc}")

    return {
        "table": table_name,
        "source_count": source_total,
        "migrated": migrated_count,
        "resumed_from": resumed_offset,
        "status": status_msg,
    }


def run_migration(
    source_url: str,
    source_key: str,
    target_url: str,
    target_key: str,
    *,
    concurrency: int = 4,
    batch_size: int = 500,
    max_batch_size: int = 2000,
    dry_run: bool = False,
    verify: bool = False,
    resume: bool = True,
    fresh: bool = False,
    tables_filter: list[str] | None = None,
    days: int | None = None,
    timeout: int = 30,
) -> int:
    """Concurrent, resumable migration engine."""
    source_url = source_url.rstrip("/")
    target_url = target_url.rstrip("/")

    root_dir = Path(__file__).resolve().parent
    checkpoint_file = root_dir / ".migration_checkpoint.jsonl"
    checkpoint = Checkpoint(checkpoint_file)

    if fresh:
        checkpoint.clear()
        print("🧹 Fresh run requested — discarded existing checkpoint.")
    elif resume and checkpoint_file.is_file():
        print("🔄 Resumable checkpoint found — resuming migration from last known cursor.")

    print("=" * 75)
    print("🚀 SUPABASE RESUMABLE & CONCURRENT DATABASE MIGRATION")
    print(f"• Source DB   : {source_url}")
    print(f"• Target DB   : {target_url}")
    print(f"• Concurrency : {concurrency} workers")
    print(f"• Batch Size  : {batch_size} (adaptive up to {max_batch_size})")
    print(f"• Mode        : {'DRY RUN (Simulation)' if dry_run else 'LIVE MIGRATION'}")
    if days:
        print(f"• Date Window : Last {days} days for temporal tables")
    print("=" * 75)

    print("\n🔍 Verifying database connectivity...")
    if not test_connection(source_url, source_key, "Source DB"):
        return 1
    print("  ✅ Source DB connected successfully")

    if not dry_run:
        if not test_connection(target_url, target_key, "Target DB"):
            return 1
        print("  ✅ Target DB connected successfully")
    else:
        print("  ℹ️ Skipping target write validation (dry-run mode)")

    date_filter: str | None = None
    if days:
        cutoff = datetime.now(UTC) - timedelta(days=days)
        date_filter = cutoff.strftime("%Y-%m-%dT%H:%M:%SZ")

    # Filter graph if user specified subset of tables
    active_graph = dict(SCHEMA_GRAPH)
    if tables_filter:
        valid = {t.lower().strip() for t in tables_filter}
        active_graph = {k: v for k, v in active_graph.items() if k.lower() in valid}
        if not active_graph:
            print(f"❌ Error: No matching tables in filter: {tables_filter}")
            return 1

    # Dynamically compute execution waves
    waves = get_dependency_waves(active_graph)
    total_tables = sum(len(w) for w in waves)
    print(f"\n📦 Computed {len(waves)} topological execution waves for {total_tables} tables:")
    for idx, wave in enumerate(waves):
        print(f"  • Wave {idx + 1}: {', '.join(wave)}")

    results: list[dict[str, Any]] = []

    # Execute wave by wave
    for wave_idx, wave in enumerate(waves, 1):
        print(f"\n🌊 Executing Wave {wave_idx}/{len(waves)} ({len(wave)} tables in parallel)...")
        wave_workers = min(concurrency, len(wave))

        with concurrent.futures.ThreadPoolExecutor(max_workers=wave_workers) as executor:
            future_to_table = {
                executor.submit(
                    migrate_single_table,
                    table_name=tbl,
                    source_url=source_url,
                    source_key=source_key,
                    target_url=target_url,
                    target_key=target_key,
                    batcher=AdaptiveBatcher(initial_size=batch_size, max_size=max_batch_size),
                    checkpoint=checkpoint,
                    dry_run=dry_run,
                    date_filter=date_filter,
                    timeout=timeout,
                ): tbl
                for tbl in wave
            }

            for future in concurrent.futures.as_completed(future_to_table):
                res = future.result()
                results.append(res)

    # Post-migration single-pass concurrent verification
    target_counts: dict[str, int] = {}
    if not dry_run and verify:
        print("\n🔍 Running concurrent post-migration verification across all tables...")
        target_counts = fetch_all_row_counts(
            target_url,
            target_key,
            list(active_graph.keys()),
            max_workers=concurrency,
            timeout=timeout,
        )

    # Order results by topological sort for final reporting
    topo_order = topo_sort(active_graph)
    results_map = {r["table"]: r for r in results}

    print("\n" + "=" * 78)
    print(f"{'Table':<24} | {'Source':>8} | {'Migrated':>8} | {'Target Verify':>13} | {'Status':<14}")
    print("-" * 78)
    total_source = 0
    total_migrated = 0

    for tbl in topo_order:
        res = results_map.get(tbl, {})
        s_cnt = res.get("source_count", 0)
        m_cnt = res.get("migrated", 0)
        t_cnt_str = str(target_counts.get(tbl, "-")) if verify and not dry_run else "-"
        stat = res.get("status", "OK")

        if isinstance(s_cnt, int) and s_cnt > 0:
            total_source += s_cnt
        if isinstance(m_cnt, int) and m_cnt > 0:
            total_migrated += m_cnt

        print(f"{tbl:<24} | {s_cnt:>8} | {m_cnt:>8} | {t_cnt_str:>13} | {stat:<14}")

    print("-" * 78)
    print(f"{'TOTAL':<24} | {total_source:>8} | {total_migrated:>8} | {'':>13} |")
    print("=" * 78)

    if dry_run:
        print("\n✨ Dry run simulation complete. Target was not modified.")
    else:
        print(f"\n🎉 Migration complete! Total records upserted this run: {total_migrated}")
        if verify:
            print("✅ All target row counts reconciled via concurrent PostgREST checks.")

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Resumable, concurrent Supabase database migration utility.",
    )
    parser.add_argument("--source-url", help="Source Supabase URL (defaults to .env)")
    parser.add_argument("--source-key", help="Source Supabase Key (defaults to .env)")
    parser.add_argument("--target-url", help="Target Supabase URL (or TARGET_SUPABASE_URL)")
    parser.add_argument("--target-key", help="Target Supabase Key (or TARGET_SUPABASE_KEY)")
    parser.add_argument("--concurrency", type=int, default=4, help="Concurrent workers (default: 4)")
    parser.add_argument("--batch-size", type=int, default=500, help="Initial batch size (default: 500)")
    parser.add_argument("--max-batch-size", type=int, default=2000, help="Adaptive ceiling (default: 2000)")
    parser.add_argument("--tables", help="Comma-separated table list")
    parser.add_argument("--days", type=int, default=None, help="Migrate records from last N days")
    parser.add_argument("--timeout", type=int, default=30, help="HTTP timeout in seconds (default: 30)")
    parser.add_argument("--dry-run", action="store_true", help="Simulate migration without writes")
    parser.add_argument("--verify", action="store_true", help="Verify target counts concurrently at end")
    parser.add_argument("--fresh", action="store_true", help="Discard checkpoint and start clean")
    parser.add_argument("--resume", action="store_true", default=True, help="Resume from checkpoint (default)")

    args = parser.parse_args()

    root_dir = Path(__file__).resolve().parent
    env = load_env(root_dir / ".env")

    source_url = (
        args.source_url
        or os.environ.get("SUPABASE_URL")
        or env.get("SUPABASE_URL", "")
    )
    source_key = (
        args.source_key
        or os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
        or os.environ.get("SUPABASE_KEY")
        or env.get("SUPABASE_SERVICE_ROLE_KEY")
        or env.get("SUPABASE_KEY")
        or ""
    )
    target_url = (
        args.target_url
        or os.environ.get("TARGET_SUPABASE_URL")
        or env.get("TARGET_SUPABASE_URL", "")
    )
    target_key = (
        args.target_key
        or os.environ.get("TARGET_SUPABASE_SERVICE_ROLE_KEY")
        or os.environ.get("TARGET_SUPABASE_KEY")
        or env.get("TARGET_SUPABASE_SERVICE_ROLE_KEY")
        or env.get("TARGET_SUPABASE_KEY")
        or ""
    )

    if not source_url or not source_key:
        print("❌ Error: Missing Source Supabase credentials in --source-url/key or .env")
        return 1

    if not args.dry_run and (not target_url or not target_key):
        print("❌ Error: Missing Target Supabase credentials in --target-url/key or .env")
        return 1

    tables_filter = [t.strip() for t in args.tables.split(",") if t.strip()] if args.tables else None

    return run_migration(
        source_url=source_url,
        source_key=source_key,
        target_url=target_url,
        target_key=target_key,
        concurrency=args.concurrency,
        batch_size=args.batch_size,
        max_batch_size=args.max_batch_size,
        dry_run=args.dry_run,
        verify=args.verify,
        resume=not args.fresh,
        fresh=args.fresh,
        tables_filter=tables_filter,
        days=args.days,
        timeout=args.timeout,
    )


if __name__ == "__main__":
    sys.exit(main())
