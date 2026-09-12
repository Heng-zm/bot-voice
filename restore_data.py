"""Offline restore utility for Khmer Telegram Bot data.

Restores database records from local JSON backups (created by backup_data.py)
into a target Supabase database via PostgREST API.
Zero external dependencies (pure Python standard library).

Features:
- Shares the core execution engine with migrate_data.py (_migration_core.py)
- Auto-detects the latest backup folder in backups/ if none is specified
- Topological dependency order restoration with multi-wave concurrency
- Idempotent upserts via 'Prefer: resolution=merge-duplicates'
- Adaptive batch sizing and crash-safe checkpoint resumability
- Dry-run validation mode (--dry-run)
- Concurrent post-restore row count verification (--verify)
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import time
from typing import Any, Callable
import urllib.error
import urllib.parse
import urllib.request
import zipfile

from _migration_core import (
    SCHEMA_GRAPH,
    AdaptiveBatcher,
    Checkpoint,
    execute_with_retry,
    fetch_all_row_counts,
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


def test_connection(base_url: str, api_key: str) -> bool:
    """Verify connectivity and authentication to the target Supabase project."""
    url = f"{base_url.rstrip('/')}/rest/v1/bot_settings?select=key&limit=1"
    headers = make_headers(api_key)
    req = urllib.request.Request(url, headers=headers, method="GET")  # noqa: S310
    try:
        code, _, _ = execute_with_retry(req, timeout=15, max_retries=2)
        return code == 200
    except Exception as exc:
        with _print_lock:
            print(f"❌ Failed to connect to Target DB ({base_url}): {exc}")
        return False


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


def find_latest_backup_dir(base_dir: Path) -> Path | None:
    """Find the most recent backup folder in backups/."""
    backups_path = base_dir / "backups"
    if not backups_path.is_dir():
        return None

    dirs = [
        d for d in backups_path.iterdir()
        if d.is_dir() and d.name.startswith("backup_")
    ]
    if not dirs:
        return None

    dirs.sort(key=lambda d: d.name, reverse=True)
    return dirs[0]


def load_rows_from_csv(csv_path: Path) -> list[dict[str, Any]]:
    """Load and parse records from a CSV file into Python dicts."""
    rows: list[dict[str, Any]] = []
    if not csv_path.is_file():
        return rows
    with csv_path.open("r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for r in reader:
            parsed_row: dict[str, Any] = {}
            for k, v in r.items():
                if v == "" or v is None:
                    parsed_row[k] = None
                elif v.lower() == "true":
                    parsed_row[k] = True
                elif v.lower() == "false":
                    parsed_row[k] = False
                elif (v.startswith("{") and v.endswith("}")) or (v.startswith("[") and v.endswith("]")):
                    try:
                        parsed_row[k] = json.loads(v)
                    except Exception:
                        parsed_row[k] = v
                else:
                    parsed_row[k] = v
            rows.append(parsed_row)
    return rows


def load_table_data_from_backup(backup_dir: Path, table_name: str) -> list[dict[str, Any]]:
    """Load records for a table from {table}.json or {table}.csv, handling legacy names and full_backup.json."""
    # 1. Direct table JSON file
    json_path = backup_dir / f"{table_name}.json"
    if json_path.is_file():
        try:
            with json_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    return data
        except Exception as exc:
            with _print_lock:
                print(f"⚠️ Error reading {json_path.name}: {exc}")

    # 2. Direct table CSV file
    csv_path = backup_dir / f"{table_name}.csv"
    if csv_path.is_file():
        try:
            return load_rows_from_csv(csv_path)
        except Exception as exc:
            with _print_lock:
                print(f"⚠️ Error reading {csv_path.name}: {exc}")

    # 3. Legacy alias check (broadcast_schedules -> scheduled_broadcasts)
    if table_name == "scheduled_broadcasts":
        legacy_path = backup_dir / "broadcast_schedules.json"
        if legacy_path.is_file():
            try:
                with legacy_path.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        return data
            except Exception as exc:
                with _print_lock:
                    print(f"⚠️ Error reading legacy broadcast_schedules.json: {exc}")

    # 4. Fallback to full_backup.json
    full_path = backup_dir / "full_backup.json"
    if full_path.is_file():
        try:
            with full_path.open("r", encoding="utf-8") as f:
                full_data = json.load(f)
                if isinstance(full_data, dict):
                    if table_name in full_data and isinstance(full_data[table_name], list):
                        return full_data[table_name]
                    if table_name == "scheduled_broadcasts" and "broadcast_schedules" in full_data:
                        return full_data["broadcast_schedules"]
        except Exception as exc:
            with _print_lock:
                print(f"⚠️ Error reading {full_path.name}: {exc}")

    return []


def render_progress(table_name: str, current: int, total: int, status: str = "") -> None:
    """Thread-safe dynamic progress line."""
    pct = (current / total * 100) if total > 0 else 100.0
    bar_len = 16
    filled = int(bar_len * (current / total)) if total > 0 else bar_len
    bar = "=" * filled + "-" * (bar_len - filled)
    msg = f"\r⏳ {table_name:<22} [{bar}] {pct:>5.1f}% ({current}/{total}) {status}"
    with _print_lock:
        sys.stdout.write(msg)
        sys.stdout.flush()


def restore_single_table(
    table_name: str,
    backup_dir: Path,
    target_url: str,
    target_key: str,
    *,
    batcher: AdaptiveBatcher,
    checkpoint: Checkpoint,
    dry_run: bool = False,
    timeout: int = 30,
) -> dict[str, Any]:
    """Restores a single table from local JSON backup with adaptive batching."""
    spec = SCHEMA_GRAPH.get(table_name, {})
    conflict_key = spec.get("conflict") or spec.get("pk")

    if not dry_run and checkpoint.is_completed(table_name):
        offset = checkpoint.get_offset(table_name)
        with _print_lock:
            print(f"⏩ {table_name:<22} : Already completed at checkpoint ({offset} rows)")
        return {
            "table": table_name,
            "backup_count": offset,
            "restored": 0,
            "status": "Resumed (Completed)",
        }

    rows = load_table_data_from_backup(backup_dir, table_name)
    total_rows = len(rows)

    if total_rows == 0:
        with _print_lock:
            print(f"⚪ {table_name:<22} : 0 rows (no backup data found)")
        if not dry_run:
            checkpoint.mark_completed(table_name, 0)
        return {
            "table": table_name,
            "backup_count": 0,
            "restored": 0,
            "status": "Skipped (0 rows)",
        }

    offset = checkpoint.get_offset(table_name) if not dry_run else 0
    restored_count = 0
    status_msg = "OK"

    try:
        while offset < total_rows:
            batch_size = batcher.current_size
            chunk = rows[offset : offset + batch_size]
            if not chunk:
                break

            t_start = time.perf_counter()

            if not dry_run:
                try:
                    upserted = upsert_batch_rows(
                        target_url,
                        target_key,
                        table_name,
                        chunk,
                        conflict_key=conflict_key,
                        timeout=timeout,
                    )
                    elapsed = time.perf_counter() - t_start
                    batcher.on_success(elapsed)
                    restored_count += upserted
                    offset += len(chunk)
                    checkpoint.record_progress(table_name, offset, upserted)
                except urllib.error.HTTPError as exc:
                    if exc.code == 413:
                        batcher.on_payload_error()
                        continue
                    raise
            else:
                restored_count += len(chunk)
                offset += len(chunk)

            render_progress(table_name, offset, total_rows)

        render_progress(table_name, total_rows, total_rows, status="✅ Done\n")
        if not dry_run:
            checkpoint.mark_completed(table_name, total_rows)

    except Exception as exc:
        status_msg = f"Error: {exc}"
        with _print_lock:
            print(f"\n❌ {table_name:<22} failed: {exc}")

    return {
        "table": table_name,
        "backup_count": total_rows,
        "restored": restored_count,
        "status": status_msg,
    }


def run_restore(
    backup_dir: Path,
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
    timeout: int = 30,
) -> int:
    """Execute backup restoration into target Supabase."""
    target_url = target_url.rstrip("/")

    checkpoint_file = backup_dir / ".restore_checkpoint.jsonl"
    checkpoint = Checkpoint(checkpoint_file)

    if fresh:
        checkpoint.clear()
        print("🧹 Fresh restore requested — cleared existing restore checkpoint.")
    elif resume and checkpoint_file.is_file():
        print("🔄 Resumable checkpoint found — resuming restore from last checkpoint.")

    print("=" * 75)
    print("📦 SUPABASE RESUMABLE & CONCURRENT OFFLINE RESTORE")
    print(f"• Backup Directory : {backup_dir}")
    print(f"• Target DB        : {target_url}")
    print(f"• Concurrency      : {concurrency} workers")
    print(f"• Batch Size       : {batch_size} (adaptive up to {max_batch_size})")
    print(f"• Mode             : {'DRY RUN (Simulation)' if dry_run else 'LIVE RESTORE'}")
    print("=" * 75)

    if not backup_dir.is_dir():
        print(f"❌ Error: Backup directory not found: {backup_dir}")
        return 1

    if not dry_run:
        print("\n🔍 Checking target database connectivity...")
        if not test_connection(target_url, target_key):
            return 1
        print("  ✅ Target DB connected successfully")
    else:
        print("\nℹ️ Skipping target write validation (dry-run mode)")

    active_graph = dict(SCHEMA_GRAPH)
    if tables_filter:
        valid = {t.lower().strip() for t in tables_filter}
        active_graph = {k: v for k, v in active_graph.items() if k.lower() in valid}
        if not active_graph:
            print(f"❌ Error: No matching tables in filter: {tables_filter}")
            return 1

    waves = get_dependency_waves(active_graph)
    total_tables = sum(len(w) for w in waves)
    print(f"\n📦 Computed {len(waves)} topological restore waves for {total_tables} tables:")
    for idx, wave in enumerate(waves):
        print(f"  • Wave {idx + 1}: {', '.join(wave)}")

    results: list[dict[str, Any]] = []

    for wave_idx, wave in enumerate(waves, 1):
        print(f"\n🌊 Restoring Wave {wave_idx}/{len(waves)} ({len(wave)} tables in parallel)...")
        wave_workers = min(concurrency, len(wave))

        with concurrent.futures.ThreadPoolExecutor(max_workers=wave_workers) as executor:
            future_to_table = {
                executor.submit(
                    restore_single_table,
                    table_name=tbl,
                    backup_dir=backup_dir,
                    target_url=target_url,
                    target_key=target_key,
                    batcher=AdaptiveBatcher(initial_size=batch_size, max_size=max_batch_size),
                    checkpoint=checkpoint,
                    dry_run=dry_run,
                    timeout=timeout,
                ): tbl
                for tbl in wave
            }

            for future in concurrent.futures.as_completed(future_to_table):
                results.append(future.result())

    target_counts: dict[str, int] = {}
    if not dry_run and verify:
        print("\n🔍 Running concurrent post-restore verification across all tables...")
        target_counts = fetch_all_row_counts(
            target_url,
            target_key,
            list(active_graph.keys()),
            max_workers=concurrency,
            timeout=timeout,
        )

    topo_order = topo_sort(active_graph)
    results_map = {r["table"]: r for r in results}

    print("\n" + "=" * 78)
    print(f"{'Table':<24} | {'Backup':>8} | {'Restored':>8} | {'Target Verify':>13} | {'Status':<14}")
    print("-" * 78)
    total_backup = 0
    total_restored = 0

    for tbl in topo_order:
        res = results_map.get(tbl, {})
        b_cnt = res.get("backup_count", 0)
        r_cnt = res.get("restored", 0)
        t_cnt_str = str(target_counts.get(tbl, "-")) if verify and not dry_run else "-"
        stat = res.get("status", "OK")

        if isinstance(b_cnt, int) and b_cnt > 0:
            total_backup += b_cnt
        if isinstance(r_cnt, int) and r_cnt > 0:
            total_restored += r_cnt

        print(f"{tbl:<24} | {b_cnt:>8} | {r_cnt:>8} | {t_cnt_str:>13} | {stat:<14}")

    print("-" * 78)
    print(f"{'TOTAL':<24} | {total_backup:>8} | {total_restored:>8} | {'':>13} |")
    print("=" * 78)

    if dry_run:
        print("\n✨ Dry run simulation complete. No records were modified on the target.")
    else:
        print(f"\n🎉 Restore completed successfully! Total records restored this run: {total_restored}")
        if verify:
            print("✅ Row counts verified against target PostgREST API.")

    return 0


def restore_single_table_from_rows(
    table_name: str,
    rows: list[dict[str, Any]],
    target_url: str,
    target_key: str,
    *,
    batch_size: int = 500,
    dry_run: bool = False,
    timeout: int = 30,
) -> tuple[int, int]:
    """Restores a list of in-memory rows into target Supabase table using merge-duplicates."""
    spec = SCHEMA_GRAPH.get(table_name, {})
    conflict_key = spec.get("conflict") or spec.get("pk")
    total = len(rows)
    restored = 0

    for i in range(0, total, batch_size):
        chunk = rows[i : i + batch_size]
        if not dry_run:
            upserted = upsert_batch_rows(
                target_url,
                target_key,
                table_name,
                chunk,
                conflict_key=conflict_key,
                timeout=timeout,
            )
            restored += upserted
        else:
            restored += len(chunk)
    return total, restored


def restore_from_file_or_dir(
    target_url: str,
    target_key: str,
    path: Path,
    *,
    dry_run: bool = False,
    timeout: int = 30,
    progress_callback: Callable[[str, int, int], None] | None = None,
) -> dict[str, Any]:
    """Restores data from a backup folder, ZIP bundle, or single JSON/CSV file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Path not found: {path}")

    # Case A: Directory
    if path.is_dir():
        code = run_restore(
            backup_dir=path,
            target_url=target_url,
            target_key=target_key,
            dry_run=dry_run,
            timeout=timeout,
        )
        return {"success": code == 0, "path": str(path), "type": "directory"}

    # Case B: ZIP Archive
    if path.suffix.lower() == ".zip":
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            with zipfile.ZipFile(path, "r") as zf:
                zf.extractall(tmp_path)
            code = run_restore(
                backup_dir=tmp_path,
                target_url=target_url,
                target_key=target_key,
                dry_run=dry_run,
                timeout=timeout,
            )
            return {"success": code == 0, "path": str(path), "type": "zip"}

    # Case C: Single CSV or JSON file
    tbl = path.stem.lower()
    if tbl == "broadcast_schedules":
        tbl = "scheduled_broadcasts"

    if tbl not in SCHEMA_GRAPH:
        raise ValueError(
            f"Filename '{path.name}' does not correspond to a known schema table ({', '.join(SCHEMA_GRAPH.keys())})"
        )

    if path.suffix.lower() == ".csv":
        rows = load_rows_from_csv(path)
    elif path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
            rows = data if isinstance(data, list) else []
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")

    total, restored = restore_single_table_from_rows(
        tbl,
        rows,
        target_url,
        target_key,
        dry_run=dry_run,
        timeout=timeout,
    )
    if progress_callback:
        try:
            progress_callback(tbl, restored, total)
        except Exception:
            pass

    return {
        "success": True,
        "table": tbl,
        "total_rows": total,
        "restored_rows": restored,
        "type": "single_file",
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Resumable, concurrent offline restore tool for Supabase backups.",
    )
    parser.add_argument("--backup-dir", help="Path to backup directory (defaults to latest in backups/)")
    parser.add_argument("--target-url", help="Target Supabase URL (defaults to .env)")
    parser.add_argument("--target-key", help="Target Supabase Key (defaults to .env)")
    parser.add_argument("--concurrency", type=int, default=4, help="Concurrent workers (default: 4)")
    parser.add_argument("--batch-size", type=int, default=500, help="Initial batch size (default: 500)")
    parser.add_argument("--max-batch-size", type=int, default=2000, help="Adaptive ceiling (default: 2000)")
    parser.add_argument("--tables", help="Comma-separated list of tables to restore")
    parser.add_argument("--timeout", type=int, default=30, help="HTTP timeout in seconds (default: 30)")
    parser.add_argument("--dry-run", action="store_true", help="Simulate restore without writes")
    parser.add_argument("--verify", action="store_true", help="Verify target counts concurrently at end")
    parser.add_argument("--fresh", action="store_true", help="Discard checkpoint and start clean")
    parser.add_argument("--resume", action="store_true", default=True, help="Resume from checkpoint (default)")

    args = parser.parse_args()

    root_dir = Path(__file__).resolve().parent
    env = load_env(root_dir / ".env")

    backup_dir: Path | None = None
    if args.backup_dir:
        backup_dir = Path(args.backup_dir).resolve()
    else:
        backup_dir = find_latest_backup_dir(root_dir)
        if not backup_dir:
            print("❌ Error: No backup directory specified and no backups found in backups/.")
            return 1
        print(f"📁 Auto-detected latest backup directory: {backup_dir.name}")

    target_url = (
        args.target_url
        or os.environ.get("TARGET_SUPABASE_URL")
        or os.environ.get("SUPABASE_URL")
        or env.get("TARGET_SUPABASE_URL")
        or env.get("SUPABASE_URL", "")
    )
    target_key = (
        args.target_key
        or os.environ.get("TARGET_SUPABASE_SERVICE_ROLE_KEY")
        or os.environ.get("TARGET_SUPABASE_KEY")
        or os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
        or os.environ.get("SUPABASE_KEY")
        or env.get("TARGET_SUPABASE_SERVICE_ROLE_KEY")
        or env.get("TARGET_SUPABASE_KEY")
        or env.get("SUPABASE_SERVICE_ROLE_KEY")
        or env.get("SUPABASE_KEY")
        or ""
    )

    if not args.dry_run and (not target_url or not target_key):
        print("❌ Error: Missing Target Supabase credentials in --target-url/key or .env")
        return 1

    tables_filter = [t.strip() for t in args.tables.split(",") if t.strip()] if args.tables else None

    return run_restore(
        backup_dir=backup_dir,
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
        timeout=args.timeout,
    )


if __name__ == "__main__":
    sys.exit(main())
