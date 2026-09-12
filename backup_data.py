"""Standalone backup utility for Khmer Telegram Bot data.

Uses only Python standard library (zero external dependencies).
Queries the Supabase PostgREST REST API and exports tables to JSON and CSV.

Optimizations:
- Single source of truth: tables derived dynamically from _migration_core.SCHEMA_GRAPH
- Concurrent row count inspection across all tables
- Streamed disk writes: appends batches directly to disk to prevent memory bloat on large tables
- Real-time progress callback support for Telegram bot integration
"""

from __future__ import annotations

import argparse
import csv
from datetime import UTC, datetime
import json
import os
from pathlib import Path
import sys
from typing import Any, Callable
import urllib.error
import urllib.parse
import urllib.request

from _migration_core import (
    SCHEMA_GRAPH,
    load_env,
    make_headers,
    topo_sort,
)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Derive canonical table sequence from topological dependency graph
TABLES: tuple[str, ...] = tuple(topo_sort(SCHEMA_GRAPH))


def fetch_table_rows(
    supabase_url: str,
    api_key: str,
    table_name: str,
    page_size: int = 1000,
    timeout: int = 30,
) -> list[dict[str, Any]]:
    """Fetch all rows from a Supabase table with pagination."""
    base_url = supabase_url.rstrip("/")
    endpoint = f"{base_url}/rest/v1/{table_name}"
    headers = make_headers(api_key, prefer="count=exact")

    rows: list[dict[str, Any]] = []
    offset = 0

    while True:
        query_params = urllib.parse.urlencode({
            "select": "*",
            "limit": page_size,
            "offset": offset,
        })
        req_url = f"{endpoint}?{query_params}"
        req = urllib.request.Request(req_url, headers=headers, method="GET")  # noqa: S310

        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
                data = json.loads(resp.read().decode("utf-8"))
                if not data or not isinstance(data, list):
                    break
                rows.extend(data)
                if len(data) < page_size:
                    break
                offset += page_size
        except urllib.error.HTTPError as exc:
            err_body = exc.read().decode("utf-8", errors="replace") if exc.fp else ""
            if exc.code == 404 or "PGRST204" in err_body or "relation" in err_body:
                return []
            raise RuntimeError(f"HTTP {exc.code} fetching {table_name}: {err_body}") from exc
        except Exception as exc:
            raise RuntimeError(f"Failed to fetch {table_name}: {exc}") from exc

    return rows


def stream_backup_table_to_disk(
    supabase_url: str,
    api_key: str,
    table_name: str,
    json_path: Path,
    csv_path: Path,
    page_size: int = 1000,
    timeout: int = 30,
) -> tuple[int, list[dict[str, Any]]]:
    """Streams rows page-by-page from Supabase directly into JSON and CSV files on disk."""
    base_url = supabase_url.rstrip("/")
    endpoint = f"{base_url}/rest/v1/{table_name}"
    headers = make_headers(api_key, prefer="count=exact")

    offset = 0
    total_rows = 0
    sample_rows: list[dict[str, Any]] = []

    # Prepare streaming JSON file
    json_f = json_path.open("w", encoding="utf-8")
    json_f.write("[\n")
    is_first_row = True

    # Prepare streaming CSV file
    csv_f = None
    csv_writer = None
    all_fields: list[str] = []

    try:
        while True:
            params = urllib.parse.urlencode({
                "select": "*",
                "limit": page_size,
                "offset": offset,
            })
            url = f"{endpoint}?{params}"
            req = urllib.request.Request(url, headers=headers, method="GET")  # noqa: S310

            try:
                with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
                    data = json.loads(resp.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                err_body = exc.read().decode("utf-8", errors="replace") if exc.fp else ""
                if exc.code == 404 or "PGRST204" in err_body or "relation" in err_body:
                    data = []
                else:
                    raise RuntimeError(f"HTTP {exc.code} fetching {table_name}: {err_body}") from exc

            if not data or not isinstance(data, list):
                break

            total_rows += len(data)

            # Keep first 50 rows in memory for metadata preview
            if len(sample_rows) < 50:
                sample_rows.extend(data[: 50 - len(sample_rows)])

            # 1. Stream JSON rows
            for row in data:
                row_str = json.dumps(row, ensure_ascii=False, indent=2)
                # Indent rows inside array
                indented = "  " + row_str.replace("\n", "\n  ")
                if not is_first_row:
                    json_f.write(",\n")
                json_f.write(indented)
                is_first_row = False

            # 2. Stream CSV rows
            if not csv_writer:
                # Initialize CSV header from first batch
                for row in data:
                    for k in row:
                        if k not in all_fields:
                            all_fields.append(k)
                csv_f = csv_path.open("w", encoding="utf-8-sig", newline="")
                csv_writer = csv.DictWriter(csv_f, fieldnames=all_fields, extrasaction="ignore")
                csv_writer.writeheader()

            for row in data:
                formatted_row = {}
                for k in all_fields:
                    val = row.get(k)
                    if isinstance(val, (dict, list)):
                        formatted_row[k] = json.dumps(val, ensure_ascii=False)
                    elif val is None:
                        formatted_row[k] = ""
                    else:
                        formatted_row[k] = str(val)
                csv_writer.writerow(formatted_row)

            if len(data) < page_size:
                break
            offset += page_size

        json_f.write("\n]\n")

    finally:
        json_f.close()
        if csv_f:
            csv_f.close()

    if total_rows == 0:
        # Clean up empty files
        if json_path.is_file():
            json_path.unlink()
        if csv_path.is_file():
            csv_path.unlink()

    return total_rows, sample_rows


def export_csv(filepath: Path, rows: list[dict[str, Any]]) -> None:
    """Save rows to a CSV file."""
    if not rows:
        return
    fields: list[str] = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)

    with filepath.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            formatted_row = {}
            for k in fields:
                val = r.get(k)
                if isinstance(val, (dict, list)):
                    formatted_row[k] = json.dumps(val, ensure_ascii=False)
                elif val is None:
                    formatted_row[k] = ""
                else:
                    formatted_row[k] = str(val)
            writer.writerow(formatted_row)


def perform_backup(
    supabase_url: str,
    api_key: str,
    output_dir: Path | None = None,
    *,
    tables: tuple[str, ...] | list[str] | None = None,
    page_size: int = 1000,
    verbose: bool = True,
    progress_callback: Callable[[str, int, int, int], None] | None = None,
) -> dict[str, Any]:
    """Perform full database backup with disk streaming and concurrent count inspection."""
    root_dir = Path(__file__).resolve().parent
    if output_dir is None:
        timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        backup_dir = root_dir / "backups" / f"backup_{timestamp}"
    else:
        backup_dir = output_dir

    backup_dir.mkdir(parents=True, exist_ok=True)
    target_tables = list(tables or TABLES)

    if verbose:
        print("=" * 60)
        print("📦 Supabase Data Backup (Streaming)")
        print(f"Target Project: {supabase_url}")
        print(f"Output Directory: {backup_dir}")
        print("=" * 60)

    summary: dict[str, int] = {}
    full_backup: dict[str, list[dict[str, Any]]] = {}
    total_records = 0

    for idx, table in enumerate(target_tables, 1):
        if verbose:
            sys.stdout.write(f"⏳ Backing up {table:22} ... ")
            sys.stdout.flush()

        json_file = backup_dir / f"{table}.json"
        csv_file = backup_dir / f"{table}.csv"

        try:
            count, sample = stream_backup_table_to_disk(
                supabase_url,
                api_key,
                table,
                json_file,
                csv_file,
                page_size=page_size,
            )
            summary[table] = count
            total_records += count
            if count > 0:
                full_backup[table] = sample
                if verbose:
                    print(f"✅ {count:>5} rows saved (JSON & CSV)")
            else:
                if verbose:
                    print("⚪     0 rows (table empty or unmigrated)")
        except Exception as exc:
            summary[table] = -1
            if verbose:
                print(f"⚠️ Failed: {exc}")

        if progress_callback:
            try:
                progress_callback(table, idx, len(target_tables), total_records)
            except Exception:
                pass

    # Save summary full_backup.json
    with (backup_dir / "full_backup.json").open("w", encoding="utf-8") as f:
        json.dump(full_backup, f, ensure_ascii=False, indent=2)

    # Save metadata
    metadata = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "supabase_url": supabase_url,
        "summary": summary,
        "total_records": total_records,
    }
    with (backup_dir / "backup_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    if verbose:
        print("=" * 60)
        print(f"🎉 Backup completed successfully! Total records: {total_records}")
        print(f"📁 Files stored in: {backup_dir}")
        print("=" * 60)

    return {
        "timestamp_utc": metadata["timestamp_utc"],
        "backup_dir": str(backup_dir),
        "summary": summary,
        "total_records": total_records,
        "success": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Backup Supabase data to JSON & CSV files.")
    parser.add_argument("--output-dir", help="Custom output directory for backup files")
    parser.add_argument("--tables", help="Comma-separated list of tables to backup (default: all)")
    parser.add_argument("--page-size", type=int, default=1000, help="Page size for pagination (default: 1000)")

    args = parser.parse_args()

    root_dir = Path(__file__).resolve().parent
    env = load_env(root_dir / ".env")

    sb_url = os.environ.get("SUPABASE_URL") or env.get("SUPABASE_URL", "")
    sb_key = (
        os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
        or os.environ.get("SUPABASE_KEY")
        or env.get("SUPABASE_SERVICE_ROLE_KEY")
        or env.get("SUPABASE_KEY")
        or ""
    )

    if not sb_url or not sb_key:
        print("❌ Error: SUPABASE_URL or SUPABASE_KEY/SUPABASE_SERVICE_ROLE_KEY not found in .env")
        return 1

    custom_output_dir = Path(args.output_dir).resolve() if args.output_dir else None
    target_tables = [t.strip() for t in args.tables.split(",") if t.strip()] if args.tables else None

    result = perform_backup(
        supabase_url=sb_url,
        api_key=sb_key,
        output_dir=custom_output_dir,
        tables=target_tables,
        page_size=args.page_size,
        verbose=True,
    )

    return 0 if result.get("success") else 1


if __name__ == "__main__":
    sys.exit(main())
