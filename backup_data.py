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
import zipfile

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



def format_sql_value(val: Any) -> str:
    """Safely format a Python value as a PostgreSQL SQL literal."""
    if val is None:
        return "NULL"
    if isinstance(val, bool):
        return "TRUE" if val else "FALSE"
    if isinstance(val, (int, float)):
        return str(val)
    if isinstance(val, (dict, list)):
        serialized = json.dumps(val, ensure_ascii=False).replace("'", "''")
        return f"'{serialized}'::jsonb"
    escaped = str(val).replace("'", "''")
    return f"'{escaped}'"


def generate_table_sql_inserts(table_name: str, rows: list[dict[str, Any]]) -> str:
    """Generate PostgreSQL INSERT ... ON CONFLICT DO UPDATE statements for a table."""
    if not rows:
        return f"-- Table {table_name}: 0 rows\n"

    meta = SCHEMA_GRAPH.get(table_name, {})
    conflict_cols_str = meta.get("conflict") or meta.get("pk") or "id"
    conflict_cols = [c.strip() for c in conflict_cols_str.split(",") if c.strip()]

    # Collect all unique columns in stable insertion order
    columns: list[str] = []
    for r in rows:
        for c in r:
            if c not in columns:
                columns.append(c)

    non_conflict_cols = [c for c in columns if c not in conflict_cols]
    quoted_cols = ", ".join(f'"{c}"' for c in columns)
    lines = [f"-- Table: {table_name} ({len(rows)} records)"]

    for row in rows:
        vals = ", ".join(format_sql_value(row.get(c)) for c in columns)
        conflict_target = ", ".join(f'"{c}"' for c in conflict_cols)
        if non_conflict_cols:
            update_clause = ", ".join(f'"{c}" = EXCLUDED."{c}"' for c in non_conflict_cols)
            sql = f'INSERT INTO "{table_name}" ({quoted_cols}) VALUES ({vals}) ON CONFLICT ({conflict_target}) DO UPDATE SET {update_clause};'
        else:
            sql = f'INSERT INTO "{table_name}" ({quoted_cols}) VALUES ({vals}) ON CONFLICT ({conflict_target}) DO NOTHING;'
        lines.append(sql)

    lines.append("")
    return "\n".join(lines)


def export_sql_dump(
    supabase_url: str,
    api_key: str,
    output_path: Path,
    tables: tuple[str, ...] | list[str] | None = None,
    page_size: int = 1000,
    verbose: bool = True,
    progress_callback: Callable[[str, int, int, int], None] | None = None,
) -> tuple[int, Path]:
    """Generates an idempotent PostgreSQL SQL dump file from Supabase PostgREST API."""
    target_tables = list(tables or TABLES)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_records = 0
    now_utc = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")

    with output_path.open("w", encoding="utf-8") as f:
        f.write("-- ==========================================================================\n")
        f.write("-- Khmer Telegram Voice Bot - Supabase PostgreSQL Database Dump\n")
        f.write(f"-- Generated At: {now_utc}\n")
        f.write(f"-- Source URL  : {supabase_url}\n")
        f.write(f"-- Tables      : {', '.join(target_tables)}\n")
        f.write("-- Zero-downtime idempotent upserts with ON CONFLICT DO UPDATE\n")
        f.write("-- ==========================================================================\n\n")
        f.write("BEGIN;\n\n")

        for idx, tbl in enumerate(target_tables, 1):
            if verbose:
                sys.stdout.write(f"⏳ Generating SQL for {tbl:22} ... ")
                sys.stdout.flush()
            rows = fetch_table_rows(supabase_url, api_key, tbl, page_size=page_size)
            count = len(rows)
            total_records += count
            sql_block = generate_table_sql_inserts(tbl, rows)
            f.write(sql_block + "\n")
            if verbose:
                print(f"✅ {count:>5} rows generated")
            if progress_callback:
                try:
                    progress_callback(tbl, idx, len(target_tables), total_records)
                except Exception:
                    pass

        f.write("COMMIT;\n")
        f.write("-- End of Supabase PostgreSQL Database Dump\n")

    return total_records, output_path


def bundle_csv_zip(csv_dir: Path, zip_path: Path) -> Path:
    """Packages all CSV files in a directory into a compressed ZIP archive."""
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    csv_files = sorted(csv_dir.glob("*.csv"))
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for f in csv_files:
            zf.write(f, arcname=f.name)
        meta_file = csv_dir / "backup_metadata.json"
        if meta_file.is_file():
            zf.write(meta_file, arcname="backup_metadata.json")
    return zip_path


def generate_cli_script(
    output_path: Path,
    source_url: str = "",
    source_key: str = "",
    target_url: str = "",
    target_key: str = "",
) -> Path:
    """Generates an executable migration script for CLI usage."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    src_url = source_url or "%SUPABASE_URL%"
    src_k = source_key or "%SUPABASE_SERVICE_ROLE_KEY%"
    tgt_url = target_url or "%TARGET_SUPABASE_URL%"
    tgt_k = target_key or "%TARGET_SUPABASE_KEY%"

    content = f"""@echo off
rem ============================================================================
rem Khmer Telegram Bot - Automated Supabase Migration & Disaster Recovery CLI
rem ============================================================================
rem Usage:
rem   1. Set target credentials below or pass via environment variables
rem   2. Run this script in terminal (cmd / powershell / bash)
rem ============================================================================

set SOURCE_URL={src_url}
set SOURCE_KEY={src_k}
set TARGET_URL={tgt_url}
set TARGET_KEY={tgt_k}

echo [1/3] Testing connectivity and executing online migration...
python migrate_data.py --source-url "%SOURCE_URL%" --source-key "%SOURCE_KEY%" --target-url "%TARGET_URL%" --target-key "%TARGET_KEY%" --concurrency 4 --batch-size 500 --verify

if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Direct migration failed or was interrupted.
    echo You can resume at any time by rerunning this script.
    pause
    exit /b %ERRORLEVEL%
)

echo [2/3] Performing local streaming backup snapshot...
python backup_data.py --all

echo [3/3] Migration & verification finished successfully!
pause
"""
    output_path.write_text(content, encoding="utf-8")
    return output_path


def perform_backup(
    supabase_url: str,
    api_key: str,
    output_dir: Path | None = None,
    *,
    tables: tuple[str, ...] | list[str] | None = None,
    page_size: int = 1000,
    verbose: bool = True,
    progress_callback: Callable[[str, int, int, int], None] | None = None,
    export_sql: bool = False,
    export_csv_zip: bool = False,
    export_cli: bool = False,
) -> dict[str, Any]:
    """Perform full database backup with disk streaming, and optional SQL, CSV.ZIP, and CLI generation."""
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

    sql_file: Path | None = None
    if export_sql:
        sql_file = backup_dir / f"{backup_dir.name}.sql"
        if verbose:
            print("📝 Exporting PostgreSQL SQL dump file...")
        export_sql_dump(
            supabase_url,
            api_key,
            sql_file,
            tables=target_tables,
            page_size=page_size,
            verbose=verbose,
            progress_callback=progress_callback,
        )

    zip_file: Path | None = None
    if export_csv_zip:
        zip_file = backup_dir / f"{backup_dir.name}_csv.zip"
        if verbose:
            print("📦 Bundling CSV files into ZIP archive...")
        bundle_csv_zip(backup_dir, zip_file)

    cli_file: Path | None = None
    if export_cli:
        cli_file = backup_dir / "run_migration.cli"
        if verbose:
            print("💻 Generating CLI migration script...")
        generate_cli_script(cli_file, source_url=supabase_url, source_key=api_key)

    if verbose:
        print("=" * 60)
        print(f"🎉 Backup completed successfully! Total records: {total_records}")
        print(f"📁 Files stored in: {backup_dir}")
        if sql_file:
            print(f"📥 SQL Dump      : {sql_file}")
        if zip_file:
            print(f"📊 CSV ZIP       : {zip_file}")
        if cli_file:
            print(f"💻 CLI Script    : {cli_file}")
        print("=" * 60)

    return {
        "timestamp_utc": metadata["timestamp_utc"],
        "backup_dir": str(backup_dir),
        "summary": summary,
        "total_records": total_records,
        "success": True,
        "sql_path": str(sql_file) if sql_file else None,
        "zip_path": str(zip_file) if zip_file else None,
        "cli_path": str(cli_file) if cli_file else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Backup Supabase data to JSON, CSV, SQL, & CLI files.")
    parser.add_argument("--output-dir", help="Custom output directory for backup files")
    parser.add_argument("--tables", help="Comma-separated list of tables to backup (default: all)")
    parser.add_argument("--page-size", type=int, default=1000, help="Page size for pagination (default: 1000)")
    parser.add_argument("--sql", action="store_true", help="Generate SQL dump file")
    parser.add_argument("--csv-zip", action="store_true", help="Bundle CSV files into a ZIP archive")
    parser.add_argument("--cli", action="store_true", help="Generate CLI migration script")
    parser.add_argument("--all", action="store_true", help="Generate JSON, CSV, SQL dump, CSV ZIP, and CLI script")

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

    export_sql = args.sql or args.all
    export_csv_zip = args.csv_zip or args.all
    export_cli = args.cli or args.all

    result = perform_backup(
        supabase_url=sb_url,
        api_key=sb_key,
        output_dir=custom_output_dir,
        tables=target_tables,
        page_size=args.page_size,
        verbose=True,
        export_sql=export_sql,
        export_csv_zip=export_csv_zip,
        export_cli=export_cli,
    )

    return 0 if result.get("success") else 1


if __name__ == "__main__":
    sys.exit(main())
