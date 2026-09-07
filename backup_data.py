"""Standalone backup utility for Khmer Telegram Bot data.

Uses only Python standard library (zero external dependencies).
Queries the Supabase PostgREST REST API and exports tables to JSON and CSV.

Usage:
    python backup_data.py
"""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import urllib.error
import urllib.parse
import urllib.request

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

TABLES = (
    "user_prefs",
    "bot_settings",
    "conversation_history",
    "text_cache",
    "broadcast_schedules",
    "feature_requests",
    "blocked_users",
    "ai_api_keys",
)


def load_env(env_path: Path) -> dict[str, str]:
    """Parse .env file without external dependencies."""
    env = {}
    if not env_path.is_file():
        return env
    with env_path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, val = line.split("=", 1)
            key = key.strip()
            val = val.strip().strip("'\"")
            env[key] = val
    return env


def fetch_table_rows(
    supabase_url: str,
    api_key: str,
    table_name: str,
    page_size: int = 1000,
) -> list[dict]:
    """Fetch all rows from a Supabase table with pagination."""
    base_url = supabase_url.rstrip("/")
    endpoint = f"{base_url}/rest/v1/{table_name}"
    headers = {
        "apikey": api_key,
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
        "Prefer": "count=exact",
    }

    rows: list[dict] = []
    offset = 0

    while True:
        query_params = urllib.parse.urlencode({
            "select": "*",
            "limit": page_size,
            "offset": offset,
        })
        req_url = f"{endpoint}?{query_params}"
        req = urllib.request.Request(req_url, headers=headers, method="GET")

        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
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
                # Table does not exist in this database
                return []
            raise RuntimeError(f"HTTP {exc.code} fetching {table_name}: {err_body}") from exc
        except Exception as exc:
            raise RuntimeError(f"Failed to fetch {table_name}: {exc}") from exc

    return rows


def export_csv(filepath: Path, rows: list[dict]) -> None:
    """Save rows to a CSV file."""
    if not rows:
        return
    # Collect all fieldnames across all rows
    fields: list[str] = []
    for r in rows:
        for k in r.keys():
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


def main() -> int:
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
        print("Please check your .env file in the project root.")
        return 1

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    backup_dir = root_dir / "backups" / f"backup_{timestamp}"
    backup_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("📦 Supabase Data Backup")
    print(f"Target Project: {sb_url}")
    print(f"Output Directory: {backup_dir}")
    print("=" * 60)

    summary: dict[str, int] = {}
    full_backup: dict[str, list[dict]] = {}

    for table in TABLES:
        sys.stdout.write(f"⏳ Backing up {table:22} ... ")
        sys.stdout.flush()
        try:
            rows = fetch_table_rows(sb_url, sb_key, table)
            count = len(rows)
            summary[table] = count

            if count > 0:
                json_file = backup_dir / f"{table}.json"
                with json_file.open("w", encoding="utf-8") as f:
                    json.dump(rows, f, ensure_ascii=False, indent=2)

                csv_file = backup_dir / f"{table}.csv"
                export_csv(csv_file, rows)

                full_backup[table] = rows
                print(f"✅ {count:>5} rows saved (JSON & CSV)")
            else:
                print("⚪     0 rows (table empty or unmigrated)")
        except Exception as exc:
            summary[table] = -1
            print(f"⚠️ Failed: {exc}")

    # Save full combined backup file
    with (backup_dir / "full_backup.json").open("w", encoding="utf-8") as f:
        json.dump(full_backup, f, ensure_ascii=False, indent=2)

    # Save metadata
    metadata = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "supabase_url": sb_url,
        "summary": summary,
        "total_records": sum(c for c in summary.values() if c > 0),
    }
    with (backup_dir / "backup_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    total = sum(c for c in summary.values() if c > 0)
    print("=" * 60)
    print(f"🎉 Backup completed successfully! Total records: {total}")
    print(f"📁 Files stored in: {backup_dir}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
