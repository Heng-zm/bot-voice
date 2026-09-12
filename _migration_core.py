"""Shared engine for Supabase Database Migration, Offline Restore, and Backup.

Zero external dependencies (pure Python standard library).
Provides:
- Data-driven schema graph with Kahn's topological sort and wave partitioning
- Checkpoint manager for crash-safe resumability
- Adaptive batch size controller based on latency and payload constraints
- Bounded concurrent worker pool execution
- Concurrent row count inspection
"""

from __future__ import annotations

import collections
import concurrent.futures
from datetime import UTC, datetime
import json
from pathlib import Path
import sys
import time
from typing import Any
import urllib.error
import urllib.parse
import urllib.request

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Canonical Schema Graph - Single Source of Truth for all database operations
SCHEMA_GRAPH: dict[str, dict[str, Any]] = {
    "bot_settings": {
        "pk": "key",
        "conflict": "key",
        "has_created_at": False,
        "depends_on": [],
        "description": "System & runtime configuration",
    },
    "ai_api_keys": {
        "pk": "id",
        "conflict": "key_hash",
        "has_created_at": True,
        "depends_on": [],
        "description": "Admin AI API keys & hash records",
    },
    "user_prefs": {
        "pk": "user_id",
        "conflict": "user_id",
        "has_created_at": True,
        "depends_on": [],
        "description": "User preferences & profiles",
    },
    "donations": {
        "pk": "id",
        "conflict": "id",
        "has_created_at": True,
        "depends_on": ["user_prefs"],
        "description": "Supporter donations & Hall of Fame",
    },
    "feature_requests": {
        "pk": "id",
        "conflict": "id",
        "has_created_at": True,
        "depends_on": ["user_prefs"],
        "description": "User feedback & feature requests",
    },
    "scheduled_broadcasts": {
        "pk": "id",
        "conflict": "id",
        "has_created_at": True,
        "depends_on": [],
        "description": "Scheduled announcement messages",
    },
    "text_cache": {
        "pk": "id",
        "conflict": "chat_id,message_id",
        "has_created_at": True,
        "depends_on": [],
        "description": "Cached message text for callbacks",
    },
    "conversation_history": {
        "pk": "id",
        "conflict": "id",
        "has_created_at": True,
        "depends_on": ["user_prefs"],
        "description": "AI conversation chat history",
    },
    "blocked_users": {
        "pk": "user_id",
        "conflict": "user_id",
        "has_created_at": False,
        "depends_on": [],
        "description": "Blocked user entries",
    },
}


def topo_sort(graph: dict[str, dict[str, Any]]) -> list[str]:
    """Compute topological ordering using Kahn's algorithm.

    Raises ValueError if a cycle is detected.
    """
    in_degree: dict[str, int] = {node: 0 for node in graph}
    adj: dict[str, list[str]] = collections.defaultdict(list)

    for node, meta in graph.items():
        deps = meta.get("depends_on", [])
        for dep in deps:
            if dep in graph:
                adj[dep].append(node)
                in_degree[node] += 1

    queue = collections.deque([n for n, deg in in_degree.items() if deg == 0])
    ordered: list[str] = []

    while queue:
        curr = queue.popleft()
        ordered.append(curr)
        for neighbor in adj[curr]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    if len(ordered) != len(graph):
        unresolved = [n for n, deg in in_degree.items() if deg > 0]
        raise ValueError(f"Cycle detected in schema dependencies among tables: {unresolved}")

    return ordered


def get_dependency_waves(graph: dict[str, dict[str, Any]]) -> list[list[str]]:
    """Partition tables into concurrent execution waves.

    All tables in wave N have all their dependencies satisfied by waves < N,
    allowing tables within the same wave to run concurrently.
    """
    resolved: set[str] = set()
    remaining = dict(graph)
    waves: list[list[str]] = []

    while remaining:
        current_wave: list[str] = []
        for node, meta in list(remaining.items()):
            deps = set(meta.get("depends_on", []))
            # Only consider dependencies present in the active graph
            active_deps = deps.intersection(graph.keys())
            if active_deps.issubset(resolved):
                current_wave.append(node)

        if not current_wave:
            raise ValueError(f"Cycle or unresolvable dependencies in: {list(remaining.keys())}")

        for node in current_wave:
            resolved.add(node)
            del remaining[node]

        waves.append(current_wave)

    return waves


def load_env(env_path: Path) -> dict[str, str]:
    """Parse .env file without external dependencies."""
    env: dict[str, str] = {}
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


def make_headers(api_key: str, *, prefer: str | None = None, content_type: str | None = None) -> dict[str, str]:
    """Build standard Supabase PostgREST request headers."""
    headers = {
        "apikey": api_key,
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
    }
    if prefer:
        headers["Prefer"] = prefer
    if content_type:
        headers["Content-Type"] = content_type
    return headers


def execute_with_retry(
    req: urllib.request.Request,
    *,
    timeout: int = 30,
    max_retries: int = 3,
    backoff_factor: float = 1.5,
) -> tuple[int, bytes, dict[str, str]]:
    """Execute urllib HTTP request with automatic retry for transient errors."""
    last_exc: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
                code = resp.getcode()
                body = resp.read()
                headers = dict(resp.headers)
                return code, body, headers
        except urllib.error.HTTPError as exc:
            last_exc = exc
            body = exc.read() if exc.fp else b""
            # HTTP 413 (Payload Too Large) or non-retryable 4xx
            if exc.code == 413:
                raise
            if exc.code not in {429, 500, 502, 503, 504}:
                err_text = body.decode("utf-8", errors="replace")
                raise RuntimeError(f"HTTP {exc.code}: {err_text}") from exc
            if attempt < max_retries:
                sleep_s = backoff_factor ** attempt
                time.sleep(sleep_s)
        except (urllib.error.URLError, TimeoutError) as exc:
            last_exc = exc
            if attempt < max_retries:
                sleep_s = backoff_factor ** attempt
                time.sleep(sleep_s)

    raise RuntimeError(f"Request failed after {max_retries} attempts: {last_exc}") from last_exc


def fetch_row_count(base_url: str, api_key: str, table_name: str, *, timeout: int = 15) -> int:
    """Query exact table row count using PostgREST HEAD request."""
    url = f"{base_url.rstrip('/')}/rest/v1/{table_name}?select=*"
    headers = make_headers(api_key, prefer="count=exact")
    headers["Range"] = "0-0"
    req = urllib.request.Request(url, headers=headers, method="HEAD")  # noqa: S310
    try:
        _, _, resp_headers = execute_with_retry(req, timeout=timeout, max_retries=2)
        content_range = resp_headers.get("content-range") or resp_headers.get("Content-Range")
        if content_range and "/" in content_range:
            total_str = content_range.split("/")[-1].strip()
            if total_str.isdigit():
                return int(total_str)
    except Exception:
        pass
    return -1


def fetch_all_row_counts(
    base_url: str,
    api_key: str,
    tables: list[str] | tuple[str, ...],
    *,
    max_workers: int = 4,
    timeout: int = 15,
) -> dict[str, int]:
    """Fetch row counts for multiple tables concurrently."""
    counts: dict[str, int] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_tbl = {
            executor.submit(fetch_row_count, base_url, api_key, tbl, timeout=timeout): tbl
            for tbl in tables
        }
        for future in concurrent.futures.as_completed(future_to_tbl):
            tbl = future_to_tbl[future]
            try:
                counts[tbl] = future.result()
            except Exception:
                counts[tbl] = -1
    return counts


class AdaptiveBatcher:
    """Dynamically scales batch size based on request latency and payload errors."""

    def __init__(
        self,
        initial_size: int = 500,
        min_size: int = 50,
        max_size: int = 2000,
        fast_latency_ms: float = 300.0,
    ) -> None:
        self.current_size = initial_size
        self.min_size = min_size
        self.max_size = max_size
        self.fast_latency_ms = fast_latency_ms
        self._consecutive_fast = 0

    def on_success(self, elapsed_s: float) -> None:
        elapsed_ms = elapsed_s * 1000.0
        if elapsed_ms < self.fast_latency_ms:
            self._consecutive_fast += 1
            if self._consecutive_fast >= 2:
                # Grow by 25%
                new_size = min(self.max_size, int(self.current_size * 1.25))
                self.current_size = new_size
                self._consecutive_fast = 0
        else:
            self._consecutive_fast = 0

    def on_payload_error(self) -> None:
        """Cut batch size in half upon HTTP 413 or gateway timeout."""
        self.current_size = max(self.min_size, self.current_size // 2)
        self._consecutive_fast = 0


class Checkpoint:
    """Append-only, crash-safe checkpoint recorder for resumable migration."""

    def __init__(self, filepath: Path) -> None:
        self.filepath = filepath
        self.completed_tables: set[str] = set()
        self.table_offsets: dict[str, int] = {}
        self._load()

    def _load(self) -> None:
        if not self.filepath.is_file():
            return
        try:
            with self.filepath.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    entry = json.loads(line)
                    tbl = entry.get("table")
                    if not tbl:
                        continue
                    if entry.get("completed"):
                        self.completed_tables.add(tbl)
                    offset = entry.get("offset", 0)
                    if offset > self.table_offsets.get(tbl, 0):
                        self.table_offsets[tbl] = offset
        except Exception:
            pass

    def record_progress(self, table: str, offset: int, rows_written: int) -> None:
        self.table_offsets[table] = offset
        entry = {
            "table": table,
            "offset": offset,
            "rows_written": rows_written,
            "timestamp": datetime.now(UTC).isoformat(),
        }
        with self.filepath.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

    def mark_completed(self, table: str, total_rows: int) -> None:
        self.completed_tables.add(table)
        entry = {
            "table": table,
            "completed": True,
            "total_rows": total_rows,
            "timestamp": datetime.now(UTC).isoformat(),
        }
        with self.filepath.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

    def get_offset(self, table: str) -> int:
        return self.table_offsets.get(table, 0)

    def is_completed(self, table: str) -> bool:
        return table in self.completed_tables

    def clear(self) -> None:
        if self.filepath.is_file():
            self.filepath.unlink()
        self.completed_tables.clear()
        self.table_offsets.clear()
