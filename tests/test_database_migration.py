"""Unit tests for Supabase Database Migration, Restore, Backup, and Admin Management.

Tests:
- Data-driven topological sorting (Kahn's algorithm, cycle rejection, linearization)
- Multi-tier dependency wave partitioning for safe concurrency
- Checkpoint crash recovery and resumability
- Adaptive batch size scaling (grows on fast latency, shrinks on HTTP 413)
- Disk streaming backup without RAM bloat
- Telegram admin 30s cache and non-blocking background backup execution
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from _migration_core import (
    SCHEMA_GRAPH,
    AdaptiveBatcher,
    Checkpoint,
    get_dependency_waves,
    make_headers,
    topo_sort,
)
import backup_data
import migrate_data
import restore_data
from app.legacy import get_admin_dashboard_kb, get_admin_db_kb


class TestSchemaGraphAndTopologicalSort(unittest.TestCase):
    """Verify data-driven schema graph, topological linearization, and cycle detection."""

    def test_topo_sort_valid_linearization(self) -> None:
        """Every table must appear strictly after all tables it depends on."""
        order = topo_sort(SCHEMA_GRAPH)
        self.assertEqual(len(order), len(SCHEMA_GRAPH))

        seen = set()
        for tbl in order:
            deps = SCHEMA_GRAPH[tbl].get("depends_on", [])
            for dep in deps:
                self.assertIn(dep, seen, f"Table '{tbl}' appeared before its dependency '{dep}'")
            seen.add(tbl)

    def test_topo_sort_detects_cycle(self) -> None:
        """Kahn's algorithm must reject cycles with ValueError."""
        cyclic_graph = {
            "table_a": {"depends_on": ["table_b"]},
            "table_b": {"depends_on": ["table_a"]},
        }
        with self.assertRaises(ValueError) as ctx:
            topo_sort(cyclic_graph)
        self.assertIn("Cycle detected", str(ctx.exception))

    def test_dependency_waves_partitioning(self) -> None:
        """Independent tables belong to Wave 1; dependents belong to later waves."""
        waves = get_dependency_waves(SCHEMA_GRAPH)
        self.assertGreaterEqual(len(waves), 2)

        # Wave 0 must contain user_prefs, bot_settings, etc.
        self.assertIn("user_prefs", waves[0])
        self.assertIn("bot_settings", waves[0])
        self.assertIn("ai_api_keys", waves[0])

        # Wave 1 must contain donations and conversation_history (depend on user_prefs)
        self.assertIn("donations", waves[1])
        self.assertIn("conversation_history", waves[1])
        self.assertIn("feature_requests", waves[1])

    def test_ephemeral_locks_excluded(self) -> None:
        """bot_locks contains short-lived leases and should never be in SCHEMA_GRAPH."""
        self.assertNotIn("bot_locks", SCHEMA_GRAPH)

    def test_conflict_keys_configured(self) -> None:
        for tbl, meta in SCHEMA_GRAPH.items():
            conflict = meta.get("conflict") or meta.get("pk")
            self.assertIsNotNone(conflict, f"Table {tbl} missing conflict/pk definition")


class TestCheckpointManager(unittest.TestCase):
    """Test append-only crash-safe checkpoint recorder for resumable migration."""

    def test_checkpoint_record_and_resume(self) -> None:
        with tempfile.NamedTemporaryFile("w", delete=False) as f:
            cp_path = Path(f.name)
        try:
            cp = Checkpoint(cp_path)
            self.assertEqual(cp.get_offset("user_prefs"), 0)
            self.assertFalse(cp.is_completed("user_prefs"))

            # Record partial progress (e.g. 500 rows written)
            cp.record_progress("user_prefs", offset=500, rows_written=500)

            # Re-instantiate to simulate crash recovery
            cp2 = Checkpoint(cp_path)
            self.assertEqual(cp2.get_offset("user_prefs"), 500)
            self.assertFalse(cp2.is_completed("user_prefs"))

            # Mark completed
            cp2.mark_completed("user_prefs", total_rows=1200)

            # Re-instantiate again
            cp3 = Checkpoint(cp_path)
            self.assertTrue(cp3.is_completed("user_prefs"))

            # Clear
            cp3.clear()
            self.assertFalse(cp_path.exists())
            self.assertEqual(cp3.get_offset("user_prefs"), 0)
        finally:
            if cp_path.exists():
                cp_path.unlink()


class TestAdaptiveBatcher(unittest.TestCase):
    """Test adaptive batch sizing based on latency and payload size."""

    def test_grows_on_fast_responses(self) -> None:
        batcher = AdaptiveBatcher(initial_size=500, max_size=2000, fast_latency_ms=300.0)
        self.assertEqual(batcher.current_size, 500)

        # 1 fast response (<300ms)
        batcher.on_success(elapsed_s=0.15)
        self.assertEqual(batcher.current_size, 500)

        # 2nd consecutive fast response -> scales by 25% (500 * 1.25 = 625)
        batcher.on_success(elapsed_s=0.20)
        self.assertEqual(batcher.current_size, 625)

    def test_halves_on_payload_error(self) -> None:
        batcher = AdaptiveBatcher(initial_size=1000, min_size=50)
        batcher.on_payload_error()
        self.assertEqual(batcher.current_size, 500)

        batcher.on_payload_error()
        self.assertEqual(batcher.current_size, 250)

    def test_clamps_to_boundaries(self) -> None:
        batcher = AdaptiveBatcher(initial_size=1900, max_size=2000, min_size=50)
        # Fast consecutive
        batcher.on_success(0.1)
        batcher.on_success(0.1)
        self.assertEqual(batcher.current_size, 2000)

        # Extreme shrink
        for _ in range(10):
            batcher.on_payload_error()
        self.assertEqual(batcher.current_size, 50)


class TestMigrateAndRestoreExecution(unittest.TestCase):
    """Test online migration and offline restore CLI execution flows."""

    def test_make_headers(self) -> None:
        headers = make_headers("dummy_key", prefer="count=exact", content_type="application/json")
        self.assertEqual(headers["apikey"], "dummy_key")
        self.assertEqual(headers["Authorization"], "Bearer dummy_key")
        self.assertEqual(headers["Prefer"], "count=exact")
        self.assertEqual(headers["Content-Type"], "application/json")

    @patch("migrate_data.test_connection", return_value=True)
    @patch("migrate_data.fetch_row_count", return_value=1)
    @patch("migrate_data.fetch_batch_rows", return_value=[{"key": "tts_enabled", "value": "1"}])
    def test_run_migration_dry_run(
        self,
        mock_batch: MagicMock,
        mock_count: MagicMock,
        mock_conn: MagicMock,
    ) -> None:
        ret = migrate_data.run_migration(
            source_url="https://source.supabase.co",
            source_key="source_key",
            target_url="https://target.supabase.co",
            target_key="target_key",
            batch_size=10,
            dry_run=True,
            fresh=True,
            tables_filter=["bot_settings"],
        )
        self.assertEqual(ret, 0)

    @patch("restore_data.test_connection", return_value=True)
    def test_run_restore_dry_run(self, mock_conn: MagicMock) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            backup_dir = Path(tmpdir)
            with (backup_dir / "bot_settings.json").open("w", encoding="utf-8") as f:
                json.dump([{"key": "test_k", "value": "test_v"}], f)

            ret = restore_data.run_restore(
                backup_dir=backup_dir,
                target_url="https://target.supabase.co",
                target_key="test_key",
                dry_run=True,
                fresh=True,
                tables_filter=["bot_settings"],
            )
            self.assertEqual(ret, 0)


class TestStreamingBackup(unittest.TestCase):
    """Test streamed disk backups without memory buffering."""

    @patch("backup_data.stream_backup_table_to_disk")
    def test_perform_backup_with_progress_callback(self, mock_stream: MagicMock) -> None:
        mock_stream.return_value = (5, [{"key": "maintenance_mode", "value": "0"}])
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir) / "test_backup"
            progress_calls = []

            def on_progress(tbl: str, idx: int, total: int, rows: int) -> None:
                progress_calls.append((tbl, idx, total, rows))

            result = backup_data.perform_backup(
                supabase_url="https://test.supabase.co",
                api_key="test_key",
                output_dir=out_dir,
                tables=["bot_settings"],
                verbose=False,
                progress_callback=on_progress,
            )

            self.assertTrue(result["success"])
            self.assertEqual(result["total_records"], 5)
            self.assertEqual(len(progress_calls), 1)
            self.assertEqual(progress_calls[0][0], "bot_settings")
            self.assertTrue((out_dir / "full_backup.json").is_file())
            self.assertTrue((out_dir / "backup_metadata.json").is_file())


class TestAdminDatabaseUIAndCache(unittest.TestCase):
    """Test Admin Telegram Database UI, 30s Caching, and Non-blocking Background Tasks."""

    def test_admin_dashboard_has_database_button(self) -> None:
        kb = get_admin_dashboard_kb()
        callbacks = [btn.callback_data for row in kb.inline_keyboard for btn in row]
        self.assertIn("admin_db", callbacks)

    def test_admin_db_keyboard_buttons(self) -> None:
        kb = get_admin_db_kb()
        callbacks = [btn.callback_data for row in kb.inline_keyboard for btn in row]
        self.assertIn("admin_db_refresh", callbacks)
        self.assertIn("admin_db_backup", callbacks)
        self.assertIn("admin_home", callbacks)
        self.assertIn("admin_close", callbacks)

    @patch("app.legacy.fetch_all_row_counts")
    def test_30s_cache_avoids_redundant_queries(self, mock_counts: MagicMock) -> None:
        mock_counts.return_value = {t: 100 for t in backup_data.TABLES}
        from app import legacy
        legacy._DB_STATUS_CACHE["timestamp"] = 0.0
        legacy._DB_STATUS_CACHE["text"] = ""

        with patch.dict("os.environ", {"SUPABASE_URL": "https://testxyz.supabase.co", "SUPABASE_KEY": "dummy"}):
            # First call: populates cache
            text1 = asyncio.run(legacy._get_admin_db_text())
            self.assertEqual(mock_counts.call_count, 1)

            # Second call immediately after: hits cache
            text2 = asyncio.run(legacy._get_admin_db_text())
            self.assertEqual(mock_counts.call_count, 1)
            self.assertEqual(text1, text2)

            # Forced call: bypasses cache
            text3 = asyncio.run(legacy._get_admin_db_text(force=True))
            self.assertEqual(mock_counts.call_count, 2)
            self.assertEqual(text1, text3)

    def test_cmd_dbstatus_permissions(self) -> None:
        from app.services.telegram.commands import cmd_dbstatus

        mock_update = MagicMock()
        mock_msg = MagicMock()
        mock_msg.reply_text = AsyncMock()
        mock_update.effective_message = mock_msg
        mock_context = MagicMock()

        # Non-admin
        mock_update.effective_user.id = 999
        with patch("app.services.telegram.commands._is_admin", return_value=False):
            asyncio.run(cmd_dbstatus(mock_update, mock_context))
        self.assertIn("Database Status", mock_msg.reply_text.call_args[0][0])

        # Admin
        mock_msg.reply_text.reset_mock()
        mock_update.effective_user.id = 123
        with patch("app.services.telegram.commands._is_admin", return_value=True), \
             patch("app.legacy._get_admin_db_text", new_callable=AsyncMock) as mock_text:
            mock_text.return_value = "DB Status Live"
            asyncio.run(cmd_dbstatus(mock_update, mock_context))
        self.assertIn("DB Status Live", mock_msg.reply_text.call_args[0][0])

    @patch("app.legacy._admin_trigger_backup", new_callable=AsyncMock)
    def test_cmd_dbbackup_non_blocking_dispatch(self, mock_backup: MagicMock) -> None:
        from app.services.telegram.commands import cmd_dbbackup

        mock_update = MagicMock()
        mock_update.effective_user.id = 123
        mock_msg = MagicMock()
        mock_update.effective_message = mock_msg
        mock_context = MagicMock()

        with patch("app.services.telegram.commands._is_admin", return_value=True):
            asyncio.run(cmd_dbbackup(mock_update, mock_context))

        mock_backup.assert_called_once_with(mock_msg, 123)


if __name__ == "__main__":
    unittest.main()
