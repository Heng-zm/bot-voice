from __future__ import annotations

import os
import unittest
from pathlib import Path
from unittest.mock import patch

from app.services.db.locks import BOT_LOCKS_SQL, SupabaseLockService


class _Response:
    def __init__(self, data) -> None:
        self.data = data


class _Execute:
    def __init__(self, response=None, error: Exception | None = None) -> None:
        self.response = response
        self.error = error

    def execute(self):
        if self.error is not None:
            raise self.error
        return self.response


class _TableQuery:
    def __init__(self, client: _FallbackClient) -> None:
        self.client = client
        self.operation = ""
        self.filters: list[tuple[str, str, object]] = []

    def update(self, values: dict):
        self.operation = "update"
        self.client.last_update = dict(values)
        return self

    def upsert(self, values: dict, **options):
        self.operation = "upsert"
        self.client.last_upsert = (dict(values), dict(options))
        return self

    def delete(self):
        self.operation = "delete"
        return self

    def select(self, columns: str):
        self.operation = "select"
        self.client.selected_columns = columns
        return self

    def eq(self, column: str, value):
        self.filters.append(("eq", column, value))
        return self

    def lt(self, column: str, value):
        self.filters.append(("lt", column, value))
        return self

    def limit(self, value: int):
        self.client.limit_value = value
        return self

    def execute(self):
        self.client.operations.append((self.operation, list(self.filters)))
        if self.operation == "update":
            return _Response(self.client.update_results.pop(0))
        if self.operation == "upsert":
            return _Response(self.client.upsert_result)
        if self.operation == "delete":
            return _Response(self.client.delete_result)
        if self.operation == "select":
            return _Response(self.client.select_result)
        raise AssertionError(f"Unexpected operation: {self.operation}")


class _FallbackClient:
    def __init__(self) -> None:
        self.rpc_calls = 0
        self.table_calls: list[str] = []
        self.operations: list[tuple[str, list[tuple[str, str, object]]]] = []
        self.update_results: list[list[dict]] = [[], []]
        self.upsert_result: list[dict] = []
        self.delete_result: list[dict] = []
        self.select_result: list[dict] = []
        self.last_update: dict = {}
        self.last_upsert: tuple[dict, dict] | None = None
        self.selected_columns = ""
        self.limit_value = 0

    def rpc(self, _name: str, _params: dict):
        self.rpc_calls += 1
        return _Execute(error=RuntimeError("PGRST202 acquire_bot_lock not found"))

    def table(self, name: str):
        self.table_calls.append(name)
        return _TableQuery(self)


class SupabaseLockServiceTests(unittest.TestCase):
    def test_atomic_rpc_is_preferred(self) -> None:
        class Client:
            def __init__(self) -> None:
                self.params = None

            def rpc(self, name: str, params: dict):
                self.params = (name, params)
                return _Execute(_Response(True))

            def table(self, _name: str):
                raise AssertionError("RPC success must not use the fallback")

        client = Client()

        acquired = SupabaseLockService().acquire(client, "scheduler", "worker-a", 90)

        self.assertTrue(acquired)
        self.assertEqual("acquire_bot_lock", client.params[0])
        self.assertEqual(90, client.params[1]["p_ttl_seconds"])

    def test_duplicate_first_acquisition_uses_conflict_free_upsert(self) -> None:
        client = _FallbackClient()
        service = SupabaseLockService()

        acquired = service.acquire(client, "scheduler", "worker-b", 90)

        self.assertFalse(acquired)
        self.assertEqual(1, client.rpc_calls)
        self.assertEqual(["update", "update", "upsert"], [row[0] for row in client.operations])
        self.assertIsNotNone(client.last_upsert)
        _values, options = client.last_upsert
        self.assertEqual("lock_key", options["on_conflict"])
        self.assertTrue(options["ignore_duplicates"])

        # Missing-RPC capability is cached for this client. A later first-row
        # insertion succeeds without producing another missing-function error.
        client.update_results = [[], []]
        client.upsert_result = [{"lock_key": "scheduler", "owner": "worker-b"}]
        self.assertTrue(service.acquire(client, "scheduler", "worker-b", 90))
        self.assertEqual(1, client.rpc_calls)

    def test_owner_renewal_stops_before_insert_fallback(self) -> None:
        client = _FallbackClient()
        client.update_results = [[{"lock_key": "scheduler"}]]

        acquired = SupabaseLockService().acquire(client, "scheduler", "worker-a", 30)

        self.assertTrue(acquired)
        self.assertEqual(["update"], [row[0] for row in client.operations])
        self.assertIsNone(client.last_upsert)

    def test_lock_inputs_are_bounded(self) -> None:
        service = SupabaseLockService()
        client = _FallbackClient()
        for lock_key, owner, ttl in (("", "owner", 30), ("key", "", 30), ("key", "owner", 0)):
            with self.subTest(lock_key=lock_key, owner=owner, ttl=ttl), self.assertRaises(ValueError):
                service.acquire(client, lock_key, owner, ttl)

    def test_checked_in_sql_is_atomic_and_service_role_only(self) -> None:
        migration = (
            Path(__file__).resolve().parents[1]
            / "scripts"
            / "supabase_bot_locks.sql"
        ).read_text(encoding="utf-8")
        for sql in (BOT_LOCKS_SQL, migration):
            normalized = sql.lower()
            self.assertIn("on conflict (lock_key) do update", normalized)
            self.assertIn("current_lock.locked_until <= lease_now", normalized)
            self.assertIn("security definer", normalized)
            self.assertIn("grant execute", normalized)
            self.assertIn("to service_role", normalized)


class TelegramLeaderLockTests(unittest.IsolatedAsyncioTestCase):
    async def test_async_leader_rpc_uses_migration_parameter_names(self) -> None:
        from app import legacy

        class AsyncExecute:
            async def execute(self):
                return _Response(True)

        class AsyncClient:
            def __init__(self) -> None:
                self.call: tuple[str, dict] | None = None

            def rpc(self, name: str, params: dict):
                self.call = (name, dict(params))
                return AsyncExecute()

        async def direct_db_call(_name, factory, **_options):
            return await factory()

        client = AsyncClient()
        with (
            patch.dict(
                os.environ,
                {"TELEGRAM_LEADER_STORE": "supabase"},
                clear=False,
            ),
            patch.multiple(
                legacy,
                create=True,
                redis_client_async=None,
                redis_client=None,
                supabase_async=client,
                supabase=object(),
                TELEGRAM_ACTIVE_LOCK_ENABLED=True,
                TELEGRAM_ACTIVE_LOCK_KEY="telegram_webhook_owner",
                TELEGRAM_ACTIVE_LOCK_TTL_S=90,
                _TELEGRAM_LEADER_OWNER_ID_CACHE="instance-a",
            ),
            patch.object(legacy, "db_call", new=direct_db_call),
        ):
            acquired = await legacy._telegram_leader_acquire()

        self.assertTrue(acquired)
        self.assertEqual("acquire_bot_lock", client.call[0])
        self.assertEqual(
            {
                "p_lock_key": "telegram_webhook_owner",
                "p_owner": "instance-a",
                "p_ttl_seconds": 90,
            },
            client.call[1],
        )

    def test_disabled_redis_and_unused_pooler_do_not_warn(self) -> None:
        from app import legacy

        with (
            patch.dict(
                os.environ,
                {"REDIS_ENABLED": "false", "DISABLE_REDIS": "true"},
                clear=False,
            ),
            patch.multiple(
                legacy,
                supabase=object(),
                SUPABASE_DB_POOLER_URL="",
            ),
            patch.object(
                legacy,
                "_web_stable_secret_configured",
                return_value=False,
            ),
            patch.object(legacy.logger, "warning") as warning,
        ):
            legacy.startup_self_check()

        output = " ".join(str(call) for call in warning.call_args_list)
        self.assertNotIn("Redis WEB_SECRET_KEY", output)
        self.assertNotIn("REDIS_URL is missing", output)
        self.assertNotIn("SUPABASE_DB_POOLER_URL", output)


if __name__ == "__main__":
    unittest.main()
