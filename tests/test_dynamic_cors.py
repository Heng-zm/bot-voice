from __future__ import annotations

import asyncio
import unittest
from typing import Annotated
from unittest.mock import patch

from fastapi import Depends, FastAPI, Request
from fastapi.testclient import TestClient
from starlette.middleware.sessions import SessionMiddleware

from app import legacy
from app.api.dependencies import (
    AdminPrincipal,
    require_admin,
    require_admin_write,
)
from app.api.v1 import admin_cors
from app.core.cors import (
    CorsSnapshot,
    DynamicCORSMiddleware,
    DynamicCorsStore,
    InvalidOriginError,
    normalize_origin,
)


class FakePipeline:
    def __init__(self, redis: FakeRedis) -> None:
        self.redis = redis
        self.operations: list[tuple[str, tuple, dict]] = []

    def delete(self, *args, **kwargs):
        self.operations.append(("delete", args, kwargs))
        return self

    def sadd(self, *args, **kwargs):
        self.operations.append(("sadd", args, kwargs))
        return self

    def srem(self, *args, **kwargs):
        self.operations.append(("srem", args, kwargs))
        return self

    def set(self, *args, **kwargs):
        self.operations.append(("set", args, kwargs))
        return self

    def execute(self):
        return [
            getattr(self.redis, name)(*args, **kwargs)
            for name, args, kwargs in self.operations
        ]


class FakeRedis:
    def __init__(self) -> None:
        self.values: dict[str, str] = {}
        self.sets: dict[str, set[str]] = {}
        self.smembers_calls = 0

    def get(self, key: str):
        return self.values.get(key)

    def set(self, key: str, value: str, **kwargs):
        del kwargs
        self.values[key] = value
        return True

    def delete(self, key: str):
        changed = int(key in self.values or key in self.sets)
        self.values.pop(key, None)
        self.sets.pop(key, None)
        return changed

    def sadd(self, key: str, *values: str):
        target = self.sets.setdefault(key, set())
        before = len(target)
        target.update(values)
        return len(target) - before

    def srem(self, key: str, *values: str):
        target = self.sets.setdefault(key, set())
        before = len(target)
        target.difference_update(values)
        return before - len(target)

    def smembers(self, key: str):
        self.smembers_calls += 1
        return set(self.sets.get(key, set()))

    def pipeline(self, transaction: bool = True):
        self.last_pipeline_transaction = transaction
        return FakePipeline(self)


class AsyncFakePipeline:
    def __init__(self, redis: AsyncFakeRedis) -> None:
        self.redis = redis
        self.operations: list[tuple[str, tuple, dict]] = []
        self.reset_called = False

    def delete(self, *args, **kwargs):
        self.operations.append(("delete", args, kwargs))
        return self

    def sadd(self, *args, **kwargs):
        self.operations.append(("sadd", args, kwargs))
        return self

    def srem(self, *args, **kwargs):
        self.operations.append(("srem", args, kwargs))
        return self

    def set(self, *args, **kwargs):
        self.operations.append(("set", args, kwargs))
        return self

    async def execute(self):
        return [
            await getattr(self.redis, name)(*args, **kwargs)
            for name, args, kwargs in self.operations
        ]

    async def reset(self) -> None:
        self.reset_called = True
        self.operations.clear()


class AsyncFakeRedis(FakeRedis):
    async def get(self, key: str):
        return super().get(key)

    async def set(self, key: str, value: str, **kwargs):
        return super().set(key, value, **kwargs)

    async def delete(self, key: str):
        return super().delete(key)

    async def sadd(self, key: str, *values: str):
        return super().sadd(key, *values)

    async def srem(self, key: str, *values: str):
        return super().srem(key, *values)

    async def smembers(self, key: str):
        return super().smembers(key)

    def pipeline(self, transaction: bool = True):
        self.last_pipeline_transaction = transaction
        pipeline = AsyncFakePipeline(self)
        self.last_pipeline = pipeline
        return pipeline


class DynamicCorsStoreTests(unittest.IsolatedAsyncioTestCase):
    async def test_async_redis_pipeline_supports_policy_mutations(self) -> None:
        redis = AsyncFakeRedis()
        store = DynamicCorsStore(
            redis_client=redis,
            redis_prefix="async-tests",
        )
        await store.load(force=True)

        added, added_changed = await store.add(
            "https://admin.example",
            admin_id=7,
        )
        removed, removed_changed = await store.delete(
            "https://admin.example",
            admin_id=7,
        )

        self.assertTrue(added_changed)
        self.assertEqual(("https://admin.example",), added.origins)
        self.assertTrue(removed_changed)
        self.assertEqual((), removed.origins)
        self.assertTrue(redis.last_pipeline.reset_called)

    async def asyncSetUp(self) -> None:
        self.redis = FakeRedis()
        self.store = DynamicCorsStore(
            redis_client=self.redis,
            redis_prefix="tests",
            cache_ttl_seconds=1,
        )
        await self.store.load(force=True)

    async def test_add_delete_and_exact_normalization(self) -> None:
        added, changed = await self.store.add(
            "HTTPS://Example.COM:443/",
            admin_id=7,
        )
        self.assertTrue(changed)
        self.assertEqual(added.origins, ("https://example.com",))
        self.assertTrue(await self.store.is_allowed("https://example.com"))
        self.assertFalse(await self.store.is_allowed("https://sub.example.com"))

        removed, changed = await self.store.delete(
            "https://example.com",
            admin_id=7,
        )
        self.assertTrue(changed)
        self.assertEqual(removed.origins, ())

    async def test_duplicate_add_is_idempotent(self) -> None:
        await self.store.add("https://example.com", admin_id=7)
        _snapshot, changed = await self.store.add(
            "https://example.com/",
            admin_id=7,
        )
        self.assertFalse(changed)

    async def test_concurrent_cache_misses_share_one_backend_read(self) -> None:
        current = self.store.cached_snapshot()
        with self.store._cache_lock:
            self.store._snapshot = CorsSnapshot(
                current.origins,
                current.source,
                0.0,
            )
        calls_before = self.redis.smembers_calls

        snapshots = await asyncio.gather(
            *(self.store.load() for _ in range(12))
        )

        self.assertEqual(self.redis.smembers_calls - calls_before, 1)
        self.assertTrue(all(snapshot == snapshots[0] for snapshot in snapshots))

    async def test_reconfigure_invalidates_cached_policy(self) -> None:
        await self.store.add("https://old.example", admin_id=7)
        replacement = FakeRedis()
        replacement.values[self.store.redis_initialized_key] = "1"
        replacement.sets[self.store.redis_origins_key] = {
            "https://new.example"
        }

        self.store.configure(redis_client=replacement)

        self.assertFalse(await self.store.is_allowed("https://old.example"))
        self.assertTrue(await self.store.is_allowed("https://new.example"))

    def test_invalid_origins_are_rejected(self) -> None:
        invalid = (
            "*",
            "null",
            "ftp://example.com",
            "https://user@example.com",
            "https://example.com/path",
            "https://example.com?query=1",
            "https://exa mple.com",
            "https://*.example.com",
        )
        for origin in invalid:
            with (
                self.subTest(origin=origin),
                self.assertRaises(InvalidOriginError),
            ):
                normalize_origin(origin)


class DynamicCorsMiddlewareTests(unittest.TestCase):
    def setUp(self) -> None:
        self.redis = FakeRedis()
        self.store = DynamicCorsStore(
            redis_client=self.redis,
            redis_prefix="middleware-tests",
        )
        asyncio.run(self.store.load(force=True))
        self.application = FastAPI()
        self.writes = 0

        @self.application.post("/api/admin/test")
        async def admin_write():
            self.writes += 1
            return {"ok": True}

        self.application.add_middleware(
            DynamicCORSMiddleware,
            store=self.store,
            max_age=30,
        )

    def test_disallowed_admin_origin_is_blocked_before_endpoint(self) -> None:
        with TestClient(self.application) as client:
            response = client.post(
                "/api/admin/test",
                headers={"Origin": "https://evil.example"},
            )

        self.assertEqual(response.status_code, 403)
        self.assertEqual(self.writes, 0)
        self.assertNotIn("access-control-allow-origin", response.headers)

    def test_allowed_origin_gets_preflight_and_response_headers(self) -> None:
        asyncio.run(self.store.add("https://admin.example", admin_id=1))

        with TestClient(self.application) as client:
            preflight = client.options(
                "/api/admin/test",
                headers={
                    "Origin": "https://admin.example",
                    "Access-Control-Request-Method": "POST",
                    "Access-Control-Request-Headers": (
                        "Content-Type, X-CSRF-Token, X-Telegram-Init-Data"
                    ),
                },
            )
            response = client.post(
                "/api/admin/test",
                headers={"Origin": "https://admin.example"},
            )

        self.assertEqual(preflight.status_code, 204)
        self.assertEqual(
            preflight.headers["access-control-allow-origin"],
            "https://admin.example",
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.headers["access-control-allow-origin"],
            "https://admin.example",
        )
        self.assertEqual(self.writes, 1)


class AdminCorsApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.redis = FakeRedis()
        self.store = DynamicCorsStore(
            redis_client=self.redis,
            redis_prefix="api-tests",
        )
        asyncio.run(self.store.load(force=True))
        self.application = FastAPI()
        self.application.include_router(admin_cors.router)
        principal = AdminPrincipal(admin_id=99, auth_method="bearer")

        async def authorized():
            return principal

        self.application.dependency_overrides[require_admin] = authorized
        self.application.dependency_overrides[require_admin_write] = authorized

    def test_get_add_and_delete_routes(self) -> None:
        with (
            patch.object(admin_cors, "get_dynamic_cors_store", return_value=self.store),
            TestClient(self.application) as client,
        ):
            added = client.post(
                "/api/admin/cors",
                json={"origin": "https://admin.example"},
            )
            listed = client.get("/api/admin/cors")
            deleted = client.delete(
                "/api/admin/cors",
                params={"origin": "https://admin.example"},
            )

        self.assertEqual(added.status_code, 200)
        self.assertTrue(added.json()["changed"])
        self.assertEqual(listed.json()["origins"], ["https://admin.example"])
        self.assertEqual(deleted.status_code, 200)
        self.assertTrue(deleted.json()["changed"])
        self.assertEqual(deleted.json()["origins"], [])


class NativeAdminAuthenticationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.application = FastAPI()
        self.application.add_middleware(
            SessionMiddleware,
            secret_key="test-session-secret",
        )

        @self.application.get("/test/login")
        async def login(request: Request):
            request.session["web_admin_ok"] = True
            request.session["web_admin_id"] = 42
            request.session["web_csrf_token"] = "csrf-value"
            return {"ok": True}

        @self.application.post("/test/write")
        async def write(
            _principal: Annotated[
                AdminPrincipal,
                Depends(require_admin_write),
            ],
        ):
            return {"ok": True}

    def test_cookie_admin_write_requires_csrf(self) -> None:
        with (
            patch.object(legacy, "ADMIN_IDS", {42}),
            TestClient(self.application) as client,
        ):
            client.get("/test/login")
            missing = client.post("/test/write")
            accepted = client.post(
                "/test/write",
                headers={"X-CSRF-Token": "csrf-value"},
            )

        self.assertEqual(missing.status_code, 403)
        self.assertEqual(accepted.status_code, 200)


if __name__ == "__main__":
    unittest.main()
