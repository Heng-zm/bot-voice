from __future__ import annotations

import asyncio
import unittest

from app.core.admin_management import (
    AdminConfirmationError,
    LastAdministratorError,
    RedisAdminManager,
)
from app.services.ai.providers import NoProviderAvailable, ProviderManager
from app.services.jobs.queue import RedisJobQueue, RedisJobWorker


class FakeRedis:
    def __init__(self) -> None:
        self.strings: dict[str, str] = {}
        self.hashes: dict[str, dict[str, str]] = {}
        self.sets: dict[str, set[str]] = {}
        self.zsets: dict[str, dict[str, float]] = {}
        self.streams: dict[str, list[tuple[str, dict[str, str]]]] = {}
        self._stream_sequence = 0

    def set(
        self,
        key: str,
        value: str,
        *,
        nx: bool = False,
        ex: int | None = None,
    ) -> bool:
        del ex
        if nx and key in self.strings:
            return False
        self.strings[key] = str(value)
        return True

    def get(self, key: str) -> str | None:
        return self.strings.get(key)

    def smembers(self, key: str) -> set[str]:
        return set(self.sets.get(key, set()))

    def hgetall(self, key: str) -> dict[str, str]:
        return dict(self.hashes.get(key, {}))

    def hset(self, key: str, *, mapping: dict[str, str]) -> int:
        self.hashes.setdefault(key, {}).update(
            {str(name): str(value) for name, value in mapping.items()}
        )
        return len(mapping)

    def zrevrange(self, key: str, start: int, end: int):
        ordered = [
            member
            for member, _score in sorted(
                self.zsets.get(key, {}).items(),
                key=lambda item: item[1],
                reverse=True,
            )
        ]
        return ordered[start : end + 1]

    def zremrangebyscore(self, key: str, minimum, maximum) -> int:
        del minimum
        cutoff = float(maximum)
        members = self.zsets.setdefault(key, {})
        removed = [member for member, score in members.items() if score <= cutoff]
        for member in removed:
            members.pop(member, None)
        return len(removed)

    def zcard(self, key: str) -> int:
        return len(self.zsets.get(key, {}))

    def xrevrange(
        self,
        key: str,
        *,
        max: str,
        min: str,
        count: int,
    ) -> list[tuple[str, dict[str, str]]]:
        del max, min
        return list(reversed(self.streams.get(key, [])))[:count]

    def _zadd(self, key: str, score: float, member: str) -> None:
        self.zsets.setdefault(key, {})[member] = float(score)

    def _zrem(self, key: str, member: str) -> None:
        self.zsets.setdefault(key, {}).pop(member, None)

    def _audit(
        self,
        key: str,
        timestamp: str,
        actor_id: str,
        action: str,
        target_id: str,
        changed: bool,
    ) -> None:
        self._stream_sequence += 1
        entry_id = f"{self._stream_sequence}-0"
        self.streams.setdefault(key, []).append(
            (
                entry_id,
                {
                    "timestamp": timestamp,
                    "actor_id": actor_id,
                    "action": action,
                    "target_id": target_id,
                    "changed": "1" if changed else "0",
                },
            )
        )

    def eval(self, script: str, number_of_keys: int, *values):
        keys = list(values[:number_of_keys])
        args = [str(value) for value in values[number_of_keys:]]
        if "bot_voice:admin_add_v1" in script:
            return self._admin_add(keys, args)
        if "bot_voice:admin_remove_v1" in script:
            return self._admin_remove(keys, args)
        if "bot_voice:enqueue_v1" in script:
            return self._enqueue(keys, args)
        if "bot_voice:claim_v1" in script:
            return self._claim(keys, args)
        if "bot_voice:renew_v1" in script:
            return self._renew(keys, args)
        if "bot_voice:complete_v1" in script:
            return self._complete(keys, args)
        if "bot_voice:fail_v1" in script:
            return self._fail(keys, args)
        if "bot_voice:cancel_v1" in script:
            return self._cancel(keys, args)
        if "bot_voice:retry_v1" in script:
            return self._retry(keys, args)
        raise AssertionError("Unexpected Lua script")

    def _admin_add(self, keys: list[str], args: list[str]) -> list[int]:
        confirmation_key, admins_key, audit_key = keys
        expected, timestamp, target, actor, _max_length = args
        if self.strings.get(confirmation_key) != expected:
            return [-2, 0]
        self.strings.pop(confirmation_key)
        admins = self.sets.setdefault(admins_key, set())
        changed = target not in admins
        admins.add(target)
        self._audit(audit_key, timestamp, actor, "admin_add", target, changed)
        return [1, int(changed)]

    def _admin_remove(self, keys: list[str], args: list[str]) -> list[int]:
        confirmation_key, admins_key, audit_key = keys
        expected, timestamp, target, actor, _max_length = args
        if self.strings.get(confirmation_key) != expected:
            return [-2, 0]
        self.strings.pop(confirmation_key)
        admins = self.sets.setdefault(admins_key, set())
        if target not in admins:
            self._audit(
                audit_key,
                timestamp,
                actor,
                "admin_remove",
                target,
                False,
            )
            return [1, 0]
        if len(admins) <= 1:
            self._audit(
                audit_key,
                timestamp,
                actor,
                "admin_remove_denied_final",
                target,
                False,
            )
            return [-1, 0]
        admins.remove(target)
        self._audit(
            audit_key,
            timestamp,
            actor,
            "admin_remove",
            target,
            True,
        )
        return [1, 1]

    def _enqueue(self, keys: list[str], args: list[str]):
        ready_key, job_key, idempotency_key = keys
        (
            job_id,
            job_type,
            payload,
            priority,
            created_at,
            available_at,
            max_attempts,
            timeout_seconds,
            use_idempotency,
            _idempotency_ttl,
            max_queued_jobs,
        ) = args
        if use_idempotency == "1" and idempotency_key in self.strings:
            return [self.strings[idempotency_key], 0]
        if job_key in self.hashes:
            return [job_id, 0]
        if len(self.zsets.get(ready_key, {})) >= int(max_queued_jobs):
            return ["", -1]
        self.hashes[job_key] = {
            "id": job_id,
            "type": job_type,
            "payload": payload,
            "state": "queued",
            "priority": priority,
            "created_at": created_at,
            "available_at": available_at,
            "attempts": "0",
            "max_attempts": max_attempts,
            "timeout_seconds": timeout_seconds,
            "cancel_requested": "0",
            "progress_percent": "0",
            "progress_stage": "queued",
            "progress_detail": "",
            "updated_at": created_at,
        }
        self._zadd(ready_key, float(available_at), job_id)
        if use_idempotency == "1":
            self.strings[idempotency_key] = job_id
        return [job_id, 1]

    def _claim(self, keys: list[str], args: list[str]):
        ready_key, leased_key, dead_key, cancelled_key = keys
        now, worker, token, deadline, _retention, job_prefix = args
        now_value = float(now)
        expired = [
            job_id
            for job_id, score in self.zsets.get(leased_key, {}).items()
            if score <= now_value
        ]
        for job_id in expired:
            data = self.hashes[f"{job_prefix}{job_id}"]
            if data.get("state") == "running":
                if data.get("cancel_requested") == "1":
                    data.update(
                        state="cancelled",
                        completed_at=now,
                        last_error="cancelled while worker lease was unavailable",
                        progress_stage="cancelled",
                        updated_at=now,
                    )
                    self._zadd(cancelled_key, now_value, job_id)
                elif int(data["attempts"]) >= int(data["max_attempts"]):
                    data.update(
                        state="dead",
                        completed_at=now,
                        last_error="worker lease expired",
                        progress_stage="dead",
                        updated_at=now,
                    )
                    self._zadd(dead_key, now_value, job_id)
                else:
                    data.update(
                        state="queued",
                        available_at=now,
                        last_error="worker lease expired",
                        progress_stage="retrying",
                        updated_at=now,
                    )
                    for field in ("lease_token", "lease_deadline", "worker_id"):
                        data.pop(field, None)
                    self._zadd(ready_key, now_value, job_id)
            self._zrem(leased_key, job_id)
        due = [
            job_id
            for job_id, score in self.zsets.get(ready_key, {}).items()
            if score <= now_value
            and self.hashes[f"{job_prefix}{job_id}"].get("state") == "queued"
        ]
        if not due:
            return None
        selected = min(
            due,
            key=lambda job_id: (
                -int(self.hashes[f"{job_prefix}{job_id}"]["priority"]),
                float(self.hashes[f"{job_prefix}{job_id}"]["created_at"]),
            ),
        )
        data = self.hashes[f"{job_prefix}{selected}"]
        data["attempts"] = str(int(data["attempts"]) + 1)
        data.update(
            state="running",
            worker_id=worker,
            lease_token=token,
            lease_deadline=deadline,
            started_at=now,
            cancel_requested="0",
            progress_stage="running",
            updated_at=now,
        )
        self._zrem(ready_key, selected)
        self._zadd(leased_key, float(deadline), selected)
        return selected

    def _renew(self, keys: list[str], args: list[str]) -> int:
        leased_key, job_key = keys
        token, deadline, job_id, updated_at = args
        data = self.hashes[job_key]
        if data.get("state") != "running" or data.get("lease_token") != token:
            return 0
        if data.get("cancel_requested") == "1":
            return -1
        data["lease_deadline"] = deadline
        data["updated_at"] = updated_at
        self._zadd(leased_key, float(deadline), job_id)
        return 1

    def _complete(self, keys: list[str], args: list[str]) -> int:
        leased_key, job_key, succeeded_key, cancelled_key = keys
        token, job_id, completed_at, result, _retention = args
        data = self.hashes[job_key]
        if data.get("state") != "running" or data.get("lease_token") != token:
            return 0
        self._zrem(leased_key, job_id)
        if data.get("cancel_requested") == "1":
            data.update(
                state="cancelled",
                completed_at=completed_at,
                last_error="cancelled",
                progress_stage="cancelled",
                updated_at=completed_at,
            )
            self._zadd(cancelled_key, float(completed_at), job_id)
        else:
            data.update(
                state="succeeded",
                completed_at=completed_at,
                result=result,
                last_error="",
                progress_percent="100",
                progress_stage="succeeded",
                updated_at=completed_at,
            )
            self._zadd(succeeded_key, float(completed_at), job_id)
        for field in ("lease_token", "lease_deadline", "worker_id"):
            data.pop(field, None)
        return 1

    def _fail(self, keys: list[str], args: list[str]) -> int:
        ready_key, leased_key, job_key, dead_key, cancelled_key = keys
        token, job_id, now, error, available_at, retryable, _retention = args
        data = self.hashes[job_key]
        if data.get("state") != "running" or data.get("lease_token") != token:
            return 0
        self._zrem(leased_key, job_id)
        if data.get("cancel_requested") == "1":
            data.update(
                state="cancelled",
                completed_at=now,
                last_error="cancelled",
                progress_stage="cancelled",
                updated_at=now,
            )
            self._zadd(cancelled_key, float(now), job_id)
            return 2
        if retryable == "1" and int(data["attempts"]) < int(data["max_attempts"]):
            data.update(
                state="queued",
                available_at=available_at,
                last_error=error,
                progress_stage="retrying",
                updated_at=now,
            )
            for field in ("lease_token", "lease_deadline", "worker_id"):
                data.pop(field, None)
            self._zadd(ready_key, float(available_at), job_id)
            return 1
        data.update(
            state="dead",
            completed_at=now,
            last_error=error,
            progress_stage="dead",
            updated_at=now,
        )
        for field in ("lease_token", "lease_deadline", "worker_id"):
            data.pop(field, None)
        self._zadd(dead_key, float(now), job_id)
        return 3

    def _cancel(self, keys: list[str], args: list[str]) -> int:
        ready_key, _leased_key, job_key, cancelled_key = keys
        job_id, now, _retention = args
        data = self.hashes.get(job_key)
        if data is None:
            return -1
        if data["state"] == "queued":
            self._zrem(ready_key, job_id)
            data.update(
                state="cancelled",
                cancel_requested="1",
                completed_at=now,
                last_error="cancelled",
                progress_stage="cancelled",
                updated_at=now,
            )
            self._zadd(cancelled_key, float(now), job_id)
            return 1
        if data["state"] == "running":
            data.update(
                cancel_requested="1",
                progress_stage="cancelling",
                updated_at=now,
            )
            return 2
        return 0

    def _retry(self, keys: list[str], args: list[str]) -> int:
        ready_key, dead_key, cancelled_key, job_key = keys
        job_id, now = args
        data = self.hashes.get(job_key)
        if data is None:
            return -1
        if data["state"] not in {"dead", "cancelled"}:
            return 0
        self._zrem(dead_key, job_id)
        self._zrem(cancelled_key, job_id)
        data.update(
            state="queued",
            available_at=now,
            attempts="0",
            cancel_requested="0",
            last_error="",
            progress_percent="0",
            progress_stage="queued",
            progress_detail="",
            updated_at=now,
        )
        data.pop("completed_at", None)
        data.pop("result", None)
        self._zadd(ready_key, float(now), job_id)
        return 1


class AdminManagementTests(unittest.IsolatedAsyncioTestCase):
    async def test_confirmed_mutations_are_atomic_audited_and_one_time(self) -> None:
        redis = FakeRedis()
        manager = RedisAdminManager(redis, redis_prefix="tests")
        redis.sets[manager.admins_key] = {"1"}

        add_token, _ttl = await manager.create_confirmation(
            action="add",
            actor_id=1,
            target_id=2,
        )
        added = await manager.add(
            actor_id=1,
            target_id=2,
            confirmation_token=add_token,
        )
        self.assertTrue(added.changed)
        self.assertEqual((1, 2), await manager.list_ids())
        with self.assertRaises(AdminConfirmationError):
            await manager.add(
                actor_id=1,
                target_id=2,
                confirmation_token=add_token,
            )

        remove_token, _ttl = await manager.create_confirmation(
            action="remove",
            actor_id=2,
            target_id=1,
        )
        removed = await manager.remove(
            actor_id=2,
            target_id=1,
            confirmation_token=remove_token,
        )
        self.assertTrue(removed.changed)
        self.assertEqual((2,), await manager.list_ids())
        audit = await manager.audit()
        self.assertEqual(["admin_remove", "admin_add"], [row["action"] for row in audit])

    async def test_final_administrator_cannot_be_removed(self) -> None:
        redis = FakeRedis()
        manager = RedisAdminManager(redis, redis_prefix="tests")
        redis.sets[manager.admins_key] = {"1"}
        token, _ttl = await manager.create_confirmation(
            action="remove",
            actor_id=1,
            target_id=1,
        )

        with self.assertRaises(LastAdministratorError):
            await manager.remove(
                actor_id=1,
                target_id=1,
                confirmation_token=token,
            )
        self.assertEqual((1,), await manager.list_ids())


class ProviderManagerTests(unittest.IsolatedAsyncioTestCase):
    async def test_failure_opens_circuit_and_falls_back(self) -> None:
        manager = ProviderManager()
        manager.register(
            "primary",
            capabilities={"ai"},
            priority=1,
            failure_threshold=1,
            cooldown_seconds=60,
            timeout_seconds=1,
        )
        manager.register(
            "fallback",
            capabilities={"ai"},
            priority=2,
            timeout_seconds=1,
        )
        calls: list[str] = []

        async def operation(provider: str) -> str:
            calls.append(provider)
            if provider == "primary":
                raise RuntimeError("provider unavailable")
            return "ok"

        result, provider = await manager.execute("ai", operation)

        self.assertEqual(("ok", "fallback"), (result, provider))
        self.assertEqual(["primary", "fallback"], calls)
        self.assertFalse(manager.snapshot()["primary"]["available"])
        calls.clear()
        await manager.execute("ai", operation)
        self.assertEqual(["fallback"], calls)

    async def test_timeout_is_tracked_and_falls_back(self) -> None:
        manager = ProviderManager()
        manager.register(
            "slow",
            capabilities={"tts"},
            priority=1,
            failure_threshold=1,
            cooldown_seconds=60,
            timeout_seconds=0.1,
        )
        manager.register(
            "fast",
            capabilities={"tts"},
            priority=2,
            timeout_seconds=1,
        )

        async def operation(provider: str) -> str:
            if provider == "slow":
                await asyncio.sleep(1)
            return provider

        result, provider = await manager.execute("tts", operation)

        self.assertEqual(("fast", "fast"), (result, provider))
        self.assertEqual(1, manager.snapshot()["slow"]["failures"])

    async def test_exhaustion_reports_each_provider(self) -> None:
        manager = ProviderManager()
        manager.register("one", capabilities={"ocr"})
        manager.register("two", capabilities={"ocr"})

        with self.assertRaises(NoProviderAvailable) as raised:
            await manager.execute(
                "ocr",
                lambda provider: (_ for _ in ()).throw(RuntimeError(provider)),
            )

        self.assertEqual({"one", "two"}, set(raised.exception.errors))


class RedisJobQueueTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.redis = FakeRedis()
        self.queue = RedisJobQueue(
            self.redis,
            redis_prefix="tests",
            lease_seconds=5,
        )

    async def test_idempotent_enqueue_and_multi_worker_claims(self) -> None:
        first, created = await self.queue.enqueue(
            "tts",
            {"text": "hello"},
            priority=1,
            idempotency_key="user:1:message:2",
        )
        duplicate, duplicate_created = await self.queue.enqueue(
            "tts",
            {"text": "different"},
            priority=1,
            idempotency_key="user:1:message:2",
        )
        high, _created = await self.queue.enqueue(
            "ocr",
            {"file_id": "abc"},
            priority=10,
        )

        self.assertTrue(created)
        self.assertFalse(duplicate_created)
        self.assertEqual(first.id, duplicate.id)
        claimed_high = await self.queue.claim("worker-a")
        claimed_first = await self.queue.claim("worker-b")
        self.assertEqual(high.id, claimed_high.id)
        self.assertEqual(first.id, claimed_first.id)
        self.assertIsNone(await self.queue.claim("worker-c"))

        self.assertTrue(
            await self.queue.complete(
                claimed_high.id,
                claimed_high.lease_token,
                {"text": "result"},
            )
        )
        self.assertEqual("succeeded", (await self.queue.get(high.id)).state)

    async def test_retry_dead_letter_cancel_and_manual_retry(self) -> None:
        job, _created = await self.queue.enqueue(
            "transcription",
            {"file_id": "voice"},
            max_attempts=2,
        )
        first = await self.queue.claim("worker-a")
        self.assertEqual(
            "queued",
            await self.queue.fail(
                first.id,
                first.lease_token,
                "temporary",
                retry_delay_seconds=0,
            ),
        )
        second = await self.queue.claim("worker-b")
        self.assertEqual(
            "dead",
            await self.queue.fail(
                second.id,
                second.lease_token,
                "still broken",
            ),
        )
        self.assertEqual("dead", (await self.queue.get(job.id)).state)
        self.assertEqual(1, (await self.queue.stats())["dead"])

        self.assertTrue(await self.queue.retry(job.id))
        self.assertEqual("queued", (await self.queue.get(job.id)).state)
        running = await self.queue.claim("worker-c")
        self.assertEqual("requested", await self.queue.cancel(running.id))
        self.assertEqual(-1, await self.queue.renew(running.id, running.lease_token))
        self.assertEqual(
            "cancelled",
            await self.queue.fail(
                running.id,
                running.lease_token,
                "cancelled",
            ),
        )

    async def test_worker_retries_failed_handler(self) -> None:
        job, _created = await self.queue.enqueue(
            "broadcast",
            {"broadcast_id": 7},
            max_attempts=2,
        )

        async def fail_handler(_payload, _context):
            raise RuntimeError("send failed")

        worker = RedisJobWorker(
            self.queue,
            {"broadcast": fail_handler},
            retry_base_seconds=0.1,
            retry_max_seconds=0.1,
        )
        self.assertTrue(await worker.process_one())
        updated = await self.queue.get(job.id)
        self.assertEqual("queued", updated.state)
        self.assertEqual(1, updated.attempts)
        self.assertIn("send failed", updated.last_error)


if __name__ == "__main__":
    unittest.main()
