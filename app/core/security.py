"""Runtime secret management backed by Redis.

The three application secrets in this module are operational state, not
deployment configuration.  They are generated once with ``secrets`` and stored
without a Redis expiry.  ``SET ... NX`` is the final race-safety guarantee; the
short distributed lock only keeps first-boot logs and Redis traffic tidy when
several application instances start together.
"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import logging
import re
import secrets
import threading
import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

SECRET_NAMES = (
    "TELEGRAM_WEBHOOK_SECRET_TOKEN",
    "WEB_SECRET_KEY",
    "FLASK_SECRET_KEY",
)
_URLSAFE_TOKEN_RE = re.compile(r"^[A-Za-z0-9_-]{64}$")
_UNLOCK_SCRIPT = """
if redis.call('GET', KEYS[1]) == ARGV[1] then
  return redis.call('DEL', KEYS[1])
end
return 0
""".strip()


class RuntimeSecretError(RuntimeError):
    """Raised when persistent runtime secrets cannot be loaded safely."""


@dataclass(frozen=True)
class SecretRecord:
    name: str
    redis_key: str
    created: bool
    source: str
    value: str = field(repr=False)

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(self.value.encode("utf-8")).hexdigest()[:12]


@dataclass(frozen=True)
class RuntimeSecrets:
    records: Mapping[str, SecretRecord]
    webhook_registration_required: bool

    def value(self, name: str) -> str:
        try:
            return self.records[name].value
        except KeyError as exc:
            raise RuntimeSecretError(f"Unknown runtime secret: {name}") from exc

    @property
    def newly_created(self) -> frozenset[str]:
        return frozenset(name for name, record in self.records.items() if record.created)

    def redacted_status(self) -> dict[str, dict[str, Any]]:
        return {
            name: {
                "redis_key": record.redis_key,
                "created": record.created,
                "source": record.source,
                "fingerprint": record.fingerprint,
            }
            for name, record in self.records.items()
        }


class RuntimeSecretManager:
    """Load or atomically create application secrets in one Redis instance."""

    def __init__(self, *, redis_prefix: str = "tgbot") -> None:
        prefix = str(redis_prefix or "tgbot").strip().strip(":")
        self.redis_prefix = prefix or "tgbot"
        self._redis: Any | None = None
        self._redis_url = ""
        self._owns_redis = False
        self._records: dict[str, SecretRecord] = {}
        self._thread_lock = threading.RLock()

    def configure(
        self,
        *,
        redis_client: Any | None = None,
        redis_url: str = "",
        disable_redis: bool = False,
    ) -> RuntimeSecretManager:
        with self._thread_lock:
            if disable_redis:
                if self._owns_redis and self._redis is not None:
                    try:
                        self._redis.close()
                    except Exception:
                        logger.debug(
                            "Disabled runtime-secret Redis client close failed.",
                            exc_info=True,
                        )
                self._redis = None
                self._redis_url = ""
                self._owns_redis = False
                return self
            if redis_client is not None:
                if (
                    self._owns_redis
                    and self._redis is not None
                    and self._redis is not redis_client
                ):
                    try:
                        self._redis.close()
                    except Exception:
                        logger.debug(
                            "Superseded runtime-secret Redis client close failed.",
                            exc_info=True,
                        )
                self._redis = redis_client
                self._owns_redis = False
            if redis_url:
                self._redis_url = str(redis_url).strip()
        return self

    @property
    def redis_client(self) -> Any | None:
        return self._redis

    def _secret_key(self, name: str) -> str:
        # Keep the established web-session key so existing signed admin
        # sessions remain valid after this refactor.
        if name == "WEB_SECRET_KEY":
            return f"{self.redis_prefix}:web_secret_key:v1"
        if name == "FLASK_SECRET_KEY":
            return f"{self.redis_prefix}:flask_secret_key:v1"
        if name == "TELEGRAM_WEBHOOK_SECRET_TOKEN":
            return f"{self.redis_prefix}:telegram_webhook_secret_token:v1"
        raise RuntimeSecretError(f"Unknown runtime secret: {name}")

    def _bootstrap_lock_key(self) -> str:
        return f"{self.redis_prefix}:security:runtime_secrets:bootstrap_lock:v1"

    def _webhook_marker_key(self) -> str:
        return f"{self.redis_prefix}:telegram_webhook_secret_token:registered:v1"

    def _webhook_lock_key(self) -> str:
        return f"{self.redis_prefix}:telegram_webhook_secret_token:register_lock:v1"

    @staticmethod
    def _decode(value: Any) -> str:
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="strict")
        return str(value or "")

    @staticmethod
    def _validate_secret(name: str, value: str) -> str:
        value = str(value or "").strip()
        if not _URLSAFE_TOKEN_RE.fullmatch(value):
            raise RuntimeSecretError(
                f"Redis contains an invalid {name}; expected exactly 64 URL-safe characters."
            )
        return value

    @staticmethod
    def _new_secret() -> str:
        value = secrets.token_urlsafe(48)
        if not _URLSAFE_TOKEN_RE.fullmatch(value):  # defensive invariant
            raise RuntimeSecretError("secrets.token_urlsafe(48) returned an unexpected value.")
        return value

    def _connect(self) -> Any | None:
        if self._redis is not None:
            return self._redis
        if not self._redis_url:
            return None
        try:
            import redis

            client = redis.from_url(
                self._redis_url,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5,
                health_check_interval=30,
                retry_on_timeout=True,
                max_connections=8,
            )
            client.ping()
        except Exception as exc:
            raise RuntimeSecretError(f"Redis is unavailable for runtime secrets: {exc}") from exc
        self._redis = client
        self._owns_redis = True
        return client

    @staticmethod
    def _release_lock(client: Any, lock_key: str, lock_token: str) -> None:
        try:
            client.eval(_UNLOCK_SCRIPT, 1, lock_key, lock_token)
        except Exception:  # noqa: BLE001 - lock cleanup must be best-effort
            logger.warning("Could not release Redis runtime-secret lock key=%s.", lock_key)

    def _get_or_create_one(self, client: Any, name: str) -> SecretRecord:
        redis_key = self._secret_key(name)
        existing = self._decode(client.get(redis_key)).strip()
        if existing:
            value = self._validate_secret(name, existing)
            return SecretRecord(name, redis_key, False, "redis", value)

        generated = self._new_secret()
        created = bool(client.set(redis_key, generated, nx=True))
        if created:
            # No EX/PX option: these values intentionally persist permanently.
            value = generated
        else:
            value = self._decode(client.get(redis_key)).strip()
            if not value:
                raise RuntimeSecretError(
                    f"Redis race while creating {name}: no winning value was readable."
                )
            value = self._validate_secret(name, value)
        return SecretRecord(
            name,
            redis_key,
            created,
            "redis-generated" if created else "redis-race-winner",
            value,
        )

    def _registration_required(self, client: Any, webhook_secret: str) -> bool:
        expected = hashlib.sha256(webhook_secret.encode("utf-8")).hexdigest()
        marker = self._decode(client.get(self._webhook_marker_key())).strip()
        return not secrets.compare_digest(marker, expected)

    def bootstrap_sync(self, *, strict: bool = True) -> RuntimeSecrets:
        with self._thread_lock:
            try:
                client = self._connect()
            except RuntimeSecretError:
                if strict:
                    raise
                client = None

            if client is None:
                if strict:
                    raise RuntimeSecretError(
                        "REDIS_URL is required because runtime secrets must be persistent."
                    )
                if not self._records or any(
                    record.source != "memory-ephemeral" for record in self._records.values()
                ):
                    self._records = {
                        name: SecretRecord(
                            name,
                            self._secret_key(name),
                            True,
                            "memory-ephemeral",
                            self._new_secret(),
                        )
                        for name in SECRET_NAMES
                    }
                return RuntimeSecrets(dict(self._records), False)

            lock_key = self._bootstrap_lock_key()
            lock_token = secrets.token_urlsafe(24)
            lock_acquired = bool(client.set(lock_key, lock_token, nx=True, px=15_000))
            try:
                records = {
                    name: self._get_or_create_one(client, name)
                    for name in SECRET_NAMES
                }
            finally:
                if lock_acquired:
                    self._release_lock(client, lock_key, lock_token)

            self._records = records
            webhook_secret = records["TELEGRAM_WEBHOOK_SECRET_TOKEN"].value
            registration_required = self._registration_required(client, webhook_secret)
            for record in records.values():
                if record.created:
                    logger.warning(
                        "Generated persistent runtime secret name=%s redis_key=%s fingerprint=%s.",
                        record.name,
                        record.redis_key,
                        record.fingerprint,
                    )
            return RuntimeSecrets(dict(records), registration_required)

    async def bootstrap(self, *, strict: bool = True) -> RuntimeSecrets:
        return await asyncio.to_thread(self.bootstrap_sync, strict=strict)

    def current(self) -> RuntimeSecrets | None:
        with self._thread_lock:
            if not self._records:
                return None
            client = self._redis
            registration_required = False
            if client is not None:
                registration_required = self._registration_required(
                    client,
                    self._records["TELEGRAM_WEBHOOK_SECRET_TOKEN"].value,
                )
            return RuntimeSecrets(dict(self._records), registration_required)

    async def ensure_webhook_registered(
        self,
        register: Callable[[str], Awaitable[None] | None],
        *,
        wait_seconds: float = 20.0,
    ) -> bool:
        """Register the current token once across concurrent server startups.

        Returns ``True`` only for the process that called Telegram.  A
        fingerprint marker is written after a successful ``setWebhook`` call,
        so a crash before registration is retried by the next startup.
        """

        secrets_state = await self.bootstrap(strict=True)
        if not secrets_state.webhook_registration_required:
            return False
        client = self._redis
        if client is None:  # guarded by strict bootstrap
            raise RuntimeSecretError("Redis disappeared before webhook registration.")

        lock_key = self._webhook_lock_key()
        lock_token = secrets.token_urlsafe(24)
        acquired = await asyncio.to_thread(
            lambda: bool(client.set(lock_key, lock_token, nx=True, px=60_000))
        )
        webhook_secret = secrets_state.value("TELEGRAM_WEBHOOK_SECRET_TOKEN")
        expected_marker = hashlib.sha256(webhook_secret.encode("utf-8")).hexdigest()

        if not acquired:
            deadline = time.monotonic() + max(1.0, float(wait_seconds))
            while time.monotonic() < deadline:
                marker = await asyncio.to_thread(
                    lambda: self._decode(client.get(self._webhook_marker_key())).strip()
                )
                if secrets.compare_digest(marker, expected_marker):
                    return False
                await asyncio.sleep(0.2)
            raise RuntimeSecretError(
                "Timed out waiting for another server to register the Telegram webhook secret."
            )

        try:
            result = register(webhook_secret)
            if inspect.isawaitable(result):
                await result
            await asyncio.to_thread(
                lambda: client.set(self._webhook_marker_key(), expected_marker)
            )
            logger.info(
                "Telegram webhook secret registration completed fingerprint=%s.",
                hashlib.sha256(webhook_secret.encode("utf-8")).hexdigest()[:12],
            )
            return True
        finally:
            await asyncio.to_thread(self._release_lock, client, lock_key, lock_token)

    def persist_registered_webhook_secret_sync(self, secret_value: str) -> SecretRecord:
        """Persist a successfully registered manual rotation.

        The Telegram API must be updated before this method is called.  Writing
        the secret and matching registration marker together prevents startup
        from restoring an obsolete compatibility RUN_STATE value.
        """

        value = self._validate_secret(
            "TELEGRAM_WEBHOOK_SECRET_TOKEN",
            secret_value,
        )
        client = self._connect()
        if client is None:
            raise RuntimeSecretError(
                "Redis is required to persist a Telegram webhook secret rotation."
            )
        redis_key = self._secret_key("TELEGRAM_WEBHOOK_SECRET_TOKEN")
        marker = hashlib.sha256(value.encode("utf-8")).hexdigest()
        pipe = client.pipeline(transaction=True)
        pipe.set(redis_key, value)
        pipe.set(self._webhook_marker_key(), marker)
        pipe.execute()
        record = SecretRecord(
            "TELEGRAM_WEBHOOK_SECRET_TOKEN",
            redis_key,
            False,
            "redis-rotated",
            value,
        )
        with self._thread_lock:
            self._records["TELEGRAM_WEBHOOK_SECRET_TOKEN"] = record
        return record

    def close(self) -> None:
        with self._thread_lock:
            if self._owns_redis and self._redis is not None:
                try:
                    self._redis.close()
                except Exception:
                    logger.debug("Runtime-secret Redis client close failed.", exc_info=True)
            self._redis = None
            self._owns_redis = False


_RUNTIME_SECRET_MANAGER = RuntimeSecretManager()


def configure_runtime_secret_manager(
    *,
    redis_client: Any | None = None,
    redis_url: str = "",
    redis_prefix: str | None = None,
    disable_redis: bool = False,
) -> RuntimeSecretManager:
    global _RUNTIME_SECRET_MANAGER
    if redis_prefix:
        requested = str(redis_prefix).strip().strip(":")
        if requested and requested != _RUNTIME_SECRET_MANAGER.redis_prefix:
            if _RUNTIME_SECRET_MANAGER.current() is not None:
                raise RuntimeSecretError(
                    "Cannot change the runtime-secret Redis prefix after secrets were loaded."
                )
            _RUNTIME_SECRET_MANAGER = RuntimeSecretManager(redis_prefix=requested)
    return _RUNTIME_SECRET_MANAGER.configure(
        redis_client=redis_client,
        redis_url=redis_url,
        disable_redis=disable_redis,
    )


def get_runtime_secret_manager() -> RuntimeSecretManager:
    return _RUNTIME_SECRET_MANAGER


def bootstrap_runtime_secrets_sync(
    *,
    redis_client: Any | None = None,
    redis_url: str = "",
    redis_prefix: str = "tgbot",
    strict: bool = True,
    disable_redis: bool = False,
) -> RuntimeSecrets:
    manager = configure_runtime_secret_manager(
        redis_client=redis_client,
        redis_url=redis_url,
        redis_prefix=redis_prefix,
        disable_redis=disable_redis,
    )
    return manager.bootstrap_sync(strict=strict)


async def bootstrap_runtime_secrets(
    *,
    redis_client: Any | None = None,
    redis_url: str = "",
    redis_prefix: str = "tgbot",
    strict: bool = True,
    disable_redis: bool = False,
) -> RuntimeSecrets:
    manager = configure_runtime_secret_manager(
        redis_client=redis_client,
        redis_url=redis_url,
        redis_prefix=redis_prefix,
        disable_redis=disable_redis,
    )
    return await manager.bootstrap(strict=strict)


__all__ = [
    "SECRET_NAMES",
    "RuntimeSecretError",
    "RuntimeSecretManager",
    "RuntimeSecrets",
    "SecretRecord",
    "bootstrap_runtime_secrets",
    "bootstrap_runtime_secrets_sync",
    "configure_runtime_secret_manager",
    "get_runtime_secret_manager",
]
