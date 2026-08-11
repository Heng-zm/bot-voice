"""Durable artifact storage for job outputs.

Supabase Storage is the production backend. A local atomic backend remains
available for one-process development and tests, but is intentionally reported
as non-shared so deployments do not mistake it for cross-instance storage.
"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import logging
import os
import re
import tempfile
import time
import uuid
from collections.abc import Mapping
from contextlib import suppress
from pathlib import Path
from typing import Any

from app.services.artifacts.models import ArtifactRef

logger = logging.getLogger(__name__)
_SAFE_PATH = re.compile(r"[^A-Za-z0-9._/-]+")

# Built-in production-safe defaults. Environment variables are optional and
# only override these values when explicitly configured.
DEFAULT_BOT_ARTIFACT_STORAGE_MODE = "auto"
DEFAULT_BOT_ARTIFACT_STORAGE_BUCKET = "bot-job-artifacts"
DEFAULT_BOT_ARTIFACT_LOCAL_DIRECTORY = "data/job-artifacts"
DEFAULT_BOT_ARTIFACT_MAX_BYTES = 52_428_800


class ArtifactStorageError(RuntimeError):
    """Base artifact storage error."""


class ArtifactNotFound(ArtifactStorageError):
    """Raised when a referenced artifact no longer exists."""


class ArtifactIntegrityError(ArtifactStorageError):
    """Raised when a deterministic path contains different artifact bytes."""


def _is_not_found_error(error: BaseException) -> bool:
    """Conservatively identify a Storage API not-found response.

    Supabase client versions expose HTTP errors with slightly different
    attributes. Treat only an explicit 404/code or a clear not-found message as
    absence; transport and authentication failures must remain retryable.
    """

    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        for attribute in ("status_code", "status", "code"):
            raw = getattr(current, attribute, None)
            if str(raw or "").strip().lower() in {
                "404",
                "not_found",
                "notfound",
            }:
                return True
        message = str(current).strip().lower()
        if "not found" in message or "not_found" in message:
            return True
        current = current.__cause__ or current.__context__
    return False


def _clean_path(value: str) -> str:
    cleaned = _SAFE_PATH.sub("-", str(value or "").strip().replace("\\", "/"))
    cleaned = "/".join(part for part in cleaned.split("/") if part not in {"", ".", ".."})
    if not cleaned or len(cleaned) > 512:
        raise ValueError("Artifact path is missing or too long.")
    return cleaned


def _content_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _artifact_id(path: str, digest: str) -> str:
    return hashlib.sha256(f"{path}:{digest}".encode()).hexdigest()[:32]


class LocalArtifactStore:
    """Atomic local storage for development and single-host deployments."""

    backend = "local"
    shared = False

    def __init__(self, root: str | os.PathLike[str], *, max_bytes: int) -> None:
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.max_bytes = max(1_024, int(max_bytes))

    def _target(self, path: str) -> Path:
        clean = _clean_path(path)
        target = (self.root / clean).resolve()
        if self.root not in target.parents:
            raise ValueError("Artifact path escapes the configured root.")
        return target

    async def put_bytes(
        self,
        path: str,
        data: bytes,
        *,
        content_type: str,
        ttl_seconds: int | None = None,
    ) -> ArtifactRef:
        payload = bytes(data)
        if len(payload) > self.max_bytes:
            raise ArtifactStorageError(
                f"Artifact exceeds the {self.max_bytes}-byte limit."
            )
        target = self._target(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        digest = _content_hash(payload)

        def write() -> None:
            fd, temporary = tempfile.mkstemp(
                prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent)
            )
            try:
                with os.fdopen(fd, "wb") as handle:
                    handle.write(payload)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary, target)
            except BaseException:
                with suppress(OSError):
                    os.close(fd)
                with suppress(OSError):
                    os.unlink(temporary)
                raise

        await asyncio.to_thread(write)
        created = time.time()
        return ArtifactRef(
            id=_artifact_id(_clean_path(path), digest),
            backend=self.backend,
            path=_clean_path(path),
            content_type=str(content_type or "application/octet-stream")[:128],
            size_bytes=len(payload),
            sha256=digest,
            created_at=created,
            expires_at=(created + int(ttl_seconds)) if ttl_seconds else None,
        )

    async def get_bytes(self, artifact: ArtifactRef) -> bytes:
        target = self._target(artifact.path)
        try:
            payload = await asyncio.to_thread(target.read_bytes)
        except FileNotFoundError as exc:
            raise ArtifactNotFound(artifact.path) from exc
        if len(payload) != artifact.size_bytes or _content_hash(payload) != artifact.sha256:
            raise ArtifactIntegrityError("Artifact integrity validation failed.")
        return payload

    async def delete(self, artifact: ArtifactRef) -> bool:
        target = self._target(artifact.path)

        def remove() -> bool:
            try:
                target.unlink()
            except FileNotFoundError:
                return False
            return True

        return await asyncio.to_thread(remove)


class SupabaseArtifactStore:
    """Supabase Storage backend safe for separate web and worker services."""

    backend = "supabase"
    shared = True

    def __init__(self, client: Any, *, bucket: str, max_bytes: int) -> None:
        if client is None:
            raise ArtifactStorageError("Supabase client is required.")
        self.client = client
        self.bucket = str(bucket or "bot-job-artifacts").strip()
        if not self.bucket or len(self.bucket) > 128:
            raise ValueError("Artifact bucket is missing or too long.")
        self.max_bytes = max(1_024, int(max_bytes))

    def _bucket(self) -> Any:
        storage = getattr(self.client, "storage", None)
        selector = getattr(storage, "from_", None)
        if not callable(selector):
            raise ArtifactStorageError("Supabase Storage API is unavailable.")
        return selector(self.bucket)

    async def put_bytes(
        self,
        path: str,
        data: bytes,
        *,
        content_type: str,
        ttl_seconds: int | None = None,
    ) -> ArtifactRef:
        payload = bytes(data)
        if len(payload) > self.max_bytes:
            raise ArtifactStorageError(
                f"Artifact exceeds the {self.max_bytes}-byte limit."
            )
        clean = _clean_path(path)
        digest = _content_hash(payload)

        def upload() -> Any:
            return self._bucket().upload(
                clean,
                payload,
                file_options={
                    "content-type": str(content_type or "application/octet-stream"),
                    "upsert": "true",
                },
            )

        try:
            result = await asyncio.to_thread(upload)
            if inspect.isawaitable(result):
                await result
        except Exception as exc:
            raise ArtifactStorageError("Supabase artifact upload failed.") from exc
        created = time.time()
        return ArtifactRef(
            id=_artifact_id(clean, digest),
            backend=self.backend,
            bucket=self.bucket,
            path=clean,
            content_type=str(content_type or "application/octet-stream")[:128],
            size_bytes=len(payload),
            sha256=digest,
            created_at=created,
            expires_at=(created + int(ttl_seconds)) if ttl_seconds else None,
        )

    async def get_bytes(self, artifact: ArtifactRef) -> bytes:
        try:
            payload = await asyncio.to_thread(self._bucket().download, artifact.path)
            if inspect.isawaitable(payload):
                payload = await payload
        except Exception as exc:
            if _is_not_found_error(exc):
                raise ArtifactNotFound(artifact.path) from exc
            raise ArtifactStorageError("Supabase artifact download failed.") from exc
        data = bytes(payload or b"")
        if len(data) != artifact.size_bytes or _content_hash(data) != artifact.sha256:
            raise ArtifactIntegrityError("Artifact integrity validation failed.")
        return data

    async def delete(self, artifact: ArtifactRef) -> bool:
        try:
            result = await asyncio.to_thread(self._bucket().remove, [artifact.path])
            if inspect.isawaitable(result):
                await result
        except Exception as exc:
            raise ArtifactStorageError("Supabase artifact deletion failed.") from exc
        return True


class ArtifactService:
    """Create deterministic artifact references for retry-safe jobs."""

    def __init__(
        self,
        store: LocalArtifactStore | SupabaseArtifactStore,
        *,
        redis_client: Any | None = None,
        redis_prefix: str = "tgbot",
    ) -> None:
        self.store = store
        self.redis = redis_client
        prefix = str(redis_prefix or "tgbot").strip().strip(":") or "tgbot"
        self.expiry_key = f"{prefix}:artifacts:expires:v1"
        self.last_cleanup_at = 0.0
        self.last_cleanup_deleted = 0
        self.last_cleanup_errors = 0

    @property
    def backend(self) -> str:
        return self.store.backend

    @property
    def shared(self) -> bool:
        return self.store.shared

    async def put_text(
        self,
        *,
        job_id: str,
        name: str,
        text: str,
        ttl_seconds: int | None = None,
    ) -> ArtifactRef:
        safe_name = _clean_path(name).rsplit("/", 1)[-1]
        path = f"results/{_clean_path(job_id)}/{safe_name}"
        artifact = await self.store.put_bytes(
            path,
            str(text).encode("utf-8"),
            content_type="text/plain; charset=utf-8",
            ttl_seconds=ttl_seconds,
        )
        await self._register_expiry(artifact)
        return artifact

    async def get_text(self, artifact: ArtifactRef) -> str:
        if artifact.expires_at is not None and artifact.expires_at <= time.time():
            try:
                # Verify that the path still contains this exact artifact. A
                # retry may have replaced the deterministic path with newer
                # content that must not be removed by an older reference.
                await self.store.get_bytes(artifact)
            except ArtifactStorageError:
                pass
            else:
                await self.store.delete(artifact)
            raise ArtifactNotFound(f"Artifact {artifact.path!r} has expired.")
        return (await self.store.get_bytes(artifact)).decode("utf-8", errors="strict")

    @staticmethod
    def _registry_member(artifact: ArtifactRef) -> str:
        value = artifact.as_dict()
        # Exclude timestamps so retrying the same deterministic artifact
        # updates its ZSET score instead of creating an older cleanup entry.
        value.pop("created_at", None)
        value.pop("expires_at", None)
        return json.dumps(
            value,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )

    async def _register_expiry(self, artifact: ArtifactRef) -> None:
        if self.redis is None or artifact.expires_at is None:
            return
        member = self._registry_member(artifact)
        try:
            await asyncio.to_thread(
                self.redis.zadd,
                self.expiry_key,
                {member: float(artifact.expires_at)},
            )
        except Exception as exc:
            raise ArtifactStorageError(
                "Could not register artifact expiration in Redis."
            ) from exc

    async def cleanup_expired(
        self,
        *,
        now: float | None = None,
        limit: int = 500,
    ) -> dict[str, Any]:
        """Delete one bounded batch of expired artifacts from the active store."""

        cleanup_time = time.time() if now is None else float(now)
        batch_size = max(1, min(5_000, int(limit)))
        if self.redis is None:
            return {
                "configured": False,
                "checked": 0,
                "deleted": 0,
                "errors": 0,
                "remaining_due": 0,
            }
        try:
            raw_members = await asyncio.to_thread(
                self.redis.zrangebyscore,
                self.expiry_key,
                "-inf",
                cleanup_time,
                start=0,
                num=batch_size,
            )
        except Exception as exc:
            raise ArtifactStorageError(
                "Could not load expired artifact references from Redis."
            ) from exc

        members = list(raw_members or ())
        removable: list[Any] = []
        deleted = errors = 0
        semaphore = asyncio.Semaphore(8)

        async def remove_one(raw_member: Any) -> tuple[Any, bool, bool, bool]:
            text = (
                raw_member.decode("utf-8", errors="strict")
                if isinstance(raw_member, bytes)
                else str(raw_member)
            )
            try:
                decoded = json.loads(text)
                if not isinstance(decoded, Mapping):
                    raise ValueError("Artifact registry entry must be an object.")
                artifact = ArtifactRef.from_dict(dict(decoded))
                clean_path = _clean_path(artifact.path)
                if (
                    clean_path != artifact.path
                    or not artifact.id
                    or artifact.size_bytes < 0
                    or not re.fullmatch(r"[0-9a-f]{64}", artifact.sha256)
                ):
                    raise ValueError("Artifact registry entry is invalid.")
            except (AttributeError, TypeError, ValueError, json.JSONDecodeError):
                return raw_member, False, True, False
            if artifact.backend != self.backend:
                logger.warning(
                    "Cannot clean expired artifact for inactive backend=%s path=%s",
                    artifact.backend,
                    artifact.path,
                )
                return raw_member, True, False, False
            async with semaphore:
                try:
                    await self.store.get_bytes(artifact)
                except ArtifactNotFound:
                    return raw_member, False, True, False
                except ArtifactIntegrityError:
                    # The deterministic path now contains different content;
                    # discard only this stale registry entry.
                    return raw_member, False, True, False
                except ArtifactStorageError:
                    logger.warning(
                        "Expired artifact validation failed path=%s",
                        artifact.path,
                        exc_info=True,
                    )
                    return raw_member, True, False, False
                except Exception:  # noqa: BLE001 - cleanup batch boundary
                    logger.warning(
                        "Unexpected expired artifact validation failure path=%s",
                        artifact.path,
                        exc_info=True,
                    )
                    return raw_member, True, False, False
                try:
                    await self.store.delete(artifact)
                except ArtifactStorageError:
                    logger.warning(
                        "Expired artifact deletion failed path=%s",
                        artifact.path,
                        exc_info=True,
                    )
                    return raw_member, True, False, False
                except Exception:  # noqa: BLE001 - cleanup batch boundary
                    logger.warning(
                        "Unexpected expired artifact deletion failure path=%s",
                        artifact.path,
                        exc_info=True,
                    )
                    return raw_member, True, False, False
            return raw_member, False, True, True

        results = await asyncio.gather(*(remove_one(member) for member in members))
        for raw_member, failed, remove_registry_entry, artifact_deleted in results:
            if failed:
                errors += 1
            elif remove_registry_entry:
                removable.append(raw_member)
                deleted += int(artifact_deleted)

        if removable:
            try:
                await asyncio.to_thread(
                    self.redis.zrem,
                    self.expiry_key,
                    *removable,
                )
            except Exception as exc:
                raise ArtifactStorageError(
                    "Artifacts were deleted but their expiration entries remain."
                ) from exc

        self.last_cleanup_at = cleanup_time
        self.last_cleanup_deleted = deleted
        self.last_cleanup_errors = errors
        return {
            "configured": True,
            "checked": len(members),
            "deleted": deleted,
            "errors": errors,
            "remaining_due": max(0, len(members) - len(removable)),
        }

    def snapshot(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "shared": self.shared,
            "cleanup_configured": self.redis is not None,
            "last_cleanup_at": self.last_cleanup_at or None,
            "last_cleanup_deleted": self.last_cleanup_deleted,
            "last_cleanup_errors": self.last_cleanup_errors,
        }


_ARTIFACT_SERVICE: ArtifactService | None = None


def configure_artifact_service(
    *,
    supabase_client: Any | None,
    redis_client: Any | None = None,
    redis_prefix: str = "tgbot",
    role: str,
) -> ArtifactService:
    global _ARTIFACT_SERVICE
    mode = str(
        os.getenv(
            "BOT_ARTIFACT_STORAGE_MODE",
            DEFAULT_BOT_ARTIFACT_STORAGE_MODE,
        )
        or DEFAULT_BOT_ARTIFACT_STORAGE_MODE
    ).strip().lower()
    bucket = str(
        os.getenv(
            "BOT_ARTIFACT_STORAGE_BUCKET",
            DEFAULT_BOT_ARTIFACT_STORAGE_BUCKET,
        )
        or DEFAULT_BOT_ARTIFACT_STORAGE_BUCKET
    ).strip()
    raw_max_bytes = os.getenv(
        "BOT_ARTIFACT_MAX_BYTES",
        str(DEFAULT_BOT_ARTIFACT_MAX_BYTES),
    )
    try:
        max_bytes = int(str(raw_max_bytes).strip())
    except (TypeError, ValueError):
        logger.warning(
            "Invalid BOT_ARTIFACT_MAX_BYTES=%r; using source default %s.",
            raw_max_bytes,
            DEFAULT_BOT_ARTIFACT_MAX_BYTES,
        )
        max_bytes = DEFAULT_BOT_ARTIFACT_MAX_BYTES

    if mode not in {"auto", "supabase", "local"}:
        raise ArtifactStorageError("BOT_ARTIFACT_STORAGE_MODE must be auto, supabase, or local.")
    if mode in {"auto", "supabase"} and supabase_client is not None:
        store: LocalArtifactStore | SupabaseArtifactStore = SupabaseArtifactStore(
            supabase_client,
            bucket=bucket,
            max_bytes=max_bytes,
        )
    elif mode == "supabase":
        raise ArtifactStorageError("Supabase artifact storage was requested but is unavailable.")
    else:
        root = (
            os.getenv(
                "BOT_ARTIFACT_LOCAL_DIRECTORY",
                DEFAULT_BOT_ARTIFACT_LOCAL_DIRECTORY,
            )
            or DEFAULT_BOT_ARTIFACT_LOCAL_DIRECTORY
        )
        store = LocalArtifactStore(root, max_bytes=max_bytes)
        if str(role).lower() == "worker":
            logger.warning(
                "Worker is using local artifact storage. Use Supabase for multi-instance durability."
            )
    _ARTIFACT_SERVICE = ArtifactService(
        store,
        redis_client=redis_client,
        redis_prefix=redis_prefix,
    )
    return _ARTIFACT_SERVICE


def get_artifact_service() -> ArtifactService:
    if _ARTIFACT_SERVICE is None:
        raise ArtifactStorageError("Artifact storage is not configured.")
    return _ARTIFACT_SERVICE


def reset_artifact_service() -> None:
    global _ARTIFACT_SERVICE
    _ARTIFACT_SERVICE = None


def deterministic_artifact_name(kind: str) -> str:
    clean = _clean_path(kind).replace("/", "-")
    return f"{clean}-{uuid.uuid4().hex[:8]}.txt"


__all__ = [
    "ArtifactIntegrityError",
    "ArtifactNotFound",
    "ArtifactService",
    "ArtifactStorageError",
    "DEFAULT_BOT_ARTIFACT_LOCAL_DIRECTORY",
    "DEFAULT_BOT_ARTIFACT_MAX_BYTES",
    "DEFAULT_BOT_ARTIFACT_STORAGE_BUCKET",
    "DEFAULT_BOT_ARTIFACT_STORAGE_MODE",
    "LocalArtifactStore",
    "SupabaseArtifactStore",
    "configure_artifact_service",
    "get_artifact_service",
    "reset_artifact_service",
]
