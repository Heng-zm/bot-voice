"""Upstash Vector store client for semantic search, memory, and knowledge retrieval."""

from __future__ import annotations

import logging
import os
from typing import Any

try:
    import httpx
except (ImportError, ModuleNotFoundError):
    httpx = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


class UpstashVectorStore:
    """Async client for Upstash Vector REST API."""

    def __init__(
        self,
        url: str | None = None,
        token: str | None = None,
        *,
        timeout_s: float = 10.0,
    ) -> None:
        self.url = (url or os.environ.get("UPSTASH_VECTOR_REST_URL") or "").strip().rstrip("/")
        self.token = (token or os.environ.get("UPSTASH_VECTOR_REST_TOKEN") or "").strip().strip('"').strip("'")
        self.timeout = float(timeout_s)

    @property
    def is_configured(self) -> bool:
        return bool(self.url and self.token)

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
        }

    async def upsert(
        self,
        id: str,
        *,
        data: str | None = None,
        vector: list[float] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """Upsert a single vector or raw text data (with Upstash auto-embedding)."""
        if not self.is_configured or httpx is None:
            return False

        payload: dict[str, Any] = {"id": str(id)}
        if data is not None:
            payload["data"] = str(data)
        if vector is not None:
            payload["vector"] = list(vector)
        if metadata:
            payload["metadata"] = dict(metadata)

        endpoint = f"{self.url}/upsert"
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                res = await client.post(endpoint, json=payload, headers=self._headers())
                if res.status_code < 400:
                    return True
                logger.warning("Upstash Vector upsert failed status=%s: %s", res.status_code, res.text[:200])
        except Exception as exc:
            logger.warning("Upstash Vector upsert exception: %s", exc)
        return False

    async def upsert_many(self, documents: list[dict[str, Any]]) -> bool:
        """Upsert multiple documents/vectors in a batch."""
        if not self.is_configured or not documents or httpx is None:
            return False

        endpoint = f"{self.url}/upsert-data" if any("data" in doc for doc in documents) else f"{self.url}/upsert"
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                res = await client.post(endpoint, json=documents, headers=self._headers())
                if res.status_code < 400:
                    return True
                logger.warning("Upstash Vector batch upsert failed status=%s: %s", res.status_code, res.text[:200])
        except Exception as exc:
            logger.warning("Upstash Vector batch upsert exception: %s", exc)
        return False

    async def query(
        self,
        *,
        data: str | None = None,
        vector: list[float] | None = None,
        top_k: int = 5,
        include_metadata: bool = True,
        include_vectors: bool = False,
    ) -> list[dict[str, Any]]:
        """Query similar vectors using raw text query (auto-embedded) or dense vector."""
        if not self.is_configured or httpx is None:
            return []

        payload: dict[str, Any] = {
            "topK": max(1, int(top_k)),
            "includeMetadata": bool(include_metadata),
            "includeVectors": bool(include_vectors),
        }
        if data is not None:
            payload["data"] = str(data)
            endpoint = f"{self.url}/query-data"
        elif vector is not None:
            payload["vector"] = list(vector)
            endpoint = f"{self.url}/query"
        else:
            return []

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                res = await client.post(endpoint, json=payload, headers=self._headers())
                if res.status_code < 400:
                    data_obj = res.json()
                    return list(data_obj.get("result") or [])
                logger.warning("Upstash Vector query failed status=%s: %s", res.status_code, res.text[:200])
        except Exception as exc:
            logger.warning("Upstash Vector query exception: %s", exc)
        return []

    async def delete(self, ids: list[str]) -> bool:
        """Delete vectors by ID."""
        if not self.is_configured or not ids or httpx is None:
            return False

        endpoint = f"{self.url}/delete"
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                res = await client.post(endpoint, json={"ids": ids}, headers=self._headers())
                return res.status_code < 400
        except Exception as exc:
            logger.warning("Upstash Vector delete exception: %s", exc)
            return False

    async def info(self) -> dict[str, Any] | None:
        """Retrieve index statistics (vector count, dimension, capacity)."""
        if not self.is_configured or httpx is None:
            return None

        endpoint = f"{self.url}/info"
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                res = await client.get(endpoint, headers=self._headers())
                if res.status_code < 400:
                    data_obj = res.json()
                    return dict(data_obj.get("result") or {})
        except Exception as exc:
            logger.warning("Upstash Vector info exception: %s", exc)
        return None


_GLOBAL_VECTOR_STORE = UpstashVectorStore()


def get_global_vector_store() -> UpstashVectorStore:
    return _GLOBAL_VECTOR_STORE


__all__ = [
    "UpstashVectorStore",
    "get_global_vector_store",
]
