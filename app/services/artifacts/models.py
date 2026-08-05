"""Durable artifact metadata shared by background jobs and delivery code."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class ArtifactRef:
    """Small JSON-safe reference to content stored outside Redis job hashes."""

    id: str
    backend: str
    path: str
    content_type: str
    size_bytes: int
    sha256: str
    created_at: float
    expires_at: float | None = None
    bucket: str = ""

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ArtifactRef":
        return cls(
            id=str(value.get("id") or ""),
            backend=str(value.get("backend") or ""),
            path=str(value.get("path") or ""),
            content_type=str(value.get("content_type") or "application/octet-stream"),
            size_bytes=int(value.get("size_bytes") or 0),
            sha256=str(value.get("sha256") or ""),
            created_at=float(value.get("created_at") or 0.0),
            expires_at=(
                float(value["expires_at"])
                if value.get("expires_at") not in (None, "")
                else None
            ),
            bucket=str(value.get("bucket") or ""),
        )


__all__ = ["ArtifactRef"]
