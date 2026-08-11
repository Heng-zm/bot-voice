"""Atomic Supabase leases for scheduler and active-instance ownership."""

from __future__ import annotations

import threading
from datetime import UTC, datetime, timedelta
from typing import Any

BOT_LOCKS_SQL = """-- Atomic distributed leases for bot scheduler/leader ownership.
create table if not exists public.bot_locks (
  lock_key text primary key,
  owner text not null,
  locked_until timestamptz not null,
  updated_at timestamptz not null default now(),
  constraint bot_locks_key_length check (char_length(lock_key) between 1 and 240),
  constraint bot_locks_owner_length check (char_length(owner) between 1 and 240)
);

create index if not exists bot_locks_locked_until_idx
  on public.bot_locks (locked_until);

alter table public.bot_locks enable row level security;

drop policy if exists "service_role_bot_locks_all" on public.bot_locks;
create policy "service_role_bot_locks_all"
on public.bot_locks
for all
to service_role
using (true)
with check (true);

create or replace function public.acquire_bot_lock(
  p_lock_key text,
  p_owner text,
  p_ttl_seconds integer
)
returns boolean
language plpgsql
security definer
set search_path = public
as $$
declare
  acquired boolean := false;
  lease_now timestamptz := clock_timestamp();
begin
  if nullif(btrim(p_lock_key), '') is null or char_length(p_lock_key) > 240 then
    raise exception 'lock key must contain 1-240 characters';
  end if;
  if nullif(btrim(p_owner), '') is null or char_length(p_owner) > 240 then
    raise exception 'lock owner must contain 1-240 characters';
  end if;
  if p_ttl_seconds is null or p_ttl_seconds not between 1 and 86400 then
    raise exception 'lock TTL must be between 1 and 86400 seconds';
  end if;

  insert into public.bot_locks as current_lock (
    lock_key,
    owner,
    locked_until,
    updated_at
  )
  values (
    p_lock_key,
    p_owner,
    lease_now + make_interval(secs => p_ttl_seconds),
    lease_now
  )
  on conflict (lock_key) do update
  set owner = excluded.owner,
      locked_until = excluded.locked_until,
      updated_at = excluded.updated_at
  where current_lock.owner = excluded.owner
     or current_lock.locked_until <= lease_now
  returning true into acquired;

  return coalesce(acquired, false);
end;
$$;

revoke all on function public.acquire_bot_lock(text, text, integer)
  from public, anon, authenticated;
grant execute on function public.acquire_bot_lock(text, text, integer)
  to service_role;
"""


def _clean_identity(value: Any, label: str) -> str:
    clean = str(value or "").strip()
    if not clean or len(clean) > 240:
        raise ValueError(f"{label} must contain 1-240 characters.")
    return clean


def _clean_ttl(value: int | float) -> int:
    ttl = int(value)
    if not 1 <= ttl <= 86_400:
        raise ValueError("Lock TTL must be between 1 and 86400 seconds.")
    return ttl


def _response_bool(response: Any, function_name: str = "") -> bool:
    data = getattr(response, "data", None)
    if isinstance(data, bool):
        return data
    if isinstance(data, str):
        return data.strip().lower() in {"1", "true", "t", "yes"}
    if isinstance(data, dict):
        if function_name and function_name in data:
            return bool(data[function_name])
        return bool(data)
    if isinstance(data, list):
        if not data:
            return False
        first = data[0]
        if isinstance(first, dict) and function_name in first:
            return bool(first[function_name])
        return bool(first)
    return False


def _missing_acquire_rpc(exc: Exception) -> bool:
    message = str(exc).lower()
    return (
        "pgrst202" in message
        or "42883" in message
        or (
            "acquire_bot_lock" in message
            and any(
                marker in message
                for marker in ("could not find", "does not exist", "schema cache")
            )
        )
    )


class SupabaseLockService:
    """Acquire leases atomically, with a safe pre-migration fallback.

    The preferred RPC performs insert, renewal, and expired-owner takeover in
    one PostgreSQL statement using the database clock. If the migration is not
    installed, the fallback retains conditional updates and uses
    ``ON CONFLICT DO NOTHING`` for first creation, so concurrent acquisition
    never raises ``bot_locks_pkey``.
    """

    def __init__(self) -> None:
        self._state_lock = threading.RLock()
        self._client: Any | None = None
        self._rpc_available = True

    def _can_use_rpc(self, client: Any) -> bool:
        with self._state_lock:
            if client is not self._client:
                self._client = client
                self._rpc_available = True
            return self._rpc_available

    def _disable_rpc(self, client: Any) -> None:
        with self._state_lock:
            if client is self._client:
                self._rpc_available = False

    def acquire(
        self,
        client: Any,
        lock_key: str,
        owner: str,
        ttl_seconds: int | float,
    ) -> bool:
        clean_key = _clean_identity(lock_key, "Lock key")
        clean_owner = _clean_identity(owner, "Lock owner")
        clean_ttl = _clean_ttl(ttl_seconds)

        if self._can_use_rpc(client):
            try:
                response = client.rpc(
                    "acquire_bot_lock",
                    {
                        "p_lock_key": clean_key,
                        "p_owner": clean_owner,
                        "p_ttl_seconds": clean_ttl,
                    },
                ).execute()
                return _response_bool(response, "acquire_bot_lock")
            except Exception as exc:
                if not _missing_acquire_rpc(exc):
                    raise
                self._disable_rpc(client)

        return self._acquire_without_rpc(
            client,
            clean_key,
            clean_owner,
            clean_ttl,
        )

    @staticmethod
    def _acquire_without_rpc(
        client: Any,
        lock_key: str,
        owner: str,
        ttl_seconds: int,
    ) -> bool:
        now = datetime.now(UTC)
        now_iso = now.isoformat()
        update = {
            "owner": owner,
            "locked_until": (now + timedelta(seconds=ttl_seconds)).isoformat(),
            "updated_at": now_iso,
        }

        response = (
            client.table("bot_locks")
            .update(update)
            .eq("lock_key", lock_key)
            .eq("owner", owner)
            .execute()
        )
        if _response_bool(response):
            return True

        response = (
            client.table("bot_locks")
            .update(update)
            .eq("lock_key", lock_key)
            .lt("locked_until", now_iso)
            .execute()
        )
        if _response_bool(response):
            return True

        # resolution=ignore-duplicates maps to ON CONFLICT DO NOTHING. It
        # returns the inserted row to the winner and an empty list to losers.
        response = (
            client.table("bot_locks")
            .upsert(
                {"lock_key": lock_key, **update},
                on_conflict="lock_key",
                ignore_duplicates=True,
            )
            .execute()
        )
        return _response_bool(response)

    @staticmethod
    def release(client: Any, lock_key: str, owner: str) -> bool:
        clean_key = _clean_identity(lock_key, "Lock key")
        clean_owner = _clean_identity(owner, "Lock owner")
        response = (
            client.table("bot_locks")
            .delete()
            .eq("lock_key", clean_key)
            .eq("owner", clean_owner)
            .execute()
        )
        return _response_bool(response)

    @staticmethod
    def read(client: Any, lock_key: str) -> dict[str, Any] | None:
        clean_key = _clean_identity(lock_key, "Lock key")
        response = (
            client.table("bot_locks")
            .select("lock_key, owner, locked_until, updated_at")
            .eq("lock_key", clean_key)
            .limit(1)
            .execute()
        )
        rows = list(getattr(response, "data", None) or [])
        return dict(rows[0]) if rows else None


__all__ = ["BOT_LOCKS_SQL", "SupabaseLockService"]
