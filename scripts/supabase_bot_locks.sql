-- Atomic distributed leases for bot scheduler/leader ownership.
-- Safe to run repeatedly in the Supabase SQL Editor.

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
