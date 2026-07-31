"""Supabase v2 compatibility, initialization, and query execution helpers."""

from app._legacy_bridge import exported_dir, exported_getattr

__all__ = [
    "_SupabaseClientV2Compat",
    "_SyncExecuteProxy",
    "_init_async_clients",
    "_init_clients",
    "_load_supabase_sdk",
    "_supabase_v2_compat_client",
    "_wrap_supabase_object",
    "db_call",
    "db_call_sync",
    "supabase",
    "supabase_async",
]

__getattr__ = exported_getattr(__name__, __all__)


def __dir__() -> list[str]:
    return exported_dir(globals(), __all__)
