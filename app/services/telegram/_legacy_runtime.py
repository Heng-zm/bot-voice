"""Temporary dependency bridge for extracted Telegram handlers.

V4.1 moves the live Telegram handler bodies out of :mod:`app.legacy`.  Many
helpers and process-local caches still live in the compatibility module, so the
extracted handlers bind only the legacy globals they actually reference at call
time.  This keeps runtime settings fresh and creates a narrow, measurable seam
for the next migration phase.

Delete this module once the remaining helpers have moved to native services.
"""

from __future__ import annotations

import builtins
import dis
import functools
import types
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

from app._legacy_bridge import legacy_module

_F = TypeVar("_F", bound=Callable[..., Awaitable[Any]])
_MISSING = object()


def _referenced_global_names(code: types.CodeType) -> frozenset[str]:
    """Return true global loads, excluding attribute/method names.

    ``code.co_names`` also contains attribute names used by LOAD_ATTR and
    LOAD_METHOD.  V4.1 treated all of them as legacy globals, causing dozens of
    unnecessary getattr()/AttributeError operations on every Telegram update.
    Inspecting bytecode keeps the bridge narrow and avoids polluting module
    globals with names that are never loaded globally.
    """

    names: set[str] = set()
    for instruction in dis.get_instructions(code):
        if instruction.opname == "LOAD_GLOBAL" and isinstance(instruction.argval, str):
            names.add(instruction.argval)
    for value in code.co_consts:
        if isinstance(value, types.CodeType):
            names.update(_referenced_global_names(value))
    return frozenset(names)


def legacy_bound_handler(func: _F) -> _F:
    """Bind referenced legacy helpers immediately before running a handler.

    The handler body lives in its native Telegram module.  Only unresolved
    dependencies are refreshed from ``app.legacy``.  Extracted handlers and
    module-native imports are never overwritten by compatibility symbols.
    """

    module_globals = func.__globals__
    native_names = frozenset(module_globals)
    dependency_names = _referenced_global_names(func.__code__)

    @functools.wraps(func)
    async def wrapped(*args: Any, **kwargs: Any) -> Any:
        legacy = legacy_module()
        for name in dependency_names:
            if name in native_names or hasattr(builtins, name):
                continue
            current = module_globals.get(name, _MISSING)
            if current is not _MISSING and bool(
                getattr(current, "__telegram_extracted_handler__", False)
            ):
                continue
            try:
                module_globals[name] = getattr(legacy, name)
            except AttributeError:
                # Preserve normal Python NameError behavior for genuinely
                # missing names instead of masking programming errors.
                module_globals.pop(name, None)
        return await func(*args, **kwargs)

    setattr(wrapped, "__telegram_extracted_handler__", True)
    setattr(wrapped, "__legacy_dependencies__", dependency_names)
    return wrapped  # type: ignore[return-value]
