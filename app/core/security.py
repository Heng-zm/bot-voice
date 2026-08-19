"""Single-process runtime secret policy.

This module replaces the old Redis-backed runtime secret manager. Secrets are
resolved from explicit environment variables first, otherwise deterministically
derived from the Telegram bot token with domain separation. The webhook secret
can also be read from the Supabase-backed settings store for rotation support.
"""

from __future__ import annotations

import hashlib
import os
import re
import secrets
from dataclasses import dataclass
from typing import Any

from app.services.settings.store import SettingsStore, get_settings_store

_WEBHOOK_KEY = "runtime:TELEGRAM_WEBHOOK_SECRET_TOKEN"
_WEBHOOK_RE = re.compile(r"^[A-Za-z0-9_-]{32,256}$")


def secret_fingerprint(value: str) -> str:
    clean = str(value or "")
    return hashlib.sha256(clean.encode("utf-8")).hexdigest()[:12] if clean else "not-set"


def derive_runtime_secret(bot_token: str, purpose: str) -> str:
    token = str(bot_token or "").strip()
    label = str(purpose or "runtime").strip().lower()
    if not token:
        return ""
    return hashlib.sha256(f"bot-voice:{label}:{token}".encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class RuntimeSecurityState:
    web_secret_key: str
    flask_secret_key: str
    webhook_secret_token: str
    web_source: str
    flask_source: str
    webhook_source: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "secrets": {
                "WEB_SECRET_KEY": {
                    "configured": bool(self.web_secret_key),
                    "source": self.web_source,
                    "fingerprint": secret_fingerprint(self.web_secret_key),
                },
                "FLASK_SECRET_KEY": {
                    "configured": bool(self.flask_secret_key),
                    "source": self.flask_source,
                    "fingerprint": secret_fingerprint(self.flask_secret_key),
                },
                "TELEGRAM_WEBHOOK_SECRET_TOKEN": {
                    "configured": bool(self.webhook_secret_token),
                    "source": self.webhook_source,
                    "fingerprint": secret_fingerprint(self.webhook_secret_token),
                },
            },
            "webhook_registration_required": False,
            "webhook_registered": False,
            "redis_removed": True,
        }


async def bootstrap_runtime_security(
    bot_token: str,
    *,
    settings_store: SettingsStore | None = None,
) -> RuntimeSecurityState:
    """Resolve stable secrets without any Redis dependency."""

    token = str(bot_token or "").strip()
    web_explicit = str(os.getenv("WEB_SECRET_KEY") or "").strip()
    flask_explicit = str(os.getenv("FLASK_SECRET_KEY") or "").strip()

    if len(web_explicit) >= 32:
        web_secret, web_source = web_explicit, "environment"
    elif len(flask_explicit) >= 32:
        web_secret, web_source = flask_explicit, "flask-environment"
    else:
        web_secret = derive_runtime_secret(token, "web-session")
        web_source = "telegram-token-derived" if web_secret else "process-local"
        if not web_secret:
            web_secret = secrets.token_urlsafe(48)

    if len(flask_explicit) >= 32:
        flask_secret, flask_source = flask_explicit, "environment"
    else:
        flask_secret, flask_source = web_secret, "web-secret"

    store = settings_store or get_settings_store()
    stored_webhook = await store.get_text(_WEBHOOK_KEY, "")
    env_webhook = str(
        os.getenv("TELEGRAM_WEBHOOK_SECRET_TOKEN")
        or os.getenv("TELEGRAM_WEBHOOK_SECRET")
        or ""
    ).strip()

    if _WEBHOOK_RE.fullmatch(stored_webhook):
        webhook_secret, webhook_source = stored_webhook, store.status.backend
    elif _WEBHOOK_RE.fullmatch(env_webhook):
        webhook_secret, webhook_source = env_webhook, "environment"
    else:
        webhook_secret = derive_runtime_secret(token, "telegram-webhook")
        if webhook_secret:
            webhook_source = "telegram-token-derived"
        else:
            webhook_secret = secrets.token_urlsafe(32)
            webhook_source = "process-local"
        # Best effort: when Supabase exists, persist non-explicit generated state
        # so even token-less local/custom deployments can remain stable.
        await store.set_text(_WEBHOOK_KEY, webhook_secret)

    return RuntimeSecurityState(
        web_secret_key=web_secret,
        flask_secret_key=flask_secret,
        webhook_secret_token=webhook_secret,
        web_source=web_source,
        flask_source=flask_source,
        webhook_source=webhook_source,
    )


__all__ = [
    "RuntimeSecurityState",
    "bootstrap_runtime_security",
    "derive_runtime_secret",
    "secret_fingerprint",
]
