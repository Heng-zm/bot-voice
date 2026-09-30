"""Temporary 15-Minute Email Service using GuerrillaMail API."""

import asyncio
import html as _html_mod
import json
import logging
import os
import time
from typing import Any, Dict

import httpx

logger = logging.getLogger(__name__)

STORAGE_PATH = "data/temp_emails.json"
API_BASE = "https://api.guerrillamail.com/ajax.php"
EMAIL_LIFESPAN = 15 * 60  # 15 minutes
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 BotVoice/1.0"


def _strip_html(raw_html: str) -> str:
    """Strip HTML tags from a string without external dependencies."""
    import re
    # Replace <br> and </p> with newlines
    clean = re.sub(r"<(br|/p|/div)[^>]*>", "\n", raw_html, flags=re.IGNORECASE)
    # Remove all other HTML tags
    clean = re.sub(r"<[^>]+>", "", clean)
    # Unescape HTML entities
    clean = _html_mod.unescape(clean)
    
    # Clean up whitespace:
    # 1. Trim spaces on each line
    lines = [line.strip() for line in clean.split('\n')]
    # 2. Join back
    clean = "\n".join(lines)
    # 3. Reduce 3+ consecutive newlines to just 2 newlines (paragraph break)
    clean = re.sub(r'\n{3,}', '\n\n', clean)
    
    return clean.strip()


class TempMailManager:
    def __init__(self) -> None:
        self.active_emails: Dict[str, Dict[str, Any]] = {}
        self._load_state()

    def _load_state(self) -> None:
        try:
            os.makedirs(os.path.dirname(STORAGE_PATH), exist_ok=True)
        except OSError:
            pass
        if os.path.exists(STORAGE_PATH):
            try:
                with open(STORAGE_PATH, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for uid, info in data.items():
                        info["seen_ids"] = set(info.get("seen_ids", []))
                        self.active_emails[uid] = info
            except Exception as e:
                logger.warning("Failed to load temp emails state: %s", e)

    def _save_state(self) -> None:
        try:
            data = {}
            for uid, info in self.active_emails.items():
                data[uid] = {
                    "email": info["email"],
                    "sid_token": info["sid_token"],
                    "expires_at": info["expires_at"],
                    "seen_ids": list(info.get("seen_ids", set())),
                }
            os.makedirs(os.path.dirname(STORAGE_PATH), exist_ok=True)
            with open(STORAGE_PATH, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False)
        except Exception as e:
            logger.warning("Failed to save temp emails state: %s", e)

    async def generate_email(self, user_id: int) -> str:
        """Generate a new disposable email address for a user."""
        async with httpx.AsyncClient(
            timeout=10.0, headers={"User-Agent": USER_AGENT}
        ) as client:
            resp = await client.get(f"{API_BASE}?f=get_email_address")
            resp.raise_for_status()
            data = resp.json()

        email = data["email_addr"]
        sid = data["sid_token"]

        self.active_emails[str(user_id)] = {
            "email": email,
            "sid_token": sid,
            "expires_at": time.time() + EMAIL_LIFESPAN,
            "seen_ids": {1},  # Skip the welcome email (mail_id=1)
        }
        self._save_state()
        return email

    async def poll_inboxes(self) -> None:
        """Background coroutine — checks all active inboxes every 10 seconds."""
        while True:
            try:
                await self._poll_once()
            except Exception as exc:
                logger.error("TempMail poll loop error: %s", exc)
            await asyncio.sleep(10)

    async def _poll_once(self) -> None:
        from app.bot import get_global_telegram_app

        app = get_global_telegram_app()
        if not app or not app.bot:
            return

        bot = app.bot
        now = time.time()

        # 1) Expire old emails
        expired = [uid for uid, info in self.active_emails.items() if info["expires_at"] < now]
        for uid in expired:
            try:
                await bot.send_message(
                    chat_id=int(uid),
                    text=(
                        "⚠️ <b>អ៊ីមែលបណ្ដោះអាសន្នរបស់អ្នកបានផុតកំណត់ហើយ!</b>\n\n"
                        "សូមវាយ /email ដើម្បីបង្កើតថ្មី។"
                    ),
                    parse_mode="HTML",
                )
            except Exception:
                pass
            # Guard against race: user may have created a new email during the await
            if self.active_emails.get(uid, {}).get("expires_at", 0) <= now:
                self.active_emails.pop(uid, None)
        if expired:
            self._save_state()

        if not self.active_emails:
            return

        # 2) Poll each active inbox
        async with httpx.AsyncClient(
            timeout=10.0, headers={"User-Agent": USER_AGENT}
        ) as client:
            for uid, info in list(self.active_emails.items()):
                try:
                    resp = await client.get(
                        f"{API_BASE}?f=get_email_list&offset=0&sid_token={info['sid_token']}"
                    )
                    if resp.status_code != 200:
                        continue
                    data = resp.json()
                    messages = data.get("list", []) if isinstance(data, dict) else []
                    for msg in messages:
                        msg_id = int(msg.get("mail_id", 0))
                        if not msg_id or msg_id in info["seen_ids"]:
                            continue
                        # Fetch full body BEFORE marking as seen
                        full = await client.get(
                            f"{API_BASE}?f=fetch_email&email_id={msg_id}"
                            f"&sid_token={info['sid_token']}"
                        )
                        if full.status_code == 200:
                            full_data = full.json()
                            if isinstance(full_data, dict):
                                await self._forward_to_user(bot, int(uid), full_data)
                                # Only mark as seen AFTER successful delivery
                                info["seen_ids"].add(msg_id)
                                self._save_state()
                except Exception as e:
                    logger.warning("Error polling email for uid=%s: %s", uid, e)

    async def _forward_to_user(self, bot: Any, user_id: int, email_data: dict) -> None:
        sender = _html_mod.escape(email_data.get("mail_from", "Unknown"))
        subject = _html_mod.escape(email_data.get("mail_subject", "(no subject)"))
        date = _html_mod.escape(email_data.get("mail_date", ""))

        body = email_data.get("mail_body", "")
        if body:
            body = _strip_html(body)
        body = _html_mod.escape(body)

        if len(body) > 3000:
            body = body[:3000] + "\n\n… [ខ្លឹមសារត្រូវបានកាត់ផ្តាច់]"

        text = (
            "📥 <b>អ្នកទទួលបានអ៊ីមែលថ្មី!</b>\n"
            "━━━━━━━━━━━━━━━━━━\n"
            f"👤 <b>ពី:</b> <code>{sender}</code>\n"
            f"🏷 <b>ប្រធានបទ:</b> {subject}\n"
            f"🕒 <b>ម៉ោង:</b> {date}\n"
            "━━━━━━━━━━━━━━━━━━\n"
            f"📝 <b>ខ្លឹមសារ:</b>\n<pre>{body}</pre>"
        )
        try:
            await bot.send_message(chat_id=user_id, text=text, parse_mode="HTML")
        except Exception as e:
            logger.warning("Failed to forward email to user %s: %s", user_id, e)


temp_mail_manager = TempMailManager()
