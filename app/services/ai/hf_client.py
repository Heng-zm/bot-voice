"""Hugging Face Inference Client for Qwen 2.5 and Open Models.

Provides serverless cloud inference via Hugging Face Router:
- Primary: Qwen/Qwen2.5-72B-Instruct (Flagship, elite Khmer comprehension & formatting)
- Fallback 1: Qwen/Qwen2.5-Coder-32B-Instruct
- Fallback 2: meta-llama/Llama-3.1-8B-Instruct

Zero RAM and zero GPU load on the local hosting server.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

import httpx

logger = logging.getLogger(__name__)

HF_ROUTER_URL = "https://router.huggingface.co/v1/chat/completions"

# Primary and fallback models supported on free serverless tier
HF_MODELS = [
    "Qwen/Qwen2.5-72B-Instruct",
    "Qwen/Qwen2.5-Coder-32B-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct",
]


def get_hf_token() -> str:
    """Retrieve Hugging Face API token from environment or .env."""
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        try:
            from dotenv import load_dotenv
            load_dotenv()
            token = os.environ.get("HF_TOKEN", "").strip()
        except Exception:
            pass
    if not token:
        try:
            from app.core.config import get_settings
            token = getattr(get_settings(), "HF_TOKEN", "").strip()
        except Exception:
            pass
    return token


def is_hf_available() -> bool:
    """Check if Hugging Face token is present."""
    return bool(get_hf_token())


_async_client: httpx.AsyncClient | None = None
_sync_client: httpx.Client | None = None

def get_async_client(timeout_s: float) -> httpx.AsyncClient:
    global _async_client
    if _async_client is None:
        _async_client = httpx.AsyncClient(timeout=timeout_s, limits=httpx.Limits(max_keepalive_connections=20, max_connections=100))
    else:
        _async_client.timeout = httpx.Timeout(timeout_s)
    return _async_client

def get_sync_client(timeout_s: float) -> httpx.Client:
    global _sync_client
    if _sync_client is None:
        _sync_client = httpx.Client(timeout=timeout_s, limits=httpx.Limits(max_keepalive_connections=20, max_connections=100))
    else:
        _sync_client.timeout = httpx.Timeout(timeout_s)
    return _sync_client


async def call_hf_chat(
    messages: list[dict[str, str]],
    model: str | None = None,
    max_tokens: int = 500,
    temperature: float = 0.6,
    timeout_s: float = 30.0,
) -> str | None:
    """Send chat completion request to Hugging Face Serverless Router with automatic model fallback."""
    token = get_hf_token()
    if not token:
        logger.debug("call_hf_chat: No HF_TOKEN found in environment.")
        return None

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "User-Agent": "BotVoice/2.0 (Khmer AI Intelligence)",
    }

    models_to_try = [model] if model else HF_MODELS
    client = get_async_client(timeout_s)

    for m in models_to_try:
        if not m:
            continue
        payload = {
            "model": m,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        try:
            resp = await client.post(HF_ROUTER_URL, json=payload, headers=headers)
            if resp.status_code == 200:
                data = resp.json()
                choices = data.get("choices") or []
                if choices:
                    content = choices[0].get("message", {}).get("content", "").strip()
                    if content:
                        logger.info("Hugging Face inference succeeded using model '%s'", m)
                        return content
            else:
                logger.debug("HF inference returned %d for model %s: %s", resp.status_code, m, resp.text[:120])
        except Exception as exc:
            logger.debug("HF inference error for model %s: %s", m, exc)

    return None


def call_hf_chat_sync(
    messages: list[dict[str, str]],
    model: str | None = None,
    max_tokens: int = 500,
    temperature: float = 0.6,
    timeout_s: float = 30.0,
) -> str | None:
    """Synchronous version of call_hf_chat for worker threads."""
    token = get_hf_token()
    if not token:
        return None

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "User-Agent": "BotVoice/2.0 (Khmer AI Intelligence)",
    }

    models_to_try = [model] if model else HF_MODELS
    client = get_sync_client(timeout_s)

    for m in models_to_try:
        if not m:
            continue
        payload = {
            "model": m,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        try:
            resp = client.post(HF_ROUTER_URL, json=payload, headers=headers)
            if resp.status_code == 200:
                data = resp.json()
                choices = data.get("choices") or []
                if choices:
                    content = choices[0].get("message", {}).get("content", "").strip()
                    if content:
                        logger.info("Hugging Face sync inference succeeded using model '%s'", m)
                        return content
            else:
                logger.debug("HF sync returned %d for model %s: %s", resp.status_code, m, resp.text[:120])
        except Exception as exc:
            logger.debug("HF sync error for model %s: %s", m, exc)

    return None


async def summarize_with_qwen(
    text: str,
    title: str = "",
    archetype: str = "GENERAL_NEWS",
) -> dict[str, str] | None:
    """Generate executive summary using Qwen 2.5 on Hugging Face (English first, then translated to Khmer)."""
    system_prompt = (
        "You are an executive news editor and intelligence analyst. "
        "Analyze the provided text and produce a high-impact, modern, and beautiful summary in English.\n\n"
        "STRICT FORMAT REQUIREMENTS:\n"
        "1. Badged Title: Choose exactly one suitable badge:\n"
        "   - [BREAKING]: <Short Title> (for urgent/breaking news)\n"
        "   - [SCAM ALERT]: <Short Title> (for scam, malware, fraud, phishing)\n"
        "   - [AI in 60s]: <Short Title> (for AI, tech, gadgets)\n"
        "   - [NEW TOOL]: <Short Title> (for tools, apps, resources)\n"
        "   - [LATEST NEWS]: <Short Title> (for general news)\n"
        "2. Hook: 1-2 conversational sentences explaining the core development and why it matters.\n"
        "3. Bullets: 3-4 structured bullet points with bold keywords:\n"
        "   • <b><Keyword>:</b> <Clear insight>\n"
        "4. Practical Takeaway: 💡 <Memorable practical advice or insight>\n\n"
        "RULES:\n"
        "- Total length MUST be strictly under 850 characters.\n"
        "- Do NOT use markdown code blocks or HTML tags other than <b>."
    )

    user_content = f"Title: {title}\n\nContent:\n{text[:4000]}" if title else f"Content:\n{text[:4000]}"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]

    raw_output = await call_hf_chat(messages, max_tokens=450, temperature=0.5)
    if not raw_output:
        return None

    parsed = parse_structured_khmer_summary(raw_output, fallback_title=title)
    
    # Translate English response to Khmer
    from app.services.ai.article_translator import translate_text_async
    async def _t(t: str) -> str:
        if not t: return ""
        return (await translate_text_async(t, "km")) or t
        
    khmer_title = await _t(parsed["khmer_title"])
    badged_title = parsed["badged_title"].replace(parsed["khmer_title"], khmer_title)
    badged_title = badged_title.replace("BREAKING", "BREAKING").replace("SCAM ALERT", "ប្រយ័ត្នបោក").replace("NEW TOOL", "ឧបករណ៍ថ្មី").replace("LATEST NEWS", "ព័ត៌មានថ្មី")
    
    hook = await _t(parsed["hook"])
    
    sections_lines = []
    import re as regex
    for line in parsed["sections"].split("\n"):
        if not line.strip(): continue
        m = regex.match(r"^(•\s*<b>)(.*?)(</b>\s*[:៖]?\s*)(.*)", line)
        if m:
            kw = await _t(m.group(2))
            body = await _t(m.group(4))
            sections_lines.append(f"{m.group(1)}{kw}{m.group(3)}{body}")
        else:
            sections_lines.append(await _t(line))
    sections = "\n".join(sections_lines)
    
    takeaway = await _t(parsed["takeaway"])
    
    full_summary = f"{hook}\n\n{sections}".strip() if hook else sections
    
    parsed.update({
        "khmer_title": khmer_title,
        "badged_title": badged_title,
        "hook": hook,
        "sections": sections,
        "takeaway": takeaway,
        "khmer_summary": full_summary,
    })
    
    return parsed


def summarize_with_qwen_sync(
    text: str,
    title: str = "",
    archetype: str = "GENERAL_NEWS",
) -> dict[str, str] | None:
    """Synchronous version of summarize_with_qwen (English -> Khmer)."""
    system_prompt = (
        "You are an executive news editor and intelligence analyst. "
        "Analyze the provided text and produce a high-impact, modern, and beautiful summary in English.\n\n"
        "STRICT FORMAT REQUIREMENTS:\n"
        "1. Badged Title: Choose exactly one suitable badge:\n"
        "   - [BREAKING]: <Short Title>\n"
        "   - [SCAM ALERT]: <Short Title>\n"
        "   - [AI in 60s]: <Short Title>\n"
        "   - [NEW TOOL]: <Short Title>\n"
        "   - [LATEST NEWS]: <Short Title>\n"
        "2. Hook: 1-2 conversational sentences explaining the core development.\n"
        "3. Bullets: 3-4 structured bullet points with bold keywords:\n"
        "   • <b><Keyword>:</b> <Clear insight>\n"
        "4. Practical Takeaway: 💡 <Memorable practical advice>\n\n"
        "RULES:\n"
        "- Total length MUST be strictly under 850 characters.\n"
        "- Do NOT use markdown code blocks or HTML tags other than <b>."
    )

    user_content = f"Title: {title}\n\nContent:\n{text[:4000]}" if title else f"Content:\n{text[:4000]}"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]

    raw_output = call_hf_chat_sync(messages, max_tokens=450, temperature=0.5)
    if not raw_output:
        return None

    parsed = parse_structured_khmer_summary(raw_output, fallback_title=title)
    
    from app.services.ai.article_translator import translate_text_sync
    def _ts(t: str) -> str:
        if not t: return ""
        return translate_text_sync(t, "km") or t
        
    khmer_title = _ts(parsed["khmer_title"])
    badged_title = parsed["badged_title"].replace(parsed["khmer_title"], khmer_title)
    badged_title = badged_title.replace("BREAKING", "BREAKING").replace("SCAM ALERT", "ប្រយ័ត្នបោក").replace("NEW TOOL", "ឧបករណ៍ថ្មី").replace("LATEST NEWS", "ព័ត៌មានថ្មី")
    
    hook = _ts(parsed["hook"])
    
    sections_lines = []
    import re as regex
    for line in parsed["sections"].split("\n"):
        if not line.strip(): continue
        m = regex.match(r"^(•\s*<b>)(.*?)(</b>\s*[:៖]?\s*)(.*)", line)
        if m:
            kw = _ts(m.group(2))
            body = _ts(m.group(4))
            sections_lines.append(f"{m.group(1)}{kw}{m.group(3)}{body}")
        else:
            sections_lines.append(_ts(line))
    sections = "\n".join(sections_lines)
    
    takeaway = _ts(parsed["takeaway"])
    
    full_summary = f"{hook}\n\n{sections}".strip() if hook else sections
    
    parsed.update({
        "khmer_title": khmer_title,
        "badged_title": badged_title,
        "hook": hook,
        "sections": sections,
        "takeaway": takeaway,
        "khmer_summary": full_summary,
    })
    
    return parsed


def parse_structured_khmer_summary(raw: str, fallback_title: str = "") -> dict[str, str]:
    """Parse raw LLM output into structured dict components for Telegram editorial card."""
    lines = [l.strip() for l in raw.strip().split("\n") if l.strip()]
    if not lines:
        return {
            "badged_title": fallback_title or "ព័ត៌មានថ្មី",
            "khmer_title": fallback_title or "ព័ត៌មានថ្មី",
            "hook": "",
            "sections": raw,
            "takeaway": "",
            "khmer_summary": raw,
            "raw_output": raw,
        }

    # Extract title from line 0
    first_line = lines[0]
    first_line = re.sub(r"^#+\s*", "", first_line).strip()
    first_line = re.sub(r"^\*\*|\*\*$", "", first_line).strip()

    badged_title = first_line
    clean_title = re.sub(r"^\[[^\]]+\]\s*[:៖]?\s*", "", badged_title).strip()
    if not clean_title:
        clean_title = fallback_title or "ព័ត៌មានថ្មី"

    # Identify takeaway line if present
    takeaway = ""
    bullets: list[str] = []
    hook_lines: list[str] = []

    for l in lines[1:]:
        if l.startswith("💡") or "takeaway" in l.lower() or "ដំបូន្មាន" in l:
            takeaway = re.sub(r"^💡\s*[:៖]?\s*", "", l).strip()
        elif l.startswith(("•", "-", "*", "▪", "►", "→", "១.", "២.", "៣.", "៤.", "1.", "2.", "3.", "4.")):
            # Normalize bullet point
            clean_b = re.sub(r"^[-*▪►→\d\.]+\s*", "", l).strip()
            if not clean_b.startswith("•"):
                clean_b = f"• {clean_b}"
            bullets.append(clean_b)
        else:
            if not bullets:
                hook_lines.append(l)
            else:
                bullets.append(f"• {l}")

    hook = " ".join(hook_lines).strip()
    sections = "\n".join(bullets)
    full_summary = f"{hook}\n\n{sections}".strip() if hook else sections

    # Detect category from badged title
    category = "ព័ត៌មានទូទៅ"
    if "BREAKING" in badged_title:
        category = "ព័ត៌មានទាន់ហេតុការណ៍"
    elif "ប្រយ័ត្នបោក" in badged_title or "SCAM" in badged_title:
        category = "សន្តិសុខ & បច្ចេកវិទ្យា"
    elif "AI" in badged_title:
        category = "បច្ចេកវិទ្យាថ្ងៃនេះ"
    elif "ឧបករណ៍" in badged_title or "TOOL" in badged_title:
        category = "ឧបករណ៍ និងកម្មវិធី"

    return {
        "category": category,
        "badged_title": badged_title,
        "khmer_title": clean_title,
        "hook": hook,
        "sections": sections,
        "takeaway": takeaway,
        "khmer_summary": full_summary,
        "raw_output": raw,
    }
