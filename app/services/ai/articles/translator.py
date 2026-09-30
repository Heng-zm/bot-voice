"""Khmer and Multilingual Article Translation and News Summarization Engine."""

from __future__ import annotations

import asyncio
import contextlib
from typing import Any

_TRANSLATION_CACHE: dict[tuple[str, str], str] = {}


def is_khmer(text: str) -> bool:
    """Return True if text contains Khmer script characters."""
    if not text:
        return False
    return any(("\u1780" <= c <= "\u17ff") or ("\u19e0" <= c <= "\u19ff") for c in text)


def translate_chunk(chunk: str, target_lang: str = "km") -> str:
    """Translate a text chunk into target language."""
    if not chunk or not chunk.strip():
        return chunk
    with contextlib.suppress(Exception):
        from deep_translator import GoogleTranslator

        return GoogleTranslator(source="auto", target=target_lang).translate(chunk) or chunk
    return chunk


def translate_text(text: str, target_lang: str = "km") -> str:
    """Translate text into target language with caching and Khmer bypass."""
    if not text or not text.strip():
        return text
    if target_lang == "km" and is_khmer(text):
        return text

    cache_key = (text.strip(), target_lang)
    if cache_key in _TRANSLATION_CACHE:
        return _TRANSLATION_CACHE[cache_key]

    translated = translate_chunk(text.strip(), target_lang=target_lang)
    _TRANSLATION_CACHE[cache_key] = translated
    return translated


translate_text_sync = translate_text


async def translate_text_async(text: str, target_lang: str = "km") -> str:
    """Translate text into target language asynchronously with caching and Khmer bypass."""
    if not text or not text.strip():
        return text
    if target_lang == "km" and is_khmer(text):
        return text

    cache_key = (text.strip(), target_lang)
    if cache_key in _TRANSLATION_CACHE:
        return _TRANSLATION_CACHE[cache_key]

    return await asyncio.to_thread(translate_text, text, target_lang=target_lang)



def generate_smart_article_summary_sync(
    title: str,
    body_text: str,
    url: str = "",
    source_name: str = "",
) -> dict[str, Any]:
    """Generate structured studio-grade Khmer article summary matching modern media design."""
    if not body_text:
        return {
            "category": "ព័ត៌មាន",
            "badged_title": f"🗞️ {title}",
            "khmer_title": title,
            "hook": "",
            "sections": "",
            "takeaway": "",
            "khmer_summary": "",
            "original_summary": "",
        }

    # 1. Primary: Gemini AI (Direct Khmer Translation & News Summary)
    with contextlib.suppress(Exception):
        from app import legacy

        gemini_client = getattr(legacy, "_gemini", None)
        if not gemini_client:
            with contextlib.suppress(Exception):
                from app.services.ai.gemini import get_gemini_client

                gemini_client = get_gemini_client()

        if gemini_client:
            from app.services.ai.gemini import extract_gemini_text, generate_content_with_fallback

            preferred = getattr(legacy, "GEMINI_MODEL", "gemini-2.5-flash")

            prompt = (
                "You are an executive news editor for Bot Voice Cambodia. "
                "Read the following English article and create a high-quality news bulletin in Khmer.\n\n"
                "Return EXACTLY in this format:\n"
                "TITLE: <Professional translated Khmer headline>\n"
                "HOOK: <1-2 sentences in Khmer introducing the news>\n"
                "BULLETS:\n"
                "• <Bold Khmer Topic>: <Concise Khmer summary explanation>\n"
                "• <Bold Khmer Topic>: <Concise Khmer summary explanation>\n"
                "• <Bold Khmer Topic>: <Concise Khmer summary explanation>\n"
                "TAKEAWAY: <1 practical key takeaway in Khmer>\n\n"
                f"Original Title: {title}\n"
                f"Article Content: {body_text[:3500]}"
            )

            resp = generate_content_with_fallback(gemini_client, contents=prompt, preferred_model=preferred)
            ai_text = extract_gemini_text(resp)

            if ai_text and "BULLETS:" in ai_text:
                km_title = title
                hook = ""
                sections_list = []
                takeaway = ""

                lines = ai_text.strip().split("\n")
                section = ""
                for line in lines:
                    l = line.strip()
                    if l.startswith("TITLE:"):
                        km_title = l.replace("TITLE:", "").strip()
                    elif l.startswith("HOOK:"):
                        hook = l.replace("HOOK:", "").strip()
                    elif l.startswith("BULLETS:"):
                        section = "bullets"
                    elif l.startswith("TAKEAWAY:"):
                        takeaway = l.replace("TAKEAWAY:", "").strip()
                        section = "takeaway"
                    elif section == "bullets" and l.startswith("•"):
                        sections_list.append(l)

                sections = "ចំណុចសំខាន់ៗ៖\n" + "\n".join(sections_list) if sections_list else ""
                badged_title = f"📰 [ព័ត៌មានថ្មី]៖ {km_title} 📌"

                return {
                    "category": "ព័ត៌មានអន្តរជាតិ",
                    "badged_title": badged_title,
                    "khmer_title": km_title,
                    "hook": hook,
                    "sections": sections,
                    "takeaway": takeaway,
                    "khmer_summary": f"{hook}\n\n{sections}" if sections else hook,
                    "original_summary": body_text[:400],
                }

    # 2. Secondary Fallback: Multi-tier Article Translator
    from app.services.ai.summarizer import extractive_summary

    translate_fn = translate_text_sync

    paras = [p.strip() for p in body_text.split("\n\n") if p.strip()]
    paragraph_excerpt = "\n\n".join(paras[:2]) if len(paras) >= 2 else body_text[:800]

    try:
        summary_excerpt = extractive_summary(body_text, sentences_count=3, as_bullets=False)
        if not summary_excerpt:
            summary_excerpt = paragraph_excerpt
    except Exception:
        summary_excerpt = paragraph_excerpt

    try:
        km_title = translate_fn(title, target_lang="km") if translate_fn else title
        km_summary_raw = translate_fn(summary_excerpt, target_lang="km") if translate_fn else summary_excerpt
    except Exception:
        km_title = title
        km_summary_raw = summary_excerpt

    badged_title = f"📰 [ព័ត៌មានថ្មី]៖ {km_title} 📌"
    lines = [l.strip().lstrip("•-*▪►→").strip() for l in km_summary_raw.replace("|||", "\n").split("\n") if l.strip()]
    hook = lines[0] if lines else km_title
    bullet_lines = [f"• {line}" for line in lines[1:]] if len(lines) > 1 else [f"• {hook}"]
    sections = "ចំណុចសំខាន់ៗ៖\n" + "\n".join(bullet_lines)

    return {
        "category": "ព័ត៌មានទូទៅ",
        "badged_title": badged_title,
        "khmer_title": km_title,
        "hook": hook,
        "sections": sections,
        "takeaway": "តាមដានព័ត៌មានលម្អិតបន្ថែមតាមរយៈតំណភ្ជាប់ដើមនៃអត្ថបទ។",
        "khmer_summary": f"{hook}\n\n{sections}",
        "original_summary": paragraph_excerpt,
    }


__all__ = [
    "is_khmer",
    "translate_chunk",
    "translate_text",
    "translate_text_async",
    "translate_text_sync",
    "generate_smart_article_summary_sync",
]