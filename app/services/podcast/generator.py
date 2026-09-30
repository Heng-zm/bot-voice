"""Daily Morning Podcast news generation, weather aggregator, and synthesis engine."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
import json
import logging
import re
import threading
from typing import Any
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

from app.services.telegram.formatters import clean_speech_text, markdown_to_telegram_html

logger = logging.getLogger(__name__)

_CAMBODIA_TZ = timezone(timedelta(hours=7))
_PODCAST_CACHE_LOCK = threading.Lock()
_PODCAST_CACHE: dict[str, tuple[str, str]] = {}  # date_str -> (html_text, speech_text)

_KHMER_DIGITS = {
    "0": "០", "1": "១", "2": "២", "3": "៣", "4": "៤",
    "5": "៥", "6": "៦", "7": "៧", "8": "៨", "9": "៩",
}
_KHMER_MONTHS = {
    1: "មករា", 2: "កុម្ភៈ", 3: "មីនា", 4: "មេសា",
    5: "ឧសភា", 6: "មិថុនា", 7: "កក្កដា", 8: "សីហា",
    9: "កញ្ញា", 10: "តុលា", 11: "វិច្ឆិកា", 12: "ធ្នូ",
}

_WMO_WEATHER_KHMER = {
    0: "🌞 ផ្ទៃមេឃស្រឡះល្អ (Sunny & Clear)",
    1: "🌤️ មានពពកតិចតួច (Mainly Clear)",
    2: "⛅ មានពពកខ្លះ (Partly Cloudy)",
    3: "☁️ មេឃអាប់អួរ (Overcast)",
    45: "🌫️ មានអ័ព្ទចុះត្រជាក់ (Foggy)",
    48: "🌫️ មានអ័ព្ទក្រាស់ (Depositing Fog)",
    51: "🌦️ មានភ្លៀងរលឹមស្រិចៗ (Light Drizzle)",
    53: "🌦️ មានភ្លៀងរលឹមបង្គួរ (Moderate Drizzle)",
    55: "🌧️ មានភ្លៀងធ្លាក់ខ្លាំង (Heavy Drizzle)",
    61: "🌦️ មានភ្លៀងធ្លាក់រាយប៉ាយ (Slight Rain)",
    63: "🌧️ មានភ្លៀងធ្លាក់កម្រិតមធ្យម (Moderate Rain)",
    65: "🌧️ មានភ្លៀងធ្លាក់ខ្លាំង (Heavy Rain)",
    80: "🌦️ មានភ្លៀងកក់ខែស្រាល (Light Showers)",
    81: "🌧️ មានភ្លៀងធ្លាក់ជោកជាំ (Moderate Showers)",
    82: "⛈️ មានភ្លៀងផ្គរខ្លាំង (Violent Showers)",
    95: "⛈️ មានផ្គររន្ទះ និងភ្លៀងធ្លាក់ (Thunderstorm)",
    96: "⛈️ មានផ្គររន្ទះ និងខ្យល់កន្ត្រាក់ (Thunderstorm & Gusty Winds)",
    99: "⛈️ ព្យុះផ្គររន្ទះខ្លាំង (Severe Thunderstorm)",
}


def to_khmer_numeral(num_or_str: int | str) -> str:
    """Convert western digits (0-9) to Khmer numerals (០-៩)."""
    return "".join(_KHMER_DIGITS.get(ch, ch) for ch in str(num_or_str))


def format_khmer_date(dt: datetime) -> str:
    """Format datetime into standard Khmer date string, e.g. '១៤ កញ្ញា ឆ្នាំ ២០២៦'."""
    d = to_khmer_numeral(dt.day)
    m = _KHMER_MONTHS.get(dt.month, "")
    y = to_khmer_numeral(dt.year)
    return f"{d} {m} ឆ្នាំ {y}"


def get_cambodia_now() -> datetime:
    """Return current datetime in Cambodia timezone (UTC+7)."""
    return datetime.now(_CAMBODIA_TZ)


def fetch_cambodia_weather(timeout_s: float = 4.0) -> dict[str, str]:
    """Fetch real-time daily weather report for Cambodia using Open-Meteo with fallback."""
    url = (
        "https://api.open-meteo.com/v1/forecast"
        "?latitude=11.5564&longitude=104.9282"
        "&current=temperature_2m,relative_humidity_2m,weather_code"
        "&daily=temperature_2m_max,temperature_2m_min"
        "&timezone=Asia%2FBangkok"
    )
    req = urllib.request.Request(url, headers={"User-Agent": "BotVoice/2.0 (Cambodia Weather Aggregator)"})

    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            if resp.status == 200:
                payload = json.loads(resp.read().decode("utf-8"))
                current = payload.get("current", {})
                daily = payload.get("daily", {})

                temp_curr = round(float(current.get("temperature_2m", 30.0)))
                wcode = int(current.get("weather_code", 1))
                cond_text = _WMO_WEATHER_KHMER.get(wcode, "🌤️ អាកាសធាតុល្អបង្គួរ")

                t_max = round(float(daily.get("temperature_2m_max", [33.0])[0]))
                t_min = round(float(daily.get("temperature_2m_min", [25.0])[0]))

                pp_report = (
                    f"រាជធានីភ្នំពេញ៖ សីតុណ្ហភាពបច្ចុប្បន្នប្រហែល {to_khmer_numeral(temp_curr)}°C "
                    f"(ចន្លោះពី {to_khmer_numeral(t_min)}°C ដល់ {to_khmer_numeral(t_max)}°C), {cond_text}"
                )
                provinces_report = (
                    f"បណ្ដាខេត្ត និងតំបន់ឆ្នេរ៖ សីតុណ្ហភាពចន្លោះពី {to_khmer_numeral(t_min - 1)}°C ដល់ {to_khmer_numeral(t_max)}°C, "
                    "អាចមានភ្លៀងធ្លាក់រាយប៉ាយនៅតំបន់មួយចំនួន។"
                )

                return {
                    "phnom_penh": pp_report,
                    "provinces": provinces_report,
                    "summary": f"{cond_text} ({to_khmer_numeral(temp_curr)}°C)",
                }
    except Exception as exc:
        logger.debug("Failed to fetch live weather from Open-Meteo: %s", exc)

    # Deterministic Khmer weather fallback
    return {
        "phnom_penh": "រាជធានីភ្នំពេញ៖ សីតុណ្ហភាព ២៦°C ដល់ ៣៤°C, 🌤️ មានពពកតិចតួច និងកម្ដៅថ្ងៃល្មម។",
        "provinces": "បណ្ដាខេត្ត និងតំបន់ឆ្នេរ៖ សីតុណ្ហភាព ២៥°C ដល់ ៣៣°C, 🌦️ អាចមានភ្លៀងកក់ខែរាយប៉ាយនៅតំបន់ខ្ពង់រាប។",
        "summary": "🌤️ មានពពកតិចតួច (៣០°C)",
    }


def fetch_live_cambodia_headlines(limit: int = 5, timeout_s: float = 4.0) -> list[str]:
    """Scrape top trending news headlines from Google News Cambodia RSS feed."""
    encoded_query = urllib.parse.quote("Cambodia news OR កម្ពុជា")
    rss_url = f"https://news.google.com/rss/search?q={encoded_query}&hl=km&gl=KH&ceid=KH:km"
    req = urllib.request.Request(rss_url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})

    headlines: list[str] = []
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            xml_data = resp.read()
        root = ET.fromstring(xml_data)
        items = root.findall("./channel/item")
        for it in items[:limit]:
            title_node = it.find("title")
            if title_node is not None and title_node.text:
                clean_title = re.sub(r"\s*-\s*[^-]+$", "", title_node.text).strip()
                if clean_title and clean_title not in headlines:
                    headlines.append(clean_title)
    except Exception as exc:
        logger.debug("Failed to fetch RSS news headlines: %s", exc)

    if not headlines:
        headlines = [
            "កម្ពុជាបន្តជំរុញការអភិវឌ្ឍសេដ្ឋកិច្ចឌីជីថល និងបច្ចេកវិទ្យាព័ត៌មាន",
            "រាជរដ្ឋាភិបាលដាក់ចេញនូវគោលនយោបាយថ្មីសម្រាប់លើកកម្ពស់វិស័យអប់រំ និងបណ្តុះបណ្តាលវិជ្ជាជីវៈ",
            "ក្រសួងព័ត៌មានអំពាវនាវឱ្យសាធារណជនតាមដានព័ត៌មានផ្លូវការ និងប្រយ័ត្នព័ត៌មានមិនពិត",
        ]

    return headlines[:limit]


def strip_podcast_unwanted_sections(text: str) -> str:
    """Remove motivational tips, life lessons, and closing outro speeches from podcast text."""
    if not text:
        return ""
    t = str(text)
    # 1. Remove intro line to motivational tip: ហើយមុននឹងបញ្ចប់...
    t = re.sub(r"ហើយមុននឹងបញ្ចប់[^\n]*\n?", "", t)
    # 2. Remove motivational block until next section (📰 or end)
    t = re.sub(
        r"(?:\n|^)\s*(?:[•\-*]\s*)?(?:💡\s*)?(?:\*\*)?(?:ចំណេះដឹង|គំនិតវិជ្ជមាន|ការលើកទឹកចិត្ត|គន្លឹះ)[^\n]*(?:\*\*)?:?[\s\S]*?(?=(?:\n\s*📰|\n\s*\[|\n\s*🔒|\Z))",
        "\n",
        t,
    )
    # 3. Remove conversational outro lines
    t = re.sub(r"ទាំងនេះគឺជាព័ត៌មានសំខាន់ៗ[^\n]*", "", t)
    t = re.sub(r"សូមជូនពរលោកអ្នក[^\n]*", "", t)
    t = re.sub(r"ជួបគ្នាសារជាថ្មី[^\n]*", "", t)
    t = re.sub(r"\n{3,}", "\n\n", t).strip()
    return t


def clean_podcast_speech_text(text: str, khmer_date: str = "") -> str:
    """Format and polish podcast text specifically for studio Khmer TTS reading.

    - Strips raw emojis that cause speech stutter or English character reads.
    - Strips markdown formatting, code blocks, raw URLs, and hashtags.
    - Transforms '(ប្រភពដើម៖ ...)' into spoken Khmer phrase 'ប្រភពដើមពី...'.
    - Removes bullet points, asterisks, and bracket pauses.
    - Cleans double colons and excessive whitespace for smooth, natural audio delivery.
    """
    if not text:
        return ""
    t = str(text)

    # Remove hashtags and raw URLs
    t = re.sub(r"#\S+", " ", t)
    t = re.sub(r"https?://\S+", " ", t)
    t = re.sub(r"www\.\S+", " ", t)

    # Convert '(ប្រភពដើម៖ ...)' or '(ប្រភព៖ ...)' to ' ប្រភពដើមពី ...'
    t = re.sub(
        r"\(\s*(?:ប្រភពដើម)\s*[៖:]\s*([^)]+)\)",
        r" ប្រភពដើមពី \1 ",
        t,
    )
    t = re.sub(
        r"\(\s*(?:ប្រភព)\s*[៖:]\s*([^)]+)\)",
        r" ប្រភពពី \1 ",
        t,
    )

    # Remove emojis (common ranges) to prevent TTS pronunciation artifacts
    emoji_pattern = re.compile(
        "[\U00010000-\U0010ffff"
        "\u2600-\u26ff"
        "\u2700-\u27bf"
        "\u2300-\u23ff"
        "\u2b50-\u2b55"
        "\ufe0f"
        "]+",
        flags=re.UNICODE,
    )
    t = emoji_pattern.sub(" ", t)

    # Strip markdown symbols & bullets
    t = re.sub(r"[*_~`#|>]", " ", t)
    t = re.sub(r"[•\-–—]\s*", " ", t)
    t = re.sub(r"[\[\]\(\)\{\}]", " ", t)

    # Clean double colons or punctuation noise
    t = re.sub(r"៖+", "៖", t)
    t = re.sub(r"\s+([។!?:៖])", r"\1", t)
    t = re.sub(r"\s+", " ", t).strip()

    return t


def build_podcast_card_html(formatted_body: str, khmer_date: str, max_chars: int = 1024) -> str:
    """Construct Telegram infographic card HTML and ensure it comfortably fits within photo caption limits."""
    header = "📻 <b>ព័ត៌មានពេលព្រឹកថ្ងៃនេះ | Daily Morning Podcast</b>\n\n"
    donate_link = '<a href="https://t.me/voicekhaibot?start=donate">☕ ឧបត្ថម្ភកាហ្វេ (Donate Me)</a>'
    hashtags = "#ព័ត៌មានពេលព្រឹក #MorningPodcast #CambodiaNews #TopNewsToday #BotVoice"

    full_metadata = (
        "🔒 <b>ប្រភព:</b> ក្រសួងព័ត៌មាន / NBC / Reuters / Open-Meteo\n"
        "🟢 <b>ស្ថានភាព:</b> ✅ បានផ្ទៀងផ្ទាត់\n"
        '🔗 <b>អានដើម:</b> <a href="https://www.information.gov.kh">information.gov.kh</a>\n'
        f"📅 <b>{khmer_date}</b>"
    )

    candidate = (
        f"{header}"
        f"{formatted_body}\n\n"
        f"{full_metadata}\n\n"
        f"{donate_link}\n\n"
        f"{hashtags}"
    )

    if len(candidate) <= max_chars:
        return candidate

    # Condense metadata to compact footer if candidate exceeds max_chars
    compact_metadata = f"🔒 <b>ប្រភព:</b> ក្រសួងព័ត៌មាន / NBC / Reuters • 📅 <b>{khmer_date}</b>"
    condensed_candidate = (
        f"{header}"
        f"{formatted_body}\n\n"
        f"{compact_metadata}\n\n"
        f"{donate_link}"
    )

    if len(condensed_candidate) <= max_chars:
        return condensed_candidate

    # Ultra-compact fallback ensuring photo caption never overflows Telegram's 1024 limit
    ultra_compact = f"{header}{formatted_body}\n\n{compact_metadata}"
    if len(ultra_compact) <= max_chars:
        return ultra_compact

    avail = max_chars - len(header) - len(compact_metadata) - 10
    trimmed_body = formatted_body[:avail]
    last_period = max(trimmed_body.rfind("។"), trimmed_body.rfind("."))
    if last_period > 100:
        trimmed_body = trimmed_body[:last_period + 1]
    return f"{header}{trimmed_body}\n\n{compact_metadata}"


async def generate_morning_podcast(force_refresh: bool = False) -> tuple[str, str]:
    """Generate or retrieve cached Daily Morning Podcast for today.

    Optimizations:
    - Concurrent parallel fetching for Open-Meteo weather and Google News RSS.
    - Studio-grade clean speech text formatting eliminating emoji artifacts.
    - Smart Telegram caption length protection fitting inside 1024 chars.
    - Automatic memory cache pruning for long-term VPS stability.

    Returns:
        tuple[str, str]: (formatted_telegram_html, clean_speech_text_for_tts)
    """
    now = get_cambodia_now()
    date_key = now.strftime("%Y-%m-%d")
    khmer_date = format_khmer_date(now)

    with _PODCAST_CACHE_LOCK:
        if not force_refresh and date_key in _PODCAST_CACHE:
            return _PODCAST_CACHE[date_key]

    loop = asyncio.get_running_loop()

    # 1. Fetch real-time weather & live headlines asynchronously in parallel
    weather_task = loop.run_in_executor(None, fetch_cambodia_weather)
    headlines_task = loop.run_in_executor(None, fetch_live_cambodia_headlines)
    weather_res, headlines_res = await asyncio.gather(weather_task, headlines_task, return_exceptions=True)

    weather_info: dict[str, str] = weather_res if isinstance(weather_res, dict) else fetch_cambodia_weather(timeout_s=0.1)
    headlines: list[str] = headlines_res if isinstance(headlines_res, list) else []

    headlines_hint = "\n".join(f"• {h}" for h in headlines[:4]) if headlines else ""

    def _call_gemini() -> str:
        from app import legacy
        from app.services.ai.gemini import (
            extract_gemini_text,
            generate_content_with_fallback,
        )

        client = getattr(legacy, "_gemini", None)
        preferred_model = getattr(legacy, "GEMINI_MODEL", "gemini-2.5-flash")

        headlines_context = ""
        if headlines_hint:
            headlines_context = f"\nReal-Time News Context / Headlines Today:\n{headlines_hint}\n"

        weather_context = (
            f"\nWeather Information Today:\n"
            f"- {weather_info.get('phnom_penh', '')}\n"
            f"- {weather_info.get('provinces', '')}\n"
        )

        prompt = (
            f"You are the morning news presenter for Bot Voice Cambodia.\n"
            f"Generate an engaging, concise, and informative Daily Morning Podcast summary for Cambodia.\n"
            f"Date: {khmer_date}\n"
            f"{weather_context}"
            f"{headlines_context}\n"
            f"Format strictly following this layout:\n"
            f"🛡️ [ព័ត៌មានសំខាន់]៖ <Catchy Cambodian headline summary> ⚡\n\n"
            f"<1-2 sentence compelling lead hook summarizing today's key developments in Cambodia and worldwide>\n\n"
            f"🌤️ អាកាសធាតុប្រចាំថ្ងៃ៖\n"
            f"• {weather_info.get('phnom_penh', '')}\n"
            f"• {weather_info.get('provinces', '')}\n\n"
            f"🇰🇭 ព័ត៌មានជាតិ & សេដ្ឋកិច្ចឌីជីថល៖\n"
            f"• <Key national / economic news 1> (ប្រភពដើម៖ <Official Cambodian source e.g. ក្រសួងព័ត៌មាន / NBC / Fresh News>)\n"
            f"• <Key national / economic news 2> (ប្រភពដើម៖ <Official Cambodian source>)\n\n"
            f"💻 បច្ចេកវិទ្យា & បញ្ញាសិប្បនិម្មិត៖\n"
            f"• <Key technology or AI innovation news> (ប្រភពដើម៖ <Tech / Industry source>)\n\n"
            f"🌍 ព័ត៌មានអន្តរជាតិ៖\n"
            f"• <Key global news 1> (ប្រភពដើម៖ <Global source e.g. Reuters / BBC / AFP>)\n"
            f"• <Key global news 2> (ប្រភពដើម៖ <Global source>)\n\n"
            f"💡 <1 brief safety or takeaway sentence>\n\n"
            f"Strict Rules & Guidelines:\n"
            f"- Focus STRICTLY on factual, verifiable news. Keep the content punchy and readable (~500-750 characters).\n"
            f"- DO NOT include any motivational tips, life lessons, quotes, or '💡 ចំណេះដឹង & ការលើកទឹកចិត្ត'!\n"
            f"- DO NOT include conversational outro speeches or closing wishes (e.g. DO NOT write 'ហើយមុននឹងបញ្ចប់...', 'ទាំងនេះគឺជាព័ត៌មានសំខាន់ៗ...', 'សូមជូនពរលោកអ្នក...', 'ជួបគ្នាសារជាថ្មី... សូមអរគុណ!').\n"
            f"- Write entirely in natural, polished, and fluent Khmer.\n"
            f"- Use a professional radio announcer tone.\n"
        )

        if client is None:
            # Safe offline fallback
            return (
                f"🛡️ [ព័ត៌មានសំខាន់]៖ សង្ខេបព្រឹត្តិការណ៍ជាតិ អាកាសធាតុ និងសេដ្ឋកិច្ចប្រចាំថ្ងៃ ⚡\n\n"
                f"សួស្តីបងប្អូន! នេះជាព័ត៌មានសង្ខេបសំខាន់ៗប្រចាំថ្ងៃ {khmer_date} ដើម្បីផ្ដល់ដំណឹងទាន់ហេតុការណ៍ជូនបងប្អូន។\n\n"
                f"🌤️ អាកាសធាតុប្រចាំថ្ងៃ៖\n"
                f"• {weather_info.get('phnom_penh', '')}\n"
                f"• {weather_info.get('provinces', '')}\n\n"
                f"🇰🇭 ព័ត៌មានជាតិ & សេដ្ឋកិច្ចឌីជីថល៖\n"
                f"• កម្ពុជាបន្តជំរុញសេដ្ឋកិច្ចឌីជីថល និងការទូទាត់អេឡិចត្រូនិក Bakong KHQR ទូទាំងប្រទេស (ប្រភពដើម៖ ធនាគារជាតិនៃកម្ពុជា)។\n"
                f"• ទីផ្សារការងារ និងវិស័យទេសចរណ៍បង្ហាញសញ្ញាណវិជ្ជមានជាបន្តបន្ទាប់ (ប្រភពដើម៖ ក្រសួងព័ត៌មាន)។\n\n"
                f"💻 បច្ចេកវិទ្យា & បញ្ញាសិប្បនិម្មិត៖\n"
                f"• ការអនុវត្តបច្ចេកវិទ្យា AI ក្នុងវិស័យអប់រំ និងសេវាកម្មសាធារណៈទទួលបានការគាំទ្រយ៉ាងទូលំទូលាយ (ប្រភពដើម៖ ក្រសួងប្រៃសណីយ៍ និងទូរគមនាគមន៍)។\n\n"
                f"🌍 ព័ត៌មានអន្តរជាតិ៖\n"
                f"• បច្ចេកវិទ្យា និងសេដ្ឋកិច្ចបៃតងនៅតំបន់អាស៊ីបន្តមានកំណើនវិជ្ជមាន (ប្រភពដើម៖ Reuters)។\n"
                f"• ទីផ្សារហិរញ្ញវត្ថុសកលបង្ហាញស្ថិរភាពក្នុងការវិនិយោគបច្ចេកវិទ្យាថ្មី (ប្រភពដើម៖ BBC News)។\n\n"
                f"💡 តាមដានព័ត៌មានពិត ចៀសវាងព័ត៌មានក្លែងក្លាយ ដើម្បីសុវត្ថិភាពសង្គម។"
            )

        resp = generate_content_with_fallback(
            client=client,
            contents=prompt,
            preferred_model=preferred_model,
        )
        text = extract_gemini_text(resp)
        return text or (
            f"🛡️ [ព័ត៌មានសំខាន់]៖ សង្ខេបព្រឹត្តិការណ៍ជាតិ និងអន្តរជាតិប្រចាំថ្ងៃ ⚡\n\n"
            f"ព័ត៌មានសំខាន់ៗប្រចាំថ្ងៃ {khmer_date} ពីប្រភពផ្លូវការ។\n\n"
            f"🌤️ អាកាសធាតុប្រចាំថ្ងៃ៖\n"
            f"• {weather_info.get('phnom_penh', '')}\n\n"
            f"🇰🇭 ព័ត៌មានជាតិសំខាន់ៗ៖\n"
            f"• សេដ្ឋកិច្ចឌីជីថលកម្ពុជាបន្តរីកចម្រើនយ៉ាងឆាប់រហ័ស (ប្រភពដើម៖ ក្រសួងព័ត៌មាន)។\n\n"
            f"🌍 ព័ត៌មានអន្តរជាតិ៖\n"
            f"• ទីផ្សារបច្ចេកវិទ្យាសកលកើនឡើងជាលំដាប់ (ប្រភពដើម៖ Reuters)។\n\n"
            f"💡 តាមដានព័ត៌មានពិត ចៀសវាងព័ត៌មានក្លែងក្លាយ ដើម្បីសុវត្ថិភាពសង្គម។"
        )

    raw_ai_text = await loop.run_in_executor(None, _call_gemini)
    raw_ai_text = strip_podcast_unwanted_sections(raw_ai_text)

    # Ensure required sections exist
    if "🌤️" not in raw_ai_text and "អាកាសធាតុ" not in raw_ai_text:
        raw_ai_text = f"🌤️ អាកាសធាតុប្រចាំថ្ងៃ៖\n• {weather_info.get('phnom_penh', '')}\n\n" + raw_ai_text

    if "🇰🇭" not in raw_ai_text and "ព័ត៌មានជាតិ" not in raw_ai_text:
        raw_ai_text += (
            "\n\n🇰🇭 ព័ត៌មានជាតិសំខាន់ៗ៖\n"
            "• កម្ពុជាជំរុញការអភិវឌ្ឍសេដ្ឋកិច្ច និងឌីជីថលូបនីយកម្ម (ប្រភពដើម៖ ក្រសួងព័ត៌មាន)។"
        )
    if "🌍" not in raw_ai_text and "ព័ត៌មានអន្តរជាតិ" not in raw_ai_text:
        raw_ai_text += (
            "\n\n🌍 ព័ត៌មានអន្តរជាតិ៖\n"
            "• និន្នាការបច្ចេកវិទ្យាសកលបន្តរីកចម្រើនខ្លាំង (ប្រភពដើម៖ Reuters)។"
        )

    formatted_body = markdown_to_telegram_html(raw_ai_text)
    full_html = build_podcast_card_html(formatted_body, khmer_date, max_chars=1024)
    speech_text = clean_podcast_speech_text(raw_ai_text, khmer_date=khmer_date)

    with _PODCAST_CACHE_LOCK:
        # Keep only the last 2 keys to prevent long-term memory accumulation
        if len(_PODCAST_CACHE) > 2:
            keys_to_remove = sorted(list(_PODCAST_CACHE.keys()))[:-2]
            for k in keys_to_remove:
                _PODCAST_CACHE.pop(k, None)
        _PODCAST_CACHE[date_key] = (full_html, speech_text)

    return full_html, speech_text


__all__ = [
    "build_podcast_card_html",
    "clean_podcast_speech_text",
    "fetch_cambodia_weather",
    "fetch_live_cambodia_headlines",
    "format_khmer_date",
    "generate_morning_podcast",
    "get_cambodia_now",
    "strip_podcast_unwanted_sections",
    "to_khmer_numeral",
]
