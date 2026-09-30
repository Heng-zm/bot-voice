"""Extractive News Summarizer optimized for Telegram news bulletins.

Features:
  1. Khmer-aware & Latin-aware intelligent sentence splitting
  2. Multi-language boilerplate & ad stripping (English & Khmer news patterns)
  3. Hybrid TF-IDF token weighting (character 3-grams for Khmer, word tokens for Latin)
  4. Inverted-pyramid journalism position boosts (Lead sentence prioritization)
  5. Cosine-similarity deduplication (eliminates repetitive bullet points)
  6. Khmer Unicode-safe boundary truncation (never cuts combining marks/coeng)
  7. High performance & low memory footprint (thread-safe, cached stopwords)
"""

from __future__ import annotations

from collections import Counter
import logging
import math
import re
import threading
from typing import List

logger = logging.getLogger(__name__)

# Optional NLTK — graceful fallback if not installed
try:
    import nltk
    from nltk.tokenize import sent_tokenize as _nltk_sent_tokenize
    from nltk.corpus import stopwords as _nltk_stopwords
    import threading as _threading

    _nltk_lock = _threading.Lock()
    with _nltk_lock:
        for _pkg, _path in [
            ("punkt", "tokenizers/punkt"),
            ("punkt_tab", "tokenizers/punkt_tab"),
            ("stopwords", "corpora/stopwords"),
        ]:
            try:
                nltk.data.find(_path)
            except LookupError:
                nltk.download(_pkg, quiet=True)
    _STOP_WORDS: set = set(_nltk_stopwords.words("english"))
    _HAS_NLTK = True
except Exception:
    _nltk_sent_tokenize = None
    _STOP_WORDS = {
        "the", "a", "an", "is", "in", "on", "at", "to", "for", "of", "and", "or",
        "but", "with", "by", "from", "that", "this", "it", "as", "are", "was",
        "were", "be", "been", "have", "has", "had", "not", "no", "its", "their",
    }
    _HAS_NLTK = False

# ── Boilerplate Patterns (English & Khmer) ──────────────────────────────────
_BOILERPLATE_PATTERNS = [
    # English publisher & social junk
    r"click here",
    r"subscribe",
    r"read more",
    r"follow us",
    r"all rights reserved",
    r"photo by",
    r"image source",
    r"copyright \d{4}",
    r"sign up for",
    r"download our app",
    r"advertisement",
    r"sponsored content",
    r"terms of service",
    r"privacy policy",
    r"newsletter",
    r"stay updated",
    r"join our telegram",
    r"join our channel",
    r"share this article",
    r"for more information",
    # Khmer publisher & social junk
    r"ចុចអាន(?:បន្ត|បន្ថែម)?",
    r"សូមចុច",
    r"តាមដាន(?:ព័ត៌មាន)?",
    r"ទស្សនាវីដេអូ",
    r"ទាញយកកម្មវិធី",
    r"រក្សាសិទ្ធិ",
    r"ផ្សាយពាណិជ្ជកម្ម",
    r"ទំនាក់ទំនងផ្សាយ",
    r"តំណភ្ជាប់",
    r"អត្ថបទដោយ",
    r"រូបភាពដោយ",
    r"ប្រភព(?:ព័ត៌មាន)?\s*[:៖]",
    r"អានព័ត៌មានលម្អិត",
    r"ចូលរួម(?:ជាមួយ)?\s*telegram",
]
_BOILERPLATE_RE = re.compile("|".join(_BOILERPLATE_PATTERNS), re.IGNORECASE)

_KHMER_CHAR_RE = re.compile(r"[\u1780-\u17FF]")
_KHMER_CLAUSE_SPLIT = re.compile(
    r"\s+(?:ខណៈដែល|ដោយឡែក|ទន្ទឹមនឹងនេះ|លើសពីនេះ|គួរបញ្ជាក់ថា|ជាមួយគ្នានេះ|បន្ថែមលើនេះ|ក្នុងនោះ)\s+"
)

# Telegram photo caption constraints (~1020 chars total)
_MAX_SENTENCE_CHARS = 180
_MIN_SENTENCE_CHARS = 25

# Inverted-pyramid position boosts
_POSITION_BOOST = {
    0: 1.85,  # Lead sentence (Who, What, Where, When)
    1: 1.45,  # Direct follow-up / key impact
    2: 1.25,  # Supporting details
    3: 1.10,  # Context
}


def is_khmer_text(text: str) -> bool:
    """Check if text contains Khmer characters."""
    return bool(_KHMER_CHAR_RE.search(text))


def split_sentences(text: str) -> List[str]:
    """Split article text into clean, individual sentence candidates.
    
    Khmer text:
      - Primary: split on Khmer periods (។), question marks, exclamations, and newlines.
      - Secondary: split overly long run-on sentences (>220 chars) at natural Khmer discourse conjunctions.
    English / Latin text:
      - Primary: NLTK punkt tokenizer (or regex punctuation split fallback).
    """
    if not text:
        return []

    is_km = is_khmer_text(text)
    sentences: list[str] = []

    if is_km:
        # Split on Khmer punctuation or paragraph breaks
        raw_parts = re.split(r"[។!?\n\r]+", text)
        for part in raw_parts:
            part = part.strip()
            if not part:
                continue
            # If a run-on Khmer sentence has no punctuation and is >220 chars, split on discourse markers
            if len(part) > 220:
                clauses = _KHMER_CLAUSE_SPLIT.split(part)
                for cl in clauses:
                    cl = cl.strip()
                    if cl:
                        sentences.append(cl)
            else:
                sentences.append(part)
    else:
        # Latin text
        if _HAS_NLTK and _nltk_sent_tokenize:
            try:
                raw_parts = _nltk_sent_tokenize(text)
            except Exception:
                raw_parts = re.split(r"(?<=[.!?])\s+|\n+", text)
        else:
            raw_parts = re.split(r"(?<=[.!?])\s+|\n+", text)

        for part in raw_parts:
            part = part.strip()
            if part:
                sentences.append(part)

    # Clean internal duplicate whitespace
    return [re.sub(r"\s+", " ", s).strip() for s in sentences]


def is_boilerplate(sentence: str) -> bool:
    """Filter out non-informative sentences, ads, links, and trivial fragments."""
    if len(sentence) < _MIN_SENTENCE_CHARS or len(sentence) > 400:
        return True
    return bool(_BOILERPLATE_RE.search(sentence))


def extract_features(text: str) -> List[str]:
    """Extract informative feature tokens from mixed Khmer & Latin text.
    
    - Latin: Alphanumeric words, acronyms, technical IDs (CVEs, 5G, AI, iOS).
    - Khmer: Character 3-grams that represent morphological word stems.
    """
    features: list[str] = []

    # Latin tokens
    for match in re.finditer(r"[a-zA-Z0-9_\-\.]{2,}", text.lower()):
        tok = match.group(0).strip(".-_")
        if tok and tok not in _STOP_WORDS and len(tok) >= 2:
            features.append(tok)

    # Khmer shingle blocks (3-grams)
    khmer_blocks = re.findall(r"[\u1780-\u17ff]+", text)
    for block in khmer_blocks:
        if len(block) <= 4:
            features.append(block)
        else:
            for i in range(len(block) - 2):
                features.append(block[i : i + 3])

    return features


def compute_word_frequencies(text: str) -> dict[str, float]:
    """Compute log-damped TF frequencies for informative features."""
    features = extract_features(text)
    if not features:
        return {}
    counts = Counter(features)
    max_c = max(counts.values())
    max_log = math.log1p(max_c)
    return {w: math.log1p(c) / max_log for w, c in counts.items()}


def sentence_vector(sentence: str, freq_table: dict[str, float]) -> dict[str, float]:
    """Build sparse TF vector for a sentence to compute cosine similarity."""
    features = extract_features(sentence)
    counts = Counter(features)
    return {w: c * freq_table[w] for w, c in counts.items() if w in freq_table}


def cosine_similarity(vec1: dict[str, float], vec2: dict[str, float]) -> float:
    """Compute cosine similarity between two sparse feature vectors."""
    if not vec1 or not vec2:
        return 0.0
    dot = sum(vec1.get(k, 0.0) * v for k, v in vec2.items())
    norm1 = math.sqrt(sum(v * v for v in vec1.values()))
    norm2 = math.sqrt(sum(v * v for v in vec2.values()))
    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0
    return dot / (norm1 * norm2)


def extractive_summary(
    text: str,
    sentences_count: int = 4,
    max_chars_per_sentence: int = _MAX_SENTENCE_CHARS,
) -> str:
    """Generate high-density, informative news summary bullet points.
    
    Optimizations:
      - Khmer & English sentence boundary intelligence
      - Multi-language boilerplate and junk stripping
      - Hybrid TF-IDF scoring (Latin words + Khmer 3-gram character stems)
      - Inverted pyramid position boost
      - Information density bonus (numbers, percentages, dates, CVEs, quotes)
      - Cosine-similarity deduplication
      - Clean Unicode Khmer subscript-safe truncation
    """
    if not text or len(text.strip()) < _MIN_SENTENCE_CHARS:
        return text or ""

    try:
        # 1. Split into sentence candidates and filter boilerplate
        raw_sents = split_sentences(text)
        sentences = [s for s in raw_sents if not is_boilerplate(s)]

        if not sentences:
            sentences = [s for s in raw_sents if len(s) >= _MIN_SENTENCE_CHARS][:sentences_count]

        if len(sentences) <= sentences_count:
            return format_summary_output(sentences, max_chars_per_sentence)

        # 2. Build global document frequency table
        freq = compute_word_frequencies(text)
        if not freq:
            return format_summary_output(sentences[:sentences_count], max_chars_per_sentence)

        # 3. Score candidates with position and information density bonuses
        scored = []
        for idx, sent in enumerate(sentences):
            feats = extract_features(sent)
            if not feats:
                continue

            raw_score = sum(freq.get(f, 0.0) for f in feats)
            # Normalize by sqrt(length) to avoid biasing towards excessively long run-ons
            norm_score = raw_score / math.sqrt(len(feats))

            # Inverted pyramid position boost (prioritize opening facts)
            norm_score *= _POSITION_BOOST.get(idx, 1.0)

            # Information density bonus: numbers, dates, statistics, CVEs, quotes
            if re.search(r"\d+|[$€%]|CVE-|\"[^\"]+\"", sent):
                norm_score *= 1.15

            scored.append((idx, norm_score, sent))

        if not scored:
            return format_summary_output(sentences[:sentences_count], max_chars_per_sentence)

        # 4. Greedy selection with cosine deduplication
        scored.sort(key=lambda item: item[1], reverse=True)
        selected_indices: list[int] = []
        selected_vectors: list[dict[str, float]] = []
        sim_threshold = 0.50

        # Always include sentence 0 if it scored reasonably well (Lead sentence heuristic)
        if scored and scored[0][0] != 0:
            lead_cand = next((item for item in scored if item[0] == 0), None)
            if lead_cand and lead_cand[1] >= scored[0][1] * 0.60:
                selected_indices.append(0)
                selected_vectors.append(sentence_vector(lead_cand[2], freq))

        for idx, score, sent in scored:
            if len(selected_indices) >= sentences_count:
                break
            if idx in selected_indices:
                continue

            vec = sentence_vector(sent, freq)
            if any(cosine_similarity(vec, sv) >= sim_threshold for sv in selected_vectors):
                continue  # Skip redundant sentence covering the same facts

            selected_indices.append(idx)
            selected_vectors.append(vec)

        # 5. Restore chronological narrative order
        selected_indices.sort()
        selected_sentences = [sentences[i] for i in selected_indices]

        return format_summary_output(selected_sentences, max_chars_per_sentence)

    except Exception as exc:
        logger.warning("extractive_summary error: %s", exc)
        return text[:500] + "..."


def format_summary_output(sentences: List[str], max_chars: int) -> str:
    """Format and truncate sentences safely for Telegram bulletin bullets.
    
    Joins with ' ||| ' delimiter consumed by formatter.py.
    """
    result: list[str] = []
    for s in sentences:
        s = s.strip()
        if not s:
            continue

        # Strip trailing dangling punctuation
        s = s.rstrip(" :;,-–—")

        if len(s) > max_chars:
            if not is_khmer_text(s):
                # Latin text: cut at last word boundary
                cut = s[:max_chars].rsplit(" ", 1)[0]
                s = cut.rstrip(".,;: ") + "..."
            else:
                # Khmer text: cut safely without severing Khmer subscript / vowel signs
                cut = s[:max_chars]
                if " " in cut[-20:]:
                    cut = cut.rsplit(" ", 1)[0]
                # Avoid ending on Khmer combining vowels (\u17B4-\u17D3) or subscript coeng (\u17D2)
                while cut and ("\u17B4" <= cut[-1] <= "\u17D3" or "\u17DD" <= cut[-1] <= "\u17E9"):
                    cut = cut[:-1]
                s = cut.rstrip(" .,;:") + "..."

        result.append(s)

    return " ||| ".join(result)


__all__ = [
    "extractive_summary",
    "format_summary_output",
    "is_boilerplate",
    "is_khmer_text",
    "split_sentences",
]