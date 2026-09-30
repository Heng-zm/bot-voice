"""Extractive news summarization with Khmer and multilingual sentence segmentation."""

from __future__ import annotations

from collections import defaultdict
import logging
import math
import re
from typing import Sequence

logger = logging.getLogger(__name__)

# Common English stop words
_COMMON_STOP_WORDS = frozenset({
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are",
    "aren't", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both",
    "but", "by", "can't", "cannot", "could", "couldn't", "did", "didn't", "do", "does", "doesn't",
    "doing", "don't", "down", "during", "each", "few", "for", "from", "further", "had", "hadn't",
    "has", "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here",
    "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll",
    "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its", "itself", "let's",
    "me", "more", "most", "mustn't", "my", "myself", "no", "nor", "not", "of", "off", "on",
    "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own",
    "same", "shan't", "she", "she'd", "she'll", "she's", "should", "shouldn't", "so", "some",
    "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then",
    "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this",
    "those", "through", "to", "too", "under", "until", "up", "very", "was", "wasn't", "we",
    "we'd", "we'll", "we're", "we've", "were", "weren't", "what", "what's", "when", "when's",
    "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with",
    "won't", "would", "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours",
})

# Boilerplate patterns in news articles (English & Cambodian media)
_BOILERPLATE_RE = re.compile(
    r"click here|subscribe|read more|follow us|all rights reserved|"
    r"photo by|image source|copyright \d{4}|sign up for|download our app|"
    r"advertisement|sponsored content|terms of service|privacy policy|"
    r"ចុច subscribe|ចុច like|ទាញយក app|ទាញយកកម្មវិធី|រក្សាសិទ្ធិគ្រប់យ៉ាង|"
    r"រក្សាសិទ្ធិដោយ|ផ្សាយពាណិជ្ជកម្ម|ទំនាក់ទំនងផ្សាយពាណិជ្ជកម្ម|"
    r"តេឡេក្រាម|join telegram|telegram channel|អានបន្ត|ចុចទីនេះ",
    re.IGNORECASE,
)

_KHMER_RE = re.compile(r"[\u1780-\u17FF]")
_KHMER_SENT_SEP = re.compile(r"[។៕!?？\n\r]+")
_LATIN_SENT_SEP = re.compile(r"(?<=[.!?])\s+|\n+")

# Khmer combining vowels and diacritics
_KHMER_DIACRITICS = set(
    "\u17b4\u17b5\u17b6\u17b7\u17b8\u17b9\u17ba\u17bb\u17bc\u17bd\u17be\u17bf"
    "\u17c0\u17c1\u17c2\u17c3\u17c4\u17c5\u17c6\u17c7\u17c8\u17c9\u17ca\u17cb"
    "\u17cc\u17cd\u17ce\u17cf\u17d0\u17d1\u17d2\u17d3\u17dd"
)

_MAX_SENTENCE_CHARS = 180
_MIN_SENTENCE_CHARS = 20
_POSITION_BOOST = {0: 1.75, 1: 1.4, 2: 1.2, 3: 1.1}


def is_khmer_text(text: str) -> bool:
    """Return True if text contains Khmer Unicode characters."""
    return bool(_KHMER_RE.search(str(text or "")))


def _clean_trailing_khmer(text: str) -> str:
    """Strip trailing incomplete Khmer vowels, diacritics, and punctuation marks."""
    t = text.rstrip(" .,;:!?-–—\n\r")
    while t and t[-1] in _KHMER_DIACRITICS:
        t = t[:-1]
    return t.rstrip(" .,;:!?-–—\n\r")


def _truncate_khmer_text(text: str, max_chars: int) -> str:
    """Linguistically safe truncation for Khmer text without slicing combining marks."""
    if len(text) <= max_chars:
        return text

    cut = text[:max_chars]
    # Check if there is a space in the last 30% of the slice
    space_idx = cut.rfind(" ", int(max_chars * 0.7), max_chars)
    if space_idx != -1:
        cut = cut[:space_idx]

    cleaned = _clean_trailing_khmer(cut)
    return f"{cleaned}…"


def split_sentences(text: str) -> list[str]:
    """Split text into sentences cleanly for both Khmer and other languages."""
    if not text:
        return []

    clean_text = str(text).strip()
    if is_khmer_text(clean_text):
        raw = _KHMER_SENT_SEP.split(clean_text)
    else:
        raw = _LATIN_SENT_SEP.split(clean_text)

    sentences: list[str] = []
    for s in raw:
        line = s.strip().replace("\r", "")
        if not line:
            continue

        # If a single sentence is excessively long without punctuation, split on double spaces or major clauses
        if is_khmer_text(line) and len(line) > 350:
            sub_clauses = re.split(r"\s{2,}|(?<=[^\s])\s+(?=(?:ខណៈដែល|ហើយ|ប៉ុន្តែ|ដោយឡែក|លើសពីនេះ|ព្រមទាំង)\s+)", line)
            for sub in sub_clauses:
                clean_sub = sub.strip()
                if len(clean_sub) >= _MIN_SENTENCE_CHARS:
                    sentences.append(clean_sub)
        else:
            sentences.append(line)

    return sentences


def is_boilerplate(sentence: str) -> bool:
    """Detect boilerplate phrases or navigational artifacts."""
    clean = sentence.strip()
    if len(clean) < _MIN_SENTENCE_CHARS or len(clean) > 800:
        return True
    return bool(_BOILERPLATE_RE.search(clean))


def _tokenize(text: str) -> list[str]:
    """Hybrid tokenizer: extracts Latin words and Khmer character n-grams."""
    t_lower = text.lower()
    latin_tokens = [w for w in re.findall(r"\b[a-zA-Z0-9]{3,}\b", t_lower) if w not in _COMMON_STOP_WORDS]

    if not is_khmer_text(text):
        return latin_tokens

    # Khmer sliding 3-gram character tokens for non-segmented script overlap
    khmer_only = re.sub(r"[^\u1780-\u17FF]", "", text)
    khmer_tokens: list[str] = []
    if len(khmer_only) >= 3:
        khmer_tokens = [khmer_only[i : i + 3] for i in range(len(khmer_only) - 2)]

    return latin_tokens + khmer_tokens


def _word_freq(text: str) -> dict[str, float]:
    """Calculate normalized TF-IDF style term frequencies."""
    tokens = _tokenize(text)
    if not tokens:
        return {}

    freq: dict[str, float] = defaultdict(float)
    for w in tokens:
        freq[w] += 1.0

    max_f = max(freq.values())
    return {w: math.log1p(c) / math.log1p(max_f) for w, c in freq.items()}


def _sentence_vector(sentence: str, freq: dict[str, float]) -> dict[str, float]:
    """Build sparse term-frequency vector for cosine similarity deduplication."""
    tokens = _tokenize(sentence)
    vec: dict[str, float] = defaultdict(float)
    for w in tokens:
        if w in freq:
            vec[w] += freq[w]
    return dict(vec)


def _cosine_sim(a: dict[str, float], b: dict[str, float]) -> float:
    """Calculate cosine similarity between two sentence vectors."""
    if not a or not b:
        return 0.0
    dot = sum(a.get(k, 0.0) * v for k, v in b.items())
    norm_a = math.sqrt(sum(v * v for v in a.values()))
    norm_b = math.sqrt(sum(v * v for v in b.values()))
    norm = norm_a * norm_b
    return dot / norm if norm > 1e-9 else 0.0


def _format_bullets(
    sentences: Sequence[str],
    max_chars: int,
    *,
    as_bullets: bool = True,
    delimiter: str = "\n",
) -> str:
    """Format and join sentences into bullets or custom delimiter."""
    result: list[str] = []
    for s in sentences:
        clean = s.strip()
        if not clean:
            continue
        if len(clean) > max_chars:
            clean = _truncate_khmer_text(clean, max_chars)

        if as_bullets:
            result.append(f"• {clean}")
        else:
            result.append(clean)

    return delimiter.join(result)


def extractive_summary_list(
    text: str,
    sentences_count: int = 4,
    max_chars_per_sentence: int = _MAX_SENTENCE_CHARS,
) -> list[str]:
    """Extract top salient sentences returning a clean list of strings."""
    if not text or len(text.strip()) < _MIN_SENTENCE_CHARS:
        return [text.strip()] if text and text.strip() else []

    try:
        raw_sents = split_sentences(text)
        sentences = [s for s in raw_sents if not is_boilerplate(s)]
        if not sentences:
            sentences = [s for s in raw_sents if len(s) >= _MIN_SENTENCE_CHARS][:sentences_count]

        if len(sentences) <= sentences_count:
            return [s[:max_chars_per_sentence] for s in sentences]

        freq = _word_freq(text)
        if not freq:
            return [s[:max_chars_per_sentence] for s in sentences[:sentences_count]]

        scored: list[tuple[int, float, str]] = []
        for i, sent in enumerate(sentences):
            tokens = _tokenize(sent)
            if not tokens:
                continue
            raw_score = sum(freq.get(w, 0.0) for w in tokens)
            norm_score = raw_score / math.sqrt(len(tokens))
            norm_score *= _POSITION_BOOST.get(i, 1.0)
            scored.append((i, norm_score, sent))

        if not scored:
            return [s[:max_chars_per_sentence] for s in sentences[:sentences_count]]

        scored.sort(key=lambda x: x[1], reverse=True)
        selected_indices: list[int] = []
        selected_vectors: list[dict[str, float]] = []
        sim_threshold = 0.55

        for idx, _score, sent in scored:
            if len(selected_indices) >= sentences_count:
                break
            vec = _sentence_vector(sent, freq)
            if any(_cosine_sim(vec, sv) >= sim_threshold for sv in selected_vectors):
                continue
            selected_indices.append(idx)
            selected_vectors.append(vec)

        selected_indices.sort()
        return [_truncate_khmer_text(sentences[i], max_chars_per_sentence) for i in selected_indices]

    except Exception as exc:
        logger.warning("extractive_summary_list failed: %s", exc)
        raw_sents = split_sentences(text)[:sentences_count]
        return [_truncate_khmer_text(s, max_chars_per_sentence) for s in raw_sents]


def extractive_summary(
    text: str,
    sentences_count: int = 4,
    max_chars_per_sentence: int = _MAX_SENTENCE_CHARS,
    *,
    delimiter: str = "\n",
    as_bullets: bool = True,
) -> str:
    """Perform extractive news summarization with Khmer sentence segmentation and dedup.

    Parameters:
        text: Article content.
        sentences_count: Number of salient sentences to pick.
        max_chars_per_sentence: Max length for each sentence bullet.
        delimiter: String joining sentences (default: '\\n', or ' ||| ' for formatter pipelines).
        as_bullets: If True, prefixes with '• '.
    """
    if not text or len(text.strip()) < _MIN_SENTENCE_CHARS:
        return text or ""

    try:
        sents = extractive_summary_list(
            text,
            sentences_count=sentences_count,
            max_chars_per_sentence=max_chars_per_sentence,
        )
        if not sents:
            return ""

        return _format_bullets(
            sents,
            max_chars=max_chars_per_sentence,
            as_bullets=as_bullets,
            delimiter=delimiter,
        )
    except Exception as exc:
        logger.warning("extractive_summary fallback: %s", exc)
        sents = split_sentences(text)[:sentences_count]
        return _format_bullets(sents, max_chars_per_sentence, as_bullets=as_bullets, delimiter=delimiter)


__all__ = [
    "extractive_summary",
    "extractive_summary_list",
    "is_boilerplate",
    "is_khmer_text",
    "split_sentences",
]