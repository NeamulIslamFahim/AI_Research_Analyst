"""Shared text utilities for research retrieval and response formatting."""

from __future__ import annotations

import re
from typing import Any

from backend.helpers import strip_html


GENERIC_STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "from",
    "that",
    "this",
    "into",
    "in",
    "on",
    "of",
    "to",
    "a",
    "an",
    "is",
    "are",
    "i",
    "me",
    "my",
    "mine",
    "you",
    "your",
    "yours",
    "we",
    "us",
    "our",
    "ours",
    "need",
    "want",
    "please",
    "could",
    "would",
    "like",
    "means",
    "paper",
    "research",
    "using",
    "approach",
    "results",
    "study",
    "analysis",
    "ai",
    "artificial",
    "intelligence",
}

TITLE_STOPWORDS = GENERIC_STOPWORDS | {
    "based",
    "toward",
    "towards",
    "via",
    "paper",
    "study",
    "model",
    "models",
    "method",
    "methods",
    "framework",
    "frameworks",
    "system",
    "systems",
    "review",
    "survey",
}


def clean_text(value: Any) -> str:
    return strip_html(value or "").replace("\n", " ").strip()


def title_key(value: Any) -> str:
    cleaned = clean_text(value).lower()
    cleaned = re.sub(r"[^a-z0-9]+", " ", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


def title_tokens(value: Any) -> list[str]:
    return [
        token
        for token in title_key(value).split()
        if len(token) >= 3 and token not in TITLE_STOPWORDS
    ]


def titles_look_equivalent(left: Any, right: Any) -> bool:
    left_key = title_key(left)
    right_key = title_key(right)
    if not left_key or not right_key:
        return False
    if left_key == right_key:
        return True

    left_tokens = set(title_tokens(left))
    right_tokens = set(title_tokens(right))
    overlap = left_tokens & right_tokens
    min_size = min(len(left_tokens), len(right_tokens))

    shorter, longer = (left_key, right_key) if len(left_key) <= len(right_key) else (right_key, left_key)
    if len(shorter) >= 24 and shorter in longer:
        if min_size == 0:
            return True
        if len(overlap) / max(1, min_size) >= 0.75:
            return True

    if min_size == 0:
        return False
    if min_size <= 3:
        return min_size >= 2 and len(overlap) == min_size
    return len(overlap) >= 4 and (len(overlap) / min_size) >= 0.8


def collapse_text(text: str, max_chars: int) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > max_chars:
        return text[: max_chars - 3].rstrip() + "..."
    return text


def topic_tokens(topic: str) -> list[str]:
    parts = [t for t in topic.lower().replace("-", " ").split() if len(t) >= 2 and t not in GENERIC_STOPWORDS]
    return list(dict.fromkeys(parts))


def strong_topic_tokens(topic: str) -> list[str]:
    return [t for t in topic_tokens(topic) if len(t) >= 4]


def topic_is_specific(topic: str) -> bool:
    return len(strong_topic_tokens(topic)) >= 2


def row_matches_topic(row: dict, topic: str) -> bool:
    tokens = strong_topic_tokens(topic) or topic_tokens(topic)
    if not tokens:
        return True
    hay = " ".join(
        [
            str(row.get("title", "")),
            clean_text(row.get("abstract", "")),
            str(row.get("authors", "")),
        ]
    ).lower()
    if topic_is_specific(topic):
        hits = sum(1 for tok in tokens if tok in hay)
        return hits >= 2
    return any(tok in hay for tok in tokens)


def strip_front_matter(text: str, title: str = "") -> str:
    cleaned = clean_text(text)
    if not cleaned:
        return ""

    if title:
        title_norm = re.sub(r"\s+", " ", title).strip().lower()
        cleaned_norm = re.sub(r"\s+", " ", cleaned).strip().lower()
        if cleaned_norm.startswith(title_norm):
            cleaned = cleaned[len(title):].lstrip(" :-\u00e2\u20ac\u201c\u00e2\u20ac\u201d\t\r\n")

    markers = [
        r"\babstract\b[:\s\-\u00e2\u20ac\u201c\u00e2\u20ac\u201d]*",
        r"\bintroduction\b[:\s\-\u00e2\u20ac\u201c\u00e2\u20ac\u201d]*",
        r"\b1\.\s*introduction\b[:\s\-\u00e2\u20ac\u201c\u00e2\u20ac\u201d]*",
    ]
    for marker in markers:
        match = re.search(marker, cleaned, flags=re.IGNORECASE)
        if match:
            cleaned = cleaned[match.end():].strip()
            break

    cleaned = re.sub(r"^(arxiv|doi|conference|proceedings)\b.*?$", "", cleaned, flags=re.IGNORECASE | re.MULTILINE).strip()
    return cleaned


def sentence_snippets(text: str, limit: int = 2) -> list[str]:
    pieces = re.split(r"(?<=[.!?])\s+", text.strip())
    snippets: list[str] = []
    for piece in pieces:
        piece = piece.strip()
        if len(piece.split()) < 6:
            continue
        snippets.append(piece)
        if len(snippets) >= limit:
            break
    return snippets


def _unique_sentences(sentences: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for sentence in sentences:
        key = re.sub(r"\s+", " ", sentence).strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        unique.append(sentence)
    return unique


def _paper_snippets(text: str, limit: int = 4, min_words: int = 4, min_chars: int = 24) -> list[str]:
    pieces = re.split(r"(?<=[.!?])\s+", text.strip())
    snippets: list[str] = []
    for piece in pieces:
        piece = piece.strip()
        if len(piece) < min_chars:
            continue
        if len(piece.split()) < min_words:
            continue
        snippets.append(piece)
        if len(snippets) >= limit:
            break
    return snippets


def _paper_fragments(text: str, limit: int = 6, min_words: int = 5, min_chars: int = 32) -> list[str]:
    pieces = re.split(r"(?<=[.!?])\s+|\n+|;\s+|:\s+", text.strip())
    fragments: list[str] = []
    for piece in pieces:
        piece = piece.strip(" -\t\r\n")
        if len(piece) < min_chars:
            continue
        if len(piece.split()) < min_words:
            continue
        fragments.append(piece)
        if len(fragments) >= limit:
            break
    return fragments


def _clause_fragments(text: str, limit: int = 8, min_words: int = 4, min_chars: int = 24) -> list[str]:
    pieces = re.split(r",\s+|\band\b\s+(?=[a-z])|\bwhich\b\s+(?=[a-z])|\bthat\b\s+(?=[a-z])", text.strip(), flags=re.IGNORECASE)
    fragments: list[str] = []
    for piece in pieces:
        piece = piece.strip(" -\t\r\n,")
        if len(piece) < min_chars:
            continue
        if len(piece.split()) < min_words:
            continue
        fragments.append(piece)
        if len(fragments) >= limit:
            break
    return fragments


def human_summary_from_text(text: str, title: str = "", max_chars: int = 380) -> str:
    body = strip_front_matter(text, title)
    if not body:
        return ""
    # Larger output fields should read like a compact paper overview, not just the opening lines.
    snippet_limit = 4 if max_chars >= 900 else 2
    snippets = sentence_snippets(body, limit=snippet_limit)
    summary = " ".join(snippets) if snippets else body
    summary = re.sub(r"\s+", " ", summary).strip()
    if title:
        title_clean = re.escape(clean_text(title))
        summary = re.sub(rf"^{title_clean}\s*[-:\u00e2\u20ac\u201c\u00e2\u20ac\u201d]?\s*", "", summary, flags=re.IGNORECASE)
    return collapse_text(summary, max_chars)


def full_paper_summary_from_text(text: str, title: str = "", max_chars: int = 1600) -> str:
    """Summarize a PDF-style paper body using both early and later sections."""
    body = strip_front_matter(text, title)
    if not body:
        return ""

    third = max(1, len(body) // 3)
    early = _paper_snippets(body[: third], limit=6, min_words=3, min_chars=18)
    middle = _paper_snippets(body[third : 2 * third], limit=6, min_words=3, min_chars=18)
    late = _paper_snippets(body[2 * third :], limit=6, min_words=3, min_chars=18)
    summary_parts = _unique_sentences(early + middle + late)

    if not summary_parts:
        summary_parts = _paper_fragments(body, limit=8, min_words=3, min_chars=18)

    if len(summary_parts) < 3:
        summary_parts = _unique_sentences(summary_parts + _paper_fragments(body, limit=10, min_words=2, min_chars=14))
    if len(summary_parts) < 3:
        summary_parts = _unique_sentences(summary_parts + _clause_fragments(body, limit=10, min_words=3, min_chars=18))
    if len(summary_parts) < 3:
        # As a last resort, keep the summary readable by duplicating the best available
        # paper fragments only when the extractor yields too little text.
        summary_parts = summary_parts + summary_parts[: max(0, 3 - len(summary_parts))]

    summary_parts = summary_parts[:5]
    summary_parts = [s if s.endswith((".", "!", "?")) else f"{s}." for s in summary_parts]

    summary = " ".join(summary_parts).strip()
    summary = re.sub(r"\s+", " ", summary).strip()
    if title:
        title_clean = re.escape(clean_text(title))
        summary = re.sub(rf"^{title_clean}\s*[-:\u00e2\u20ac\u201c\u00e2\u20ac\u201d]?\s*", "", summary, flags=re.IGNORECASE)
    return collapse_text(summary or body, max_chars)


def normalize_output_text(value: Any, max_chars: int | None = None) -> str:
    text = clean_text(value)
    if not text:
        return ""
    replacements = {
        "Ã¢â‚¬â€": "-",
        "Ã¢â‚¬â€œ": "-",
        "Ã¢â‚¬Ëœ": "'",
        "Ã¢â‚¬â„¢": "'",
        "Ã¢â‚¬Å“": '"',
        "Ã¢â‚¬Â": '"',
        "Ã¢â‚¬Â¦": "...",
        "ÃƒÂ©": "é",
        "ÃƒÂ¨": "è",
        "Ãƒ ": "à",
        "ÃƒÂ±": "ñ",
        "ÃƒÂ¼": "ü",
        "ÃƒÂ¶": "ö",
    }
    for bad, good in replacements.items():
        text = text.replace(bad, good)
    text = re.sub(r"\s+", " ", text).strip()
    if max_chars and len(text) > max_chars:
        text = text[: max_chars - 3].rstrip() + "..."
    return text
