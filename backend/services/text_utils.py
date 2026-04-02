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


def clean_text(value: Any) -> str:
    return strip_html(value or "").replace("\n", " ").strip()


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


def human_summary_from_text(text: str, title: str = "", max_chars: int = 380) -> str:
    body = strip_front_matter(text, title)
    if not body:
        return ""
    snippets = sentence_snippets(body, limit=2)
    summary = " ".join(snippets) if snippets else body
    summary = re.sub(r"\s+", " ", summary).strip()
    if title:
        title_clean = re.escape(clean_text(title))
        summary = re.sub(rf"^{title_clean}\s*[-:\u00e2\u20ac\u201c\u00e2\u20ac\u201d]?\s*", "", summary, flags=re.IGNORECASE)
    return collapse_text(summary, max_chars)


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
