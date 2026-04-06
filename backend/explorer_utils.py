"""Reusable helper functions for Research Explorer response cleanup."""

from __future__ import annotations

import re
import requests
from typing import Any, Dict
from urllib.parse import urlparse

from .helpers import safe_get


def format_review_reply(review: Dict[str, Any]) -> str:
    """Convert structured review JSON into a readable text block."""
    if not isinstance(review, dict):
        return str(review)
    parts = [
        "Here is a structured peer review of the paper:",
        f"Strengths: {safe_get(review, 'strengths', '')}",
        f"Weaknesses: {safe_get(review, 'weaknesses', '')}",
        f"Novelty: {safe_get(review, 'novelty', '')}",
        f"Technical Correctness: {safe_get(review, 'technical_correctness', '')}",
        f"Reproducibility: {safe_get(review, 'reproducibility', '')}",
        f"Recommendation: {safe_get(review, 'recommendation', '')}",
        f"Suggested Venue: {safe_get(review, 'suggested_venue', '')}",
    ]
    return "\n\n".join([part for part in parts if part and not part.endswith(": ")])


def normalize_url(url: str) -> str:
    """Normalize DOI, arXiv, and partial URLs into a safer format."""
    if not url:
        return ""
    trimmed = str(url).strip().replace(" ", "")
    trimmed = trimmed.replace("https://doi.org/https://doi.org/", "https://doi.org/")
    trimmed = trimmed.replace("http://doi.org/", "https://doi.org/")
    if trimmed.startswith(("http://", "https://")):
        return trimmed
    if trimmed.startswith("doi.org/"):
        return f"https://{trimmed}"
    if trimmed.startswith("doi:"):
        return f"https://doi.org/{trimmed.replace('doi:', '').strip()}"
    if trimmed.startswith("10."):
        return f"https://doi.org/{trimmed}"
    if trimmed.startswith("arxiv.org/"):
        return f"https://{trimmed}"
    return f"https://{trimmed}"


def fix_paper_url(url: str, title: str = "") -> str:
    """Repair malformed paper URLs or fall back to Scholar search."""
    if not url or "not specified" in str(url).lower():
        return "https://scholar.google.com/scholar?q=" + requests.utils.quote(title)
    trimmed = normalize_url(url)
    trimmed = re.sub(
        r"^https?://arxiv\.org/abs/https?://arxiv\.org/abs/",
        "https://arxiv.org/abs/",
        trimmed,
        flags=re.IGNORECASE,
    )
    trimmed = re.sub(
        r"^https?://arxiv\.org/pdf/https?://arxiv\.org/pdf/",
        "https://arxiv.org/pdf/",
        trimmed,
        flags=re.IGNORECASE,
    )
    if "doi.org/" in trimmed:
        suffix = trimmed.split("doi.org/", 1)[1]
        if not suffix or "/" not in suffix:
            return f"https://scholar.google.com/scholar?q={requests.utils.quote(title)}" if title else ""
        if suffix.count(".") == 1 and suffix.replace("v", "").replace(".", "").isdigit():
            return f"https://arxiv.org/abs/{suffix}"
        arxiv_prefixes = ["hep-", "astro-", "cs.", "math.", "physics.", "stat."]
        if "/" in suffix and any(suffix.startswith(prefix) for prefix in arxiv_prefixes):
            return f"https://arxiv.org/abs/{suffix}"
    parsed = urlparse(trimmed)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return f"https://scholar.google.com/scholar?q={requests.utils.quote(title)}" if title else ""
    return trimmed


def fix_explorer_links(result: Dict[str, Any]) -> Dict[str, Any]:
    """Repair paper URLs inside the explorer result table."""
    if not isinstance(result, dict):
        return result
    table = result.get("table")
    if isinstance(table, list):
        for row in table:
            if isinstance(row, dict):
                row["paper_url"] = fix_paper_url(row.get("paper_url", ""), row.get("paper_name", ""))
    return result


def relevant_to_topic(result: Dict[str, Any], topic: str) -> bool:
    """Check whether a cached result still matches the requested topic."""
    # Filter out common academic stop words that cause false positives in cache matching
    stop_words = {"means", "paper", "research", "using", "approach", "results", "study", "analysis", "method"}
    tokens = [
        token for token in topic.lower().replace("-", " ").split() 
        if len(token) >= 1 and token not in stop_words
    ]
    if not tokens:
        # If the topic is entirely stop words, we can't reliably match context
        return True
    table = result.get("table") if isinstance(result, dict) else None
    if not isinstance(table, list):
        return True

    matched = 0
    total = 0
    for row in table:
        if not isinstance(row, dict):
            continue
        total += 1
        haystack = " ".join(
            [
                str(row.get("paper_name", "")),
                str(row.get("summary_full_paper", "")),
                str(row.get("authors_name", "")),
            ]
        ).lower()
        if any(token in haystack for token in tokens):
            matched += 1

    if total == 0:
        return True
    # Require a higher match threshold for cached results to prevent unrelated topic pollution
    return matched / total >= 0.7


def fallback_error_result(topic: str, detail: str = "") -> Dict[str, Any]:
    """Return a safe research response shape when the explorer pipeline fails."""
    message = (
        f"Research Explorer could not complete the full pipeline for '{topic}', "
        "so a fallback response is being returned instead."
    )
    if detail:
        message = f"{message}\n\nRecovery detail: {detail}"
    return {
        "table": [],
        "research_gaps": [],
        "assistant_reply": message,
        "generated_idea": (
            "Refine the topic slightly and retry, or broaden the scope to retrieve a wider evidence base before narrowing again."
        ),
        "generated_idea_steps": [
            "Try a slightly broader query with fewer specialized terms.",
            "Inspect the returned external sources and identify recurring themes.",
            "Retry with one focused subtopic after broader retrieval succeeds.",
        ],
        "generated_idea_citations": [],
        "error_recovered": True,
    }


def format_apa_reference(title: str, authors: list[str], year: str | int, url: str) -> str:
    """Format a simple APA reference string.

    This is a best-effort formatter for arXiv metadata.
    """
    if isinstance(year, int):
        year_str = str(year)
    else:
        year_str = year or "n.d."

    authors_str = ", ".join(authors) if authors else "Unknown Authors"
    title_str = title.rstrip(".")
    return f"{authors_str}. ({year_str}). {title_str}. {url}"
