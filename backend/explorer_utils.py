"""Reusable helper functions for Research Explorer response cleanup."""

from __future__ import annotations

import re
from typing import Any, Dict
from urllib.parse import quote, urlparse

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
    if not url:
        return f"https://scholar.google.com/scholar?q={quote(title)}" if title else ""
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
            return f"https://scholar.google.com/scholar?q={quote(title)}" if title else ""
        if suffix.count(".") == 1 and suffix.replace("v", "").replace(".", "").isdigit():
            return f"https://arxiv.org/abs/{suffix}"
        arxiv_prefixes = ["hep-", "astro-", "cs.", "math.", "physics.", "stat."]
        if "/" in suffix and any(suffix.startswith(prefix) for prefix in arxiv_prefixes):
            return f"https://arxiv.org/abs/{suffix}"
    parsed = urlparse(trimmed)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return f"https://scholar.google.com/scholar?q={quote(title)}" if title else ""
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
    tokens = [token for token in topic.lower().replace("-", " ").split() if len(token) > 2]
    if not tokens:
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
    return matched / total >= 0.5


def filter_result_by_topic(result: Dict[str, Any], topic: str) -> Dict[str, Any]:
    """Keep only rows that still look relevant to the current topic."""
    if not isinstance(result, dict):
        return result
    tokens = [token for token in topic.lower().replace("-", " ").split() if len(token) > 2]
    if not tokens:
        return result
    table = result.get("table")
    if not isinstance(table, list):
        return result

    filtered_rows = []
    kept_names = set()
    for row in table:
        if not isinstance(row, dict):
            continue
        haystack = " ".join(
            [
                str(row.get("paper_name", "")),
                str(row.get("summary_full_paper", "")),
                str(row.get("authors_name", "")),
            ]
        ).lower()
        if any(token in haystack for token in tokens):
            filtered_rows.append(row)
            if row.get("paper_name"):
                kept_names.add(row["paper_name"])

    result["table"] = filtered_rows
    if isinstance(result.get("research_gaps"), list):
        result["research_gaps"] = [
            gap for gap in result["research_gaps"] if any(name in gap for name in kept_names)
        ] or result["research_gaps"]
    if isinstance(result.get("generated_idea_citations"), list):
        result["generated_idea_citations"] = [
            citation for citation in result["generated_idea_citations"] if citation in kept_names
        ] or result["generated_idea_citations"]
    return result


def fallback_broader_result(result: Dict[str, Any], topic: str) -> Dict[str, Any]:
    """Use broader multi-source rows when strict topic filtering becomes too narrow."""
    if not isinstance(result, dict):
        return result
    broader = fix_explorer_links(result)
    table = broader.get("table")
    if isinstance(table, list):
        broader["table"] = table[:12]
    reply = (broader.get("assistant_reply") or "").strip()
    prefix = (
        f"No strongly matched papers were found for '{topic}'. "
        "Showing broader results gathered from multiple external sources instead."
    )
    broader["assistant_reply"] = f"{prefix}\n\n{reply}" if reply else prefix
    broader["used_broader_fallback"] = True
    return broader


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
        "used_broader_fallback": True,
        "error_recovered": True,
    }
