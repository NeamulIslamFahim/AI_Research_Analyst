"""Shared validation and scoring helpers for backend workflows."""

from __future__ import annotations

import re
from typing import Any, Dict, Optional

from pydantic import ValidationError


def validate_research_result(result: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(result, dict):
        return {"error": "Invalid research result."}
    if result.get("error"):
        return result
    result.setdefault("table", [])
    if not result.get("research_gaps"):
        result["research_gaps"] = []
    if not result.get("assistant_reply"):
        result["assistant_reply"] = "Research summary prepared."
    if "generated_idea" not in result:
        result["generated_idea"] = "Not provided."
    if "generated_idea_steps" not in result:
        result["generated_idea_steps"] = []
    return result


def validate_review_result(result: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(result, dict):
        return {"error": "Invalid review result."}
    if result.get("error"):
        return result
    for key in [
        "strengths",
        "weaknesses",
        "novelty",
        "technical_correctness",
        "reproducibility",
        "recommendation",
        "suggested_venue",
    ]:
        result.setdefault(key, "Not provided.")
    return result


def validate_reference_result(result: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(result, dict):
        return {"error": "Invalid references result."}
    if result.get("error"):
        return result
    refs = result.get("references", [])
    if not isinstance(refs, list):
        result["references"] = []
    return result


def validate_qa_result(result: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(result, dict):
        return {"error": "Invalid QA result."}
    if result.get("error"):
        return result
    result.setdefault("answer", "No answer found.")
    return result


def score_research_result(result: Dict[str, Any], topic: str) -> Dict[str, Any]:
    if not isinstance(result, dict):
        return result
    table = result.get("table")
    if not isinstance(table, list):
        return result
    topic_words = [w for w in (topic or "").lower().split() if len(w) > 2]
    for row in table:
        if not isinstance(row, dict):
            continue
        text = f"{row.get('paper_name','')} {row.get('summary_full_paper','')}".lower()
        overlap = sum(1 for w in topic_words if w in text)
        max_overlap = max(len(topic_words), 1)
        relevance = int(min(10, round((overlap / max_overlap) * 10)))
        quality = 0
        if row.get("paper_url"):
            quality += 3
        if row.get("authors_name"):
            quality += 2
        if row.get("summary_full_paper"):
            quality += 3
        if row.get("problem_solved"):
            quality += 1
        if row.get("proposed_model_or_approach"):
            quality += 1
        quality = min(10, quality)
        row["score_relevance"] = relevance
        row["score_quality"] = quality
    return result


def strict_validate(schema, result: Dict[str, Any]) -> tuple[bool, Optional[str]]:
    try:
        schema.model_validate(result)
        return True, None
    except ValidationError as exc:
        return False, str(exc)


def normalize_url(url: str) -> str:
    if not url:
        return ""
    trimmed = str(url).strip().replace(" ", "")
    trimmed = trimmed.replace("https://doi.org/https://doi.org/", "https://doi.org/")
    trimmed = trimmed.replace("http://doi.org/", "https://doi.org/")
    if trimmed.startswith("http://") or trimmed.startswith("https://"):
        return trimmed
    if trimmed.startswith("doi.org/"):
        return f"https://{trimmed}"
    if trimmed.startswith("doi:"):
        doi = trimmed.replace("doi:", "").strip()
        return f"https://doi.org/{doi}"
    if trimmed.startswith("10."):
        return f"https://doi.org/{trimmed}"
    if trimmed.startswith("arxiv.org/"):
        return f"https://{trimmed}"
    return f"https://{trimmed}"


def fix_paper_url(url: str, title: str = "") -> str:
    """Fix common invalid paper URLs (e.g., arXiv IDs wrongly formatted as DOIs)."""
    if not url:
        return ""
    trimmed = normalize_url(url)
    trimmed = re.sub(
        r"^https?://arxiv\.org/abs/https?://arxiv\.org/abs/",
        "https://arxiv.org/abs/",
        trimmed,
        flags=re.IGNORECASE,
    )
    trimmed = re.sub(r"^https?://arxiv\.org/abs/https?://arxiv\.org/pdf/", "https://arxiv.org/pdf/", trimmed, flags=re.IGNORECASE)
    if title and "arxiv" in trimmed.lower() and "/pdf/" not in trimmed.lower() and "/abs/" not in trimmed.lower():
        arxiv_match = re.search(r"(\d{4}\.\d{4,5}(v\d+)?)", url)
        if arxiv_match:
            return f"https://arxiv.org/abs/{arxiv_match.group(1)}"
    return trimmed
