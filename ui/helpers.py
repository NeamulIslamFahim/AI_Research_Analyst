"""Small utility helpers used across the Streamlit UI."""

from __future__ import annotations

import re
import tempfile
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlparse


def format_chat_history(messages: list[dict[str, Any]], max_messages: int = 100) -> str:
    """Convert recent chat messages into a plain text history string."""
    trimmed_messages = messages[-max_messages:]
    lines = []
    for msg in trimmed_messages:
        label = "User" if msg["role"] == "user" else "Assistant"
        content_raw = msg.get("content") or ""
        extra_lines: list[str] = []
        if isinstance(content_raw, dict):
            content = content_raw.get("answer") or content_raw.get("assistant_reply") or ""
            sources = content_raw.get("sources") or []
            if isinstance(sources, list) and sources:
                titles = [str(item.get("title", "")).strip() for item in sources if isinstance(item, dict) and item.get("title")]
                if titles:
                    extra_lines.append("Sources: " + " | ".join(titles[:8]))
            table = content_raw.get("table") or []
            if isinstance(table, list) and table:
                titles = [str(item.get("paper_name", "")).strip() for item in table if isinstance(item, dict) and item.get("paper_name")]
                if titles:
                    extra_lines.append("Papers: " + " | ".join(titles[:8]))
        else:
            content = msg.get("effective_query") or msg.get("display_text") or str(content_raw)
        lines.append(f"{label}: {content}")
        lines.extend(extra_lines)
    return "\n".join(lines)


def save_uploaded_pdf(uploaded_file: Any) -> str:
    """Store an uploaded Streamlit file in a temporary PDF path."""
    suffix = Path(uploaded_file.name).suffix or ".pdf"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getvalue())
        return tmp.name


def safe_paper_url(url: str, title: str = "") -> str:
    """Repair paper links or fall back to a scholar search."""
    raw = (url or "").strip()
    if not raw or raw.lower() in {"not specified", "none", "null"}:
        return f"https://scholar.google.com/scholar?q={quote(title)}" if title else ""

    cleaned = raw.replace("https://doi.org/https://doi.org/", "https://doi.org/").strip()
    cleaned = re.sub(
        r"^https?://arxiv\.org/abs/https?://arxiv\.org/abs/",
        "https://arxiv.org/abs/",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"^https?://arxiv\.org/pdf/https?://arxiv\.org/pdf/",
        "https://arxiv.org/pdf/",
        cleaned,
        flags=re.IGNORECASE,
    )
    if cleaned.startswith("doi:"):
        cleaned = f"https://doi.org/{cleaned[4:].strip()}"
    elif cleaned.startswith("doi.org/"):
        cleaned = f"https://{cleaned}"
    elif cleaned.startswith("10.") and "/" in cleaned:
        cleaned = f"https://doi.org/{cleaned}"
    elif cleaned.startswith("arxiv.org/"):
        cleaned = f"https://{cleaned}"
    elif not cleaned.startswith(("http://", "https://")):
        return f"https://scholar.google.com/scholar?q={quote(title)}" if title else ""

    cleaned = cleaned.replace(" ", "")
    parsed = urlparse(cleaned)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return f"https://scholar.google.com/scholar?q={quote(title)}" if title else ""

    if "doi.org" in parsed.netloc and "/" not in parsed.path.strip("/"):
        return f"https://scholar.google.com/scholar?q={quote(title)}" if title else ""

    return cleaned
