"""Utility helpers for the AI Research Assistant.

All helpers are intentionally small, testable, and defensive.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List

from dotenv import load_dotenv


# Ensure we load the .env from the project root relative to this file's location
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV_PATH = os.path.join(BASE_DIR, ".env")
load_dotenv(dotenv_path=ENV_PATH, override=True)


def _sanitize_json_like(text: str) -> str:
    """Best-effort cleanup for JSON-like model outputs."""
    if not text:
        return text
    cleaned = text.strip()
    # Trim to first JSON object/array boundaries if present.
    obj_start = cleaned.find("{")
    arr_start = cleaned.find("[")
    start = min([i for i in [obj_start, arr_start] if i != -1], default=0)
    end_obj = cleaned.rfind("}")
    end_arr = cleaned.rfind("]")
    end = max(end_obj, end_arr)
    if end > start:
        cleaned = cleaned[start : end + 1]

    # Replace smart quotes.
    cleaned = cleaned.replace("“", "\"").replace("”", "\"").replace("’", "'")

    # Remove trailing commas before closing braces/brackets.
    cleaned = re.sub(r",\s*(\}|\])", r"\1", cleaned)

    # If it looks like single-quoted JSON, normalize quotes.
    if '"' not in cleaned and "'" in cleaned:
        cleaned = cleaned.replace("'", "\"")

    return cleaned
def load_env_var(name: str, default: str | None = None) -> str | None:
    """Load an environment variable with a default fallback.

    Args:
        name: Environment variable name.
        default: Default value if not set.

    Returns:
        The value if present, otherwise default.
    """
    value = os.getenv(name, default)
    return value


def safe_json_loads(text: Any) -> Any:
    """Attempt to parse JSON from model output with basic cleanup.

    The model might include leading/trailing text or code fences. This function
    tries to recover the first JSON object it can find.
    """
    # If an AIMessage or similar object is passed, extract .content
    if hasattr(text, "content"):
        text = getattr(text, "content")

    if not text:
        return {"error": "Empty response from model."}

    # Remove code fences if present.
    cleaned = re.sub(r"```(json)?", "", text, flags=re.IGNORECASE).strip()

    # Try direct JSON parse first.
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Handle JSON inside markdown code fences.
    fenced = re.search(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", cleaned, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        try:
            return json.loads(fenced.group(1))
        except json.JSONDecodeError:
            pass

    # Try sanitized JSON-like content.
    sanitized = _sanitize_json_like(cleaned)
    try:
        return json.loads(sanitized)
    except json.JSONDecodeError:
        pass

    # Fallback: find the first JSON object or array substring.
    obj_match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    arr_match = re.search(r"\[.*\]", cleaned, flags=re.DOTALL)
    match = obj_match or arr_match
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return {"error": "Failed to parse JSON from model output.", "raw": text}

    return {"error": "No JSON object found in model output.", "raw": text}


def truncate_text(text: str, max_chars: int = 12000) -> str:
    """Truncate long text to keep prompts bounded for API limits."""
    if not text:
        return ""
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 200] + "\n\n[TRUNCATED]"


def clean_authors(authors: List[str]) -> List[str]:
    """Normalize author list to a small, readable set."""
    if not authors:
        return []
    # Keep at most 6 authors for concise references.
    return [a.strip() for a in authors[:6] if a.strip()]


def authors_to_str(authors: Any) -> str:
    """Normalize authors to a clean comma-separated string."""
    if not authors:
        return ""
    if isinstance(authors, str):
        return authors.strip()
    if isinstance(authors, list):
        return ", ".join([a.strip() for a in authors if isinstance(a, str) and a.strip()])
    return str(authors).strip()


def extract_datasets(text: str) -> List[str]:
    """Heuristic dataset extraction from text."""
    if not text:
        return []
    candidates = [
        "ImageNet",
        "CIFAR-10",
        "CIFAR-100",
        "MNIST",
        "COCO",
        "SQuAD",
        "GLUE",
        "SuperGLUE",
        "WikiText",
        "IMDB",
        "WMT",
        "LibriSpeech",
        "Common Crawl",
    ]
    found = [c for c in candidates if c.lower() in text.lower()]
    return list(dict.fromkeys(found))


def extract_models(text: str) -> List[str]:
    """Heuristic model extraction from text."""
    if not text:
        return []
    candidates = [
        "BERT",
        "RoBERTa",
        "GPT",
        "GPT-2",
        "GPT-3",
        "T5",
        "LLaMA",
        "Mistral",
        "Transformer",
        "CNN",
        "RNN",
        "LSTM",
        "GRU",
        "XGBoost",
        "Random Forest",
        "SVM",
    ]
    found = [c for c in candidates if c.lower() in text.lower()]
    return list(dict.fromkeys(found))


def extract_proposed_approach(text: str) -> str:
    """Heuristic extraction of proposed approach from abstract."""
    if not text:
        return ""
    lowered = text.lower()
    cues = ["we propose", "we present", "we introduce", "this paper proposes", "this paper presents"]
    for cue in cues:
        idx = lowered.find(cue)
        if idx != -1:
            snippet = text[idx : idx + 220]
            return snippet.strip()
    # Fallback: first sentence
    match = re.split(r"(?<=[.!?])\s+", text.strip())
    return match[0].strip() if match else ""


def ensure_directory(path: str) -> None:
    """Ensure a directory exists; create if missing."""
    os.makedirs(path, exist_ok=True)


def safe_get(dct: Dict[str, Any], key: str, default: Any = "") -> Any:
    """Safe get with default for dicts."""
    if not isinstance(dct, dict):
        return default
    return dct.get(key, default)


def append_chat_log_entry(entry: Dict[str, Any], path: str | None = None) -> None:
    """Append a single chat interaction entry to a JSONL file."""
    if not isinstance(entry, dict):
        return
    log_path = path or load_env_var("CHAT_LOG_PATH", "data/chat_history.jsonl") or "data/chat_history.jsonl"
    directory = os.path.dirname(log_path) or "."
    try:
        os.makedirs(directory, exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as handle:
            json.dump(entry, handle, ensure_ascii=False)
            handle.write("\n")
    except Exception:
        # Logging should never break the assistant flow.
        pass


def strip_html(text: Any) -> str:
    """Remove HTML tags and special formatting characters (markdown artifacts)."""
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    # Remove tags like <h4>...</h4>
    cleaned = re.sub(r"<[^>]+>", "", text)
    # Remove common markdown special characters used for formatting (asterisks, underscores, backticks, hashes)
    cleaned = re.sub(r"[\*\_\#`~]", "", cleaned)
    # Collapse extra whitespace and newlines
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned
