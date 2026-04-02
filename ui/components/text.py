"""Text formatting helpers for Streamlit UI components."""

from __future__ import annotations

from typing import Any


class TextPreviewer:
    @staticmethod
    def preview(value: Any, max_chars: int = 260) -> str:
        if value is None:
            return ""
        text = value.get("answer") if isinstance(value, dict) else value
        if isinstance(text, dict):
            text = text.get("assistant_reply") or text.get("answer") or ""
        text = str(text)
        replacements = {
            "ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â": "-",
            "ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â€šÂ¬Ã…â€œ": "-",
            "ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¹Ã…â€œ": "'",
            "ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢": "'",
            "ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã¢â‚¬Å“": '"',
            "ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â": '"',
            "ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â¦": "...",
        }
        for bad, good in replacements.items():
            text = text.replace(bad, good)
        text = " ".join(text.split()).strip()
        if len(text) > max_chars:
            text = text[: max_chars - 3].rstrip() + "..."
        return text

