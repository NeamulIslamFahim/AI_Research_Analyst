"""Research Explorer service wrapper and topic helpers."""

from __future__ import annotations

from typing import Any, Dict, List, Optional


class ResearchService:
    @staticmethod
    def is_generic_explorer_prompt(text: str) -> bool:
        normalized = " ".join(str(text or "").strip().lower().split())
        return normalized in {
            "more",
            "next",
            "show more",
            "give me more",
            "continue",
            "elaborate",
            "tell me more",
            "more papers",
            "more results",
            "another",
            "others",
        }

    @classmethod
    def resolve_topic_from_history(cls, topic: str, chat_history: str | None) -> str:
        if not cls.is_generic_explorer_prompt(topic) or not chat_history:
            return topic
        lines = [line.strip() for line in str(chat_history).splitlines() if line.strip()]
        for line in reversed(lines):
            if line.lower().startswith("user:"):
                clean_user_input = line.split(":", 1)[1].strip() if ":" in line else line
                if clean_user_input and not cls.is_generic_explorer_prompt(clean_user_input):
                    return clean_user_input
        return topic

    def run(
        self,
        topic: str,
        chat_history: str | None = None,
        focus_topic: str | None = None,
        use_live: bool | None = None,
        previously_returned_titles: Optional[List[str]] = None,
        force_refresh: bool = False,
    ) -> Dict[str, Any]:
        from backend.main import _run_research_explorer_impl_legacy

        return _run_research_explorer_impl_legacy(
            topic=topic,
            chat_history=chat_history,
            focus_topic=focus_topic,
            use_live_sources=use_live,
            previously_returned_titles=previously_returned_titles,
        )
