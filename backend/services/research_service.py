"""Research Explorer service wrapper and topic helpers."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional


class ResearchService:
    @staticmethod
    def _normalize_prompt(text: str) -> str:
        return " ".join(str(text or "").strip().lower().split())

    @classmethod
    def is_generic_explorer_prompt(cls, text: str) -> bool:
        normalized = cls._normalize_prompt(text)
        return normalized in {
            "more",
            "next",
            "show more",
            "give me more",
            "continue",
            "elaborate",
            "extend",
            "expand",
            "extend this",
            "expand this",
            "continue this",
            "tell me more",
            "more papers",
            "more results",
            "additional papers",
            "additional results",
            "another",
            "others",
        }

    @classmethod
    def is_expansion_request(cls, text: str) -> bool:
        normalized = cls._normalize_prompt(text)
        if not normalized:
            return False
        if cls.is_generic_explorer_prompt(normalized):
            return True

        patterns = [
            r"^(show|give|find|get|fetch|download)\s+(me\s+)?(more|additional|extra|another|next)\b",
            r"^((can|could|would)\s+you|please)\s+(show|give|find|get|fetch|download)\s+(me\s+)?(more|additional|extra|another|next)\b",
            r"^(more|additional|extra|another|next)\s+(papers?|results?|sources?|references?|studies|articles)\b",
            r"^(more|additional|extra|another|next)\s+(about|on|for)\b",
            r"^(extend|expand|continue|elaborate)\b",
            r"^((can|could|would)\s+you|please)\s+(extend|expand|continue|elaborate)\b",
            r"\btell me more\b",
            r"\b(show|give|get|fetch)\s+(me\s+)?(additional|more)\s+(papers?|results?|sources?|references?)\b",
        ]
        return any(re.search(pattern, normalized) for pattern in patterns)

    @classmethod
    def should_resolve_topic_from_history(cls, text: str) -> bool:
        normalized = cls._normalize_prompt(text)
        if not normalized:
            return False
        if cls.is_generic_explorer_prompt(normalized):
            return True
        if not cls.is_expansion_request(normalized):
            return False
        if re.search(r"\b(about|on|for)\b\s+\S+", normalized):
            return False
        if re.search(r"\b(this|it|that|them|those|topic|search|result|results)\b", normalized):
            return True
        if re.search(r"\b(more|additional|extra|another|next)\s+(papers?|results?|sources?|references?|studies|articles)\b", normalized):
            return True
        return False

    @classmethod
    def extract_topic_from_expansion_prompt(cls, text: str) -> str:
        raw_text = str(text or "").strip()
        normalized = cls._normalize_prompt(text)
        if not normalized or not cls.is_expansion_request(normalized):
            return raw_text

        patterns = [
            r"^(show|give|find|get|fetch|download)\s+(me\s+)?(more|additional|extra|another|next)(\s+(papers?|results?|sources?|references?|studies|articles))?(\s+(about|on|for))?\s+(?P<topic>.+)$",
            r"^((can|could|would)\s+you|please)\s+(show|give|find|get|fetch|download)\s+(me\s+)?(more|additional|extra|another|next)(\s+(papers?|results?|sources?|references?|studies|articles))?(\s+(about|on|for))?\s+(?P<topic>.+)$",
            r"^(more|additional|extra|another|next)(\s+(papers?|results?|sources?|references?|studies|articles))?(\s+(about|on|for))\s+(?P<topic>.+)$",
            r"^(extend|expand|continue|elaborate)(\s+(about|on|for))?\s+(?P<topic>.+)$",
            r"^((can|could|would)\s+you|please)\s+(extend|expand|continue|elaborate)(\s+(about|on|for))?\s+(?P<topic>.+)$",
        ]

        blocked_topics = {
            "this",
            "it",
            "that",
            "them",
            "those",
            "topic",
            "this topic",
            "the topic",
            "search",
            "result",
            "results",
        }
        for pattern in patterns:
            match = re.search(pattern, normalized)
            if not match:
                continue
            candidate = str(match.group("topic") or "").strip(" .,!?:;")
            if candidate and candidate not in blocked_topics:
                return candidate
        return raw_text

    @classmethod
    def resolve_topic_from_history(cls, topic: str, chat_history: str | None) -> str:
        if cls.should_resolve_topic_from_history(topic) and chat_history:
            lines = [line.strip() for line in str(chat_history).splitlines() if line.strip()]
            for line in reversed(lines):
                if line.lower().startswith("user:"):
                    clean_user_input = line.split(":", 1)[1].strip() if ":" in line else line
                    if clean_user_input and not cls.should_resolve_topic_from_history(clean_user_input):
                        return cls.extract_topic_from_expansion_prompt(clean_user_input)
        return cls.extract_topic_from_expansion_prompt(topic)

    def run(
        self,
        topic: str,
        chat_history: str | None = None,
        focus_topic: str | None = None,
        use_live: bool | None = None,
        previously_returned_titles: Optional[List[str]] = None,
        previously_returned_papers: Optional[List[Dict[str, Any]]] = None,
        force_refresh: bool = False,
    ) -> Dict[str, Any]:
        from backend.main import _run_research_explorer_impl_legacy, should_use_live_research_sources

        resolved_topic = self.resolve_topic_from_history(topic, chat_history)
        expansion_request = self.is_expansion_request(topic)
        if expansion_request:
            force_refresh = True
            if not focus_topic:
                focus_topic = resolved_topic

        if use_live is None:
            use_live = should_use_live_research_sources(
                resolved_topic,
                chat_history=chat_history,
                focus_topic=focus_topic,
                force_refresh=force_refresh,
            )

        return _run_research_explorer_impl_legacy(
            topic=resolved_topic,
            chat_history=chat_history,
            focus_topic=focus_topic,
            use_live_sources=use_live,
            previously_returned_titles=previously_returned_titles,
            previously_returned_papers=previously_returned_papers,
            force_refresh=force_refresh,
        )
