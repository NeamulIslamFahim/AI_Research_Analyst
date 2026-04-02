"""Paper QA service wrapper."""

from __future__ import annotations

from typing import Any, Dict


class QAService:
    def run(self, question: str, paper_text: str) -> Dict[str, Any]:
        from backend.main import _run_paper_qa_impl

        return _run_paper_qa_impl(question, paper_text)

