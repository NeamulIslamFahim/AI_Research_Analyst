"""Paper review service wrapper."""

from __future__ import annotations

from typing import Any, Dict


class ReviewService:
    def run(self, paper_text: str) -> Dict[str, Any]:
        from backend.main import _run_paper_reviewer_impl

        return _run_paper_reviewer_impl(paper_text)

    def followup(self, question: str, paper_text: str) -> Dict[str, Any]:
        from backend.main import _run_paper_reviewer_followup_impl

        return _run_paper_reviewer_followup_impl(question, paper_text)

