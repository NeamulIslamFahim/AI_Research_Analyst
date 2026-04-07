"""Workflow registry and thin orchestration wrappers for backend services."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from backend.services.qa_service import QAService
from backend.services.reference_service import ReferenceService
from backend.services.retry_workflow import RetryWorkflow
from backend.services.review_service import ReviewService
from backend.services.state_models import (
    QAResultSchema,
    QAState,
    ReferenceResultSchema,
    ReferenceState,
    ResearchState,
    ReviewResultSchema,
    ReviewState,
)
from backend.services.validation import (
    strict_validate,
    validate_qa_result,
    validate_reference_result,
    validate_review_result,
)


class ResearchExplorer:
    """Workflow for the Research Explorer feature."""

    def __init__(self):
        self.graph = self._build_graph()

    def _build_graph(self):
        from backend.main import _build_research_graph

        return _build_research_graph()

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
        from backend.main import should_use_live_research_sources

        resolved_topic = topic
        if chat_history:
            from backend.services.research_service import ResearchService

            resolved_topic = ResearchService.resolve_topic_from_history(topic, chat_history)

        effective_live = use_live
        if effective_live is None:
            effective_live = should_use_live_research_sources(
                resolved_topic,
                chat_history=chat_history,
                focus_topic=focus_topic,
                force_refresh=force_refresh,
            )

        state: ResearchState = {
            "topic": resolved_topic,
            "chat_history": chat_history,
            "focus_topic": focus_topic,
            "use_live": effective_live,
            "result": None,
            "retries": 0,
            "validation_error": None,
            "previously_returned_titles": previously_returned_titles,
            "previously_returned_papers": previously_returned_papers,
            "force_refresh": force_refresh,
        }
        out = self.graph.invoke(state)
        return out.get("result") or {}


class PaperReviewer(RetryWorkflow[ReviewState]):
    """Workflow for the Paper Reviewer feature."""

    state_schema = ReviewState
    result_schema = ReviewResultSchema

    def __init__(self):
        self.service = ReviewService()
        super().__init__()

    def _build_graph(self):
        from backend.main import load_env_var

        fast_mode = (load_env_var("FAST_MODE", "false") or "false").lower() == "true"
        self.max_retries = 0 if fast_mode else 2
        return super()._build_graph()

    def _run_step(self, state: ReviewState) -> Dict[str, Any]:
        return {**state, "result": self.service.run(state["paper_text"])}

    def _check_step(self, state: ReviewState) -> Dict[str, Any]:
        cleaned = validate_review_result(state.get("result") or {})
        ok, err = strict_validate(ReviewResultSchema, cleaned)
        retries = state.get("retries", 0) + (0 if ok else 1)
        return {**state, "result": cleaned, "validation_error": None if ok else err, "retries": retries}

    def run(self, paper_text: str) -> Dict[str, Any]:
        state: ReviewState = {"paper_text": paper_text, "result": None, "retries": 0, "validation_error": None}
        return self.graph.invoke(state).get("result") or {}


class PaperQA(RetryWorkflow[QAState]):
    """Workflow for Paper Q&A."""

    state_schema = QAState
    result_schema = QAResultSchema

    def __init__(self):
        self.service = QAService()
        super().__init__()

    def _run_step(self, state: QAState) -> Dict[str, Any]:
        return {**state, "result": self.service.run(state["question"], state["paper_text"])}

    def _check_step(self, state: QAState) -> Dict[str, Any]:
        cleaned = validate_qa_result(state.get("result") or {})
        ok, err = strict_validate(QAResultSchema, cleaned)
        retries = state.get("retries", 0) + (0 if ok else 1)
        return {**state, "result": cleaned, "validation_error": None if ok else err, "retries": retries}

    def run(self, question: str, paper_text: str) -> Dict[str, Any]:
        state: QAState = {"question": question, "paper_text": paper_text, "result": None, "retries": 0, "validation_error": None}
        return self.graph.invoke(state).get("result") or {}


class ReferenceGenerator(RetryWorkflow[ReferenceState]):
    """Workflow for Reference Generation."""

    state_schema = ReferenceState
    result_schema = ReferenceResultSchema

    def __init__(self):
        self.service = ReferenceService()
        super().__init__()

    def _run_step(self, state: ReferenceState) -> Dict[str, Any]:
        return {**state, "result": self.service.run(state["topic"])}

    def _check_step(self, state: ReferenceState) -> Dict[str, Any]:
        cleaned = validate_reference_result(state.get("result") or {})
        ok, err = strict_validate(ReferenceResultSchema, cleaned)
        retries = state.get("retries", 0) + (0 if ok else 1)
        return {**state, "result": cleaned, "validation_error": None if ok else err, "retries": retries}

    def run(self, topic: str) -> Dict[str, Any]:
        state: ReferenceState = {"topic": topic, "result": None, "retries": 0, "validation_error": None}
        return self.graph.invoke(state).get("result") or {}


_RESEARCH_EXPLORER: ResearchExplorer | None = None
_PAPER_REVIEWER: PaperReviewer | None = None
_PAPER_QA: PaperQA | None = None
_REFERENCE_GENERATOR: ReferenceGenerator | None = None


def _get_research_explorer() -> ResearchExplorer:
    global _RESEARCH_EXPLORER
    if _RESEARCH_EXPLORER is None:
        _RESEARCH_EXPLORER = ResearchExplorer()
    return _RESEARCH_EXPLORER


def _get_paper_reviewer() -> PaperReviewer:
    global _PAPER_REVIEWER
    if _PAPER_REVIEWER is None:
        _PAPER_REVIEWER = PaperReviewer()
    return _PAPER_REVIEWER


def _get_paper_qa() -> PaperQA:
    global _PAPER_QA
    if _PAPER_QA is None:
        _PAPER_QA = PaperQA()
    return _PAPER_QA


def _get_reference_generator() -> ReferenceGenerator:
    global _REFERENCE_GENERATOR
    if _REFERENCE_GENERATOR is None:
        _REFERENCE_GENERATOR = ReferenceGenerator()
    return _REFERENCE_GENERATOR


def run_research_explorer(
    topic: str,
    chat_history: str | None = None,
    focus_topic: str | None = None,
    use_live: bool | None = None,
    previously_returned_titles: Optional[List[str]] = None,
    previously_returned_papers: Optional[List[Dict[str, Any]]] = None,
    force_refresh: bool = False,
) -> Dict[str, Any]:
    result = _get_research_explorer().run(
        topic,
        chat_history=chat_history,
        focus_topic=focus_topic,
        use_live=use_live,
        previously_returned_titles=previously_returned_titles,
        previously_returned_papers=previously_returned_papers,
        force_refresh=force_refresh,
    )
    if isinstance(result, dict) and result.get("error"):
        from backend.main import build_research_error_response
        from backend.services.response_factory import ResearchResponseComposer

        fallback = build_research_error_response(str(result.get("error", "")))
        composer = ResearchResponseComposer(topic)
        fallback["assistant_reply"] = (
            f"I couldn't retrieve enough papers for '{topic}' right now. "
            "Try again after the topic is indexed locally, or broaden the query slightly."
        )
        fallback["generated_idea"] = (
            "Broaden the topic a little, retrieve five papers, then narrow to one focused research question once the local index has stronger coverage."
        )
        fallback["generated_idea_steps"] = [
            f"Use '{topic}' as the query anchor, but add one narrower keyword so the retrieval stays on topic.",
            "Collect five close-match papers and ignore broad fallback results that only mention the field in passing.",
            f"Compare the papers on the same evaluation axis, such as robustness, datasets, or validation depth for {composer._topic_theme()}.",
            "Turn the most repeated weakness into one concrete research question.",
        ]
        fallback["error"] = result.get("error")
        return fallback
    return result


def run_paper_reviewer(paper_text: str) -> Dict[str, Any]:
    return _get_paper_reviewer().run(paper_text)


def run_paper_reviewer_followup(question: str, paper_text: str) -> Dict[str, Any]:
    return _get_paper_reviewer().service.followup(question, paper_text)


def run_paper_qa(question: str, paper_text: str) -> Dict[str, Any]:
    return _get_paper_qa().run(question, paper_text)


def run_reference_generator(topic: str) -> Dict[str, Any]:
    return _get_reference_generator().run(topic)
