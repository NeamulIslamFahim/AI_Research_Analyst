"""Shared state and schema models for backend workflows."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, TypedDict

from pydantic import BaseModel, Field


class ResearchState(TypedDict):
    topic: str
    chat_history: Optional[str]
    focus_topic: Optional[str]
    use_live: Optional[bool]
    result: Optional[Dict[str, Any]]
    retries: int
    validation_error: Optional[str]
    previously_returned_titles: Optional[List[str]]
    force_refresh: Optional[bool]


class ReviewState(TypedDict):
    paper_text: str
    result: Optional[Dict[str, Any]]
    retries: int
    validation_error: Optional[str]


class QAState(TypedDict):
    question: str
    paper_text: str
    result: Optional[Dict[str, Any]]
    retries: int
    validation_error: Optional[str]


class ReferenceState(TypedDict):
    topic: str
    result: Optional[Dict[str, Any]]
    retries: int
    validation_error: Optional[str]


class ResearchRowSchema(BaseModel):
    paper_name: str
    paper_url: str
    authors_name: str
    summary_full_paper: str
    problem_solved: str
    proposed_model_or_approach: str
    source: str
    score_relevance: int = Field(ge=0, le=10)
    score_quality: int = Field(ge=0, le=10)


class ResearchResultSchema(BaseModel):
    table: List[ResearchRowSchema]
    research_gaps: List[str]
    generated_idea: str
    generated_idea_steps: List[str]
    generated_idea_citations: Optional[List[str]] = None
    assistant_reply: Optional[str] = None
    error: Optional[str] = None


class ReviewResultSchema(BaseModel):
    strengths: str
    weaknesses: str
    novelty: str
    technical_correctness: str
    reproducibility: str
    recommendation: str
    suggested_venue: str


class QAResultSchema(BaseModel):
    answer: str


class ReferenceResultSchema(BaseModel):
    references: List[str]
