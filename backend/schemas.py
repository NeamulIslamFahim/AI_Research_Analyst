"""Pydantic request and response models used by the API and Streamlit UI."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class ResearchExplorerRequest(BaseModel):
    topic: str
    chat_history: Optional[str] = None
    focus_topic: Optional[str] = None
    use_live: Optional[bool] = None
    force_refresh: Optional[bool] = None
    previously_returned_titles: Optional[List[str]] = None
    previously_returned_papers: Optional[List[Dict[str, Any]]] = None


class ReviewQARequest(BaseModel):
    question: str
    paper_text: str


class ReferenceRequest(BaseModel):
    topic: str


class DownloadRequest(BaseModel):
    topic: str


class WriterStepRequest(BaseModel):
    user_text: str
    state: Optional[Dict[str, Any]] = None


class WriterStepResponse(BaseModel):
    next_state: Dict[str, Any]
    messages: list[str]


class AssistantTrainRequest(BaseModel):
    force: Optional[bool] = None
