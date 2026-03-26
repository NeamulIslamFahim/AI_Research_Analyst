"""FastAPI backend for the AI Research Assistant (React frontend)."""

from __future__ import annotations

import os
import tempfile
from typing import Any, Dict, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from helpers import safe_get
from main import (
    download_papers_for_topic,
    run_paper_qa,
    run_paper_reviewer,
    run_research_explorer,
    run_reference_generator,
)
from pdf_utils import extract_text


app = FastAPI(title="AI Research Assistant API", version="1.0.0")

# Allow local CRA dev server by default.
origins = os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in origins if o.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ResearchExplorerRequest(BaseModel):
    topic: str
    chat_history: Optional[str] = None
    focus_topic: Optional[str] = None


class ReviewQARequest(BaseModel):
    question: str
    paper_text: str


class ReferenceRequest(BaseModel):
    topic: str


class DownloadRequest(BaseModel):
    topic: str


class WriterStepRequest(BaseModel):
    step: int
    user_text: str


class WriterStepResponse(BaseModel):
    next_step: int
    messages: list[str]


def _format_review_reply(review: Dict[str, Any]) -> str:
    if not isinstance(review, dict):
        return str(review)
    parts = [
        "Here is a structured peer review of the paper:",
        f"Strengths: {safe_get(review, 'strengths', '')}",
        f"Weaknesses: {safe_get(review, 'weaknesses', '')}",
        f"Novelty: {safe_get(review, 'novelty', '')}",
        f"Technical Correctness: {safe_get(review, 'technical_correctness', '')}",
        f"Reproducibility: {safe_get(review, 'reproducibility', '')}",
        f"Recommendation: {safe_get(review, 'recommendation', '')}",
        f"Suggested Venue: {safe_get(review, 'suggested_venue', '')}",
    ]
    return "\n\n".join([p for p in parts if p and not p.endswith(": ")])


def _normalize_url(url: str) -> str:
    if not url:
        return ""
    trimmed = url.strip()
    trimmed = trimmed.replace("https://doi.org/https://doi.org/", "https://doi.org/")
    if trimmed.startswith("http://") or trimmed.startswith("https://"):
        return trimmed
    if trimmed.startswith("doi.org/"):
        return f"https://{trimmed}"
    if trimmed.startswith("doi:"):
        doi = trimmed.replace("doi:", "").strip()
        return f"https://doi.org/{doi}"
    if trimmed.startswith("10."):
        return f"https://doi.org/{trimmed}"
    if trimmed.startswith("arxiv.org/"):
        return f"https://{trimmed}"
    return f"https://{trimmed}"


def _fix_paper_url(url: str) -> str:
    if not url:
        return ""
    trimmed = _normalize_url(url)
    if "doi.org/" in trimmed:
        suffix = trimmed.split("doi.org/", 1)[1]
        if suffix.count(".") == 1 and suffix.replace("v", "").replace(".", "").isdigit():
            return f"https://arxiv.org/abs/{suffix}"
    return trimmed


def _fix_explorer_links(result: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(result, dict):
        return result
    table = result.get("table")
    if isinstance(table, list):
        for row in table:
            if isinstance(row, dict):
                row["paper_url"] = _fix_paper_url(row.get("paper_url", ""))
    return result


@app.get("/api/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/api/review/upload")
async def review_upload(file: UploadFile = File(...)) -> Dict[str, Any]:
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name
    try:
        paper_text = extract_text(tmp_path)
        result = run_paper_reviewer(paper_text)
        if isinstance(result, dict) and result.get("error"):
            raise HTTPException(status_code=500, detail=result["error"])
        return {
            "paper_text": paper_text,
            "review": result,
            "review_text": _format_review_reply(result if isinstance(result, dict) else {}),
        }
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass


@app.post("/api/review/qa")
def review_qa(payload: ReviewQARequest) -> Dict[str, Any]:
    result = run_paper_qa(question=payload.question, paper_text=payload.paper_text)
    if isinstance(result, dict) and result.get("error"):
        raise HTTPException(status_code=500, detail=result["error"])
    return {"answer": safe_get(result, "answer", "No answer found.")}


@app.post("/api/research/explore")
def research_explore(payload: ResearchExplorerRequest) -> Dict[str, Any]:
    result = run_research_explorer(
        topic=payload.topic,
        chat_history=payload.chat_history,
        focus_topic=payload.focus_topic,
    )
    if isinstance(result, dict) and result.get("error"):
        raise HTTPException(status_code=500, detail=result["error"])
    return _fix_explorer_links(result)


@app.post("/api/reference")
def reference_generate(payload: ReferenceRequest) -> Dict[str, Any]:
    result = run_reference_generator(payload.topic)
    if isinstance(result, dict) and result.get("error"):
        raise HTTPException(status_code=500, detail=result["error"])
    return result


@app.post("/api/download")
def download_papers(payload: DownloadRequest) -> Dict[str, Any]:
    result = download_papers_for_topic(payload.topic)
    if isinstance(result, dict) and result.get("error"):
        raise HTTPException(status_code=500, detail=result["error"])
    return {"status": "ok"}


@app.post("/api/writer/step", response_model=WriterStepResponse)
def writer_step(payload: WriterStepRequest) -> WriterStepResponse:
    """Server-driven writer flow matching the previous Streamlit behavior."""
    step = payload.step
    text = payload.user_text.strip()
    if step == 0:
        return WriterStepResponse(
            next_step=1,
            messages=["What type of paper? Conference or Journal? Please specify Q1, Q2, Q3."],
        )
    if step == 1:
        return WriterStepResponse(
            next_step=2,
            messages=[f"Paper type: {text}. Please provide the paper name."],
        )
    if step == 2:
        return WriterStepResponse(
            next_step=0,
            messages=[
                f"Paper name: {text}. Now writing the paper step by step based on standard outline.",
                "Step 1: Introduction...",
            ],
        )
    return WriterStepResponse(
        next_step=0,
        messages=["Let's start again. What type of paper? Conference or Journal? Please specify Q1, Q2, Q3."],
    )
