"""FastAPI backend for the AI Research Assistant (React frontend)."""

from __future__ import annotations

import os
import tempfile
import json
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .assistant_model import assistant_chat, get_assistant_status, train_assistant_model
from .helpers import safe_get


app = FastAPI(title="AI Research Assistant API", version="1.0.0")

_BASE_DIR = Path(__file__).resolve().parent.parent
_FRONTEND_BUILD_DIR = _BASE_DIR / "frontend" / "build"

if _FRONTEND_BUILD_DIR.exists():
    static_dir = _FRONTEND_BUILD_DIR / "static"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=static_dir), name="static")


def _backend_main():
    # Lazy import to avoid heavy startup work during app boot.
    from . import main as backend_main
    return backend_main


@app.on_event("startup")
def _ensure_model_cache_dirs() -> None:
    # Ensure model cache directories exist to avoid repeated downloads on Render.
    for var in ("SENTENCE_TRANSFORMERS_HOME", "HF_HOME", "TRANSFORMERS_CACHE"):
        path = os.getenv(var)
        if path:
            os.makedirs(path, exist_ok=True)
    if (os.getenv("ASSISTANT_TRAIN_ON_STARTUP", "true") or "true").lower() == "true":
        def _background_train() -> None:
            try:
                train_assistant_model(force=False)
            except Exception:
                # Keep API boot resilient even if corpus training cannot complete at startup.
                pass

        threading.Thread(target=_background_train, daemon=True).start()

# Allow local CRA dev server by default.
origins = os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in origins if o.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_RESPONSE_CACHE: Dict[str, Dict[str, Any]] = {}
_CACHE_TTL_SECONDS = int(os.getenv("RESPONSE_CACHE_TTL", "900") or "900")
_CACHE_FILE = os.path.join(os.getenv("PAPER_DB_DIR", "paper_db"), "api_cache.json")


def _cache_key(topic: str, focus: Optional[str], use_live: Optional[bool]) -> str:
    return f"{topic.strip().lower()}|{(focus or '').strip().lower()}|{use_live}"


def _relevant_to_topic(result: Dict[str, Any], topic: str) -> bool:
    tokens = [t for t in topic.lower().replace("-", " ").split() if len(t) > 2]
    if not tokens:
        return True
    table = result.get("table") if isinstance(result, dict) else None
    if not isinstance(table, list):
        return True
    matched = 0
    total = 0
    for row in table:
        if not isinstance(row, dict):
            continue
        total += 1
        hay = " ".join([
            str(row.get("paper_name","")),
            str(row.get("summary_full_paper","")),
            str(row.get("authors_name","")),
        ]).lower()
        if any(tok in hay for tok in tokens):
            matched += 1
    if total == 0:
        return True
    # Require at least half the rows to match to treat cache as relevant.
    return matched / total >= 0.5


def _filter_result_by_topic(result: Dict[str, Any], topic: str) -> Dict[str, Any]:
    if not isinstance(result, dict):
        return result
    tokens = [t for t in topic.lower().replace("-", " ").split() if len(t) > 2]
    if not tokens:
        return result
    table = result.get("table")
    if not isinstance(table, list):
        return result
    filtered = []
    keep_names = set()
    for row in table:
        if not isinstance(row, dict):
            continue
        hay = " ".join([
            str(row.get("paper_name", "")),
            str(row.get("summary_full_paper", "")),
            str(row.get("authors_name", "")),
        ]).lower()
        if any(tok in hay for tok in tokens):
            filtered.append(row)
            name = row.get("paper_name")
            if name:
                keep_names.add(name)
    result["table"] = filtered
    if isinstance(result.get("research_gaps"), list):
        result["research_gaps"] = [g for g in result["research_gaps"] if any(k in g for k in keep_names)] or result["research_gaps"]
    if isinstance(result.get("generated_idea_citations"), list):
        result["generated_idea_citations"] = [c for c in result["generated_idea_citations"] if c in keep_names] or result["generated_idea_citations"]
    return result


def _load_disk_cache() -> Dict[str, Any]:
    try:
        if os.path.exists(_CACHE_FILE):
            with open(_CACHE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return {}


def _save_disk_cache(cache: Dict[str, Any]) -> None:
    try:
        os.makedirs(os.path.dirname(_CACHE_FILE), exist_ok=True)
        with open(_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(cache, f)
    except Exception:
        pass


class ResearchExplorerRequest(BaseModel):
    topic: str
    chat_history: Optional[str] = None
    focus_topic: Optional[str] = None
    use_live: Optional[bool] = None
    force_refresh: Optional[bool] = None


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


class AssistantChatRequest(BaseModel):
    prompt: str
    chat_history: Optional[str] = None


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
    trimmed = trimmed.replace("https://arxiv.org/abs/https://arxiv.org/abs/", "https://arxiv.org/abs/")
    trimmed = trimmed.replace("http://arxiv.org/abs/http://arxiv.org/abs/", "http://arxiv.org/abs/")
    if "doi.org/" in trimmed:
        suffix = trimmed.split("doi.org/", 1)[1]
        if suffix.count(".") == 1 and suffix.replace("v", "").replace(".", "").isdigit():
            return f"https://arxiv.org/abs/{suffix}"
        if "/" in suffix and any(suffix.startswith(prefix) for prefix in ["hep-", "astro-", "cs.", "math.", "physics.", "stat."]):
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
    from .pdf_utils import extract_text
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name
    try:
        paper_text = extract_text(tmp_path)
        backend_main = _backend_main()
        result = backend_main.run_paper_reviewer(paper_text)
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
    backend_main = _backend_main()
    result = backend_main.run_paper_qa(question=payload.question, paper_text=payload.paper_text)
    if isinstance(result, dict) and result.get("error"):
        raise HTTPException(status_code=500, detail=result["error"])
    return {"answer": safe_get(result, "answer", "No answer found.")}


@app.post("/api/research/explore")
def research_explore(payload: ResearchExplorerRequest) -> Dict[str, Any]:
    key = _cache_key(payload.topic, payload.focus_topic, payload.use_live)
    now = time.time()
    disk_cache = _load_disk_cache()
    if not payload.force_refresh:
        cached = _RESPONSE_CACHE.get(key)
        if cached and now - cached.get("ts", 0) <= _CACHE_TTL_SECONDS:
            data = cached.get("data", {})
            if _relevant_to_topic(data, payload.topic):
                return data
        disk_entry = disk_cache.get(key)
        if disk_entry and now - disk_entry.get("ts", 0) <= _CACHE_TTL_SECONDS:
            data = disk_entry.get("data", {})
            if _relevant_to_topic(data, payload.topic):
                _RESPONSE_CACHE[key] = disk_entry
                return data

    backend_main = _backend_main()
    result = backend_main.run_research_explorer(
        topic=payload.topic,
        chat_history=payload.chat_history,
        focus_topic=payload.focus_topic,
        use_live=payload.use_live,
    )
    if isinstance(result, dict) and result.get("error"):
        raise HTTPException(status_code=500, detail=result["error"])
    cleaned = _filter_result_by_topic(_fix_explorer_links(result), payload.topic)
    if isinstance(cleaned, dict) and isinstance(cleaned.get("table"), list) and len(cleaned.get("table")) == 0:
        raise HTTPException(status_code=404, detail="No relevant papers found for the topic. Please refine your query.")
    entry = {"ts": now, "data": cleaned}
    _RESPONSE_CACHE[key] = entry
    disk_cache[key] = entry
    _save_disk_cache(disk_cache)
    return cleaned


@app.post("/api/reference")
def reference_generate(payload: ReferenceRequest) -> Dict[str, Any]:
    backend_main = _backend_main()
    result = backend_main.run_reference_generator(payload.topic)
    if isinstance(result, dict) and result.get("error"):
        raise HTTPException(status_code=500, detail=result["error"])
    return result


@app.post("/api/download")
def download_papers(payload: DownloadRequest) -> Dict[str, Any]:
    backend_main = _backend_main()
    result = backend_main.download_papers_for_topic(payload.topic)
    if isinstance(result, dict) and result.get("error"):
        raise HTTPException(status_code=500, detail=result["error"])
    train_meta = None
    try:
        train_meta = train_assistant_model(force=True)
    except Exception:
        train_meta = None
    return {"status": "ok", "assistant_model": train_meta}


@app.get("/api/assistant/status")
def assistant_status() -> Dict[str, Any]:
    return get_assistant_status()


@app.post("/api/assistant/train")
def assistant_train(payload: AssistantTrainRequest) -> Dict[str, Any]:
    try:
        return train_assistant_model(force=bool(payload.force))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/assistant/chat")
def assistant_chat_route(payload: AssistantChatRequest) -> Dict[str, Any]:
    try:
        return assistant_chat(prompt=payload.prompt, chat_history=payload.chat_history)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/writer/step", response_model=WriterStepResponse)
def writer_step(payload: WriterStepRequest) -> WriterStepResponse:
    """Server-driven writer flow following the strict academic workflow."""
    state = payload.state or {}
    text = payload.user_text.strip()
    phase = state.get("phase") or "start"

    def conference_framework(title: str) -> str:
        return (
            f"Conference Paper Framework for: {title}\n\n"
            "1. Title\n"
            "Purpose: Concise, specific, and informative.\n"
            "Content: Key variables, method, and domain.\n"
            "Guidelines: Avoid jargon; reflect novelty.\n"
            "Length: 8–20 words.\n"
            "Tips: Use active wording; highlight contribution.\n"
            "Common mistakes: Vague titles, overloaded acronyms.\n\n"
            "2. Abstract (150–250 words)\n"
            "Purpose: One-paragraph summary of the entire paper.\n"
            "Content: Problem, method, dataset, results, contributions.\n"
            "Guidelines: Structured flow; no citations.\n"
            "Length: 150–250 words.\n"
            "Tips: Quantify results.\n"
            "Common mistakes: Missing results or contributions.\n\n"
            "3. Problem Statement\n"
            "Purpose: Define the research problem precisely.\n"
            "Content: Scope, constraints, and motivation.\n"
            "Guidelines: Clear, measurable, aligned with method.\n"
            "Length: 100–200 words.\n"
            "Tips: Tie to real-world impact.\n"
            "Common mistakes: Too broad or ambiguous.\n\n"
            "4. Proposed Method\n"
            "Purpose: Present the core approach.\n"
            "Content: Architecture, algorithm, novelty.\n"
            "Guidelines: Technical clarity; reproducibility.\n"
            "Length: 300–700 words.\n"
            "Tips: Use equations and diagrams if needed.\n"
            "Common mistakes: Missing key design details.\n\n"
            "5. Dataset\n"
            "Purpose: Describe data used for evaluation.\n"
            "Content: Source, size, splits, preprocessing.\n"
            "Guidelines: Provide enough detail to reproduce.\n"
            "Length: 150–300 words.\n"
            "Tips: Mention biases/limitations.\n"
            "Common mistakes: Vague dataset description.\n\n"
            "6. Results\n"
            "Purpose: Report experimental outcomes.\n"
            "Content: Metrics, tables, comparisons.\n"
            "Guidelines: Quantitative, objective.\n"
            "Length: 300–600 words.\n"
            "Tips: Use tables/figures for clarity.\n"
            "Common mistakes: No baselines or weak analysis.\n\n"
            "7. Contribution\n"
            "Purpose: Summarize novel contributions.\n"
            "Content: Bullet points of new methods or findings.\n"
            "Guidelines: Distinct, measurable claims.\n"
            "Length: 3–5 bullets.\n"
            "Tips: Align with abstract and results.\n"
            "Common mistakes: Repetition without novelty.\n\n"
            "8. Keywords (4–6)\n"
            "Purpose: Aid indexing and retrieval.\n"
            "Content: Core topics and methods.\n"
            "Guidelines: Specific and relevant.\n"
            "Length: 4–6 keywords.\n"
            "Tips: Include dataset/model if central.\n"
            "Common mistakes: Overly generic terms.\n\n"
            "9. Introduction\n"
            "Purpose: Context and motivation.\n"
            "Content: Background, gap, contributions.\n"
            "Guidelines: Funnel from broad to specific.\n"
            "Length: 600–900 words.\n"
            "Tips: End with contributions list.\n"
            "Common mistakes: Long background, no gap.\n\n"
            "10. Background\n"
            "Purpose: Explain necessary technical background.\n"
            "Content: Key concepts, prior frameworks.\n"
            "Guidelines: Brief and relevant.\n"
            "Length: 300–600 words.\n"
            "Tips: Define terms early.\n"
            "Common mistakes: Excessive textbook content.\n\n"
            "11. Motivation\n"
            "Purpose: Justify why the problem matters.\n"
            "Content: Practical/academic importance.\n"
            "Guidelines: Evidence-based.\n"
            "Length: 200–400 words.\n"
            "Tips: Use real-world examples.\n"
            "Common mistakes: Claims without support.\n\n"
            "12. Research Problem\n"
            "Purpose: Formalize the problem statement.\n"
            "Content: Variables, constraints, objectives.\n"
            "Guidelines: Precise, formal where possible.\n"
            "Length: 150–300 words.\n"
            "Tips: Align with evaluation metrics.\n"
            "Common mistakes: Misalignment with method.\n\n"
            "13. Contributions (bullet points)\n"
            "Purpose: Highlight novelty.\n"
            "Content: 3–5 concise bullets.\n"
            "Guidelines: Actionable and measurable.\n"
            "Length: 3–5 bullets.\n"
            "Tips: Tie each to evidence.\n"
            "Common mistakes: Duplicate points.\n\n"
            "14. Related Work\n"
            "Purpose: Position against prior studies.\n"
            "Content: Key papers, approaches, limitations.\n"
            "Guidelines: Comparative and critical.\n"
            "Length: 600–1000 words.\n"
            "Tips: Include comparison table suggestion.\n"
            "Common mistakes: Summaries without critique.\n\n"
            "15. Methodology / Proposed Model\n"
            "Purpose: Full technical description.\n"
            "Content: Architecture, math formulation, algorithm.\n"
            "Guidelines: Reproducible detail.\n"
            "Length: 800–1200 words.\n"
            "Tips: Provide workflow diagram explanation.\n"
            "Common mistakes: Skipping implementation details.\n\n"
            "16. Dataset & Experimental Setup\n"
            "Purpose: Explain data and experiment configuration.\n"
            "Content: Datasets, splits, preprocessing, metrics.\n"
            "Guidelines: Reproducible settings.\n"
            "Length: 400–800 words.\n"
            "Tips: Justify metric selection.\n"
            "Common mistakes: Missing baselines.\n\n"
            "17. Results & Discussion\n"
            "Purpose: Interpret results and compare baselines.\n"
            "Content: Tables, graphs, and analysis.\n"
            "Guidelines: Evidence-based discussion.\n"
            "Length: 600–900 words.\n"
            "Tips: Explain trends and limitations.\n"
            "Common mistakes: Only reporting numbers.\n\n"
            "18. Conclusion\n"
            "Purpose: Summarize and close.\n"
            "Content: Findings, contributions, limitations, future work.\n"
            "Guidelines: Concise and forward-looking.\n"
            "Length: 200–400 words.\n"
            "Tips: Avoid introducing new results.\n"
            "Common mistakes: Overstating claims.\n\n"
            "19. References (IEEE default)\n"
            "Purpose: Cite prior work.\n"
            "Content: All referenced sources.\n"
            "Guidelines: IEEE style unless specified.\n"
            "Length: As needed.\n"
            "Tips: Ensure consistency.\n"
            "Common mistakes: Missing citations.\n\n"
            "Conference notes: Concise (6–8 pages), focus on novelty and experiments, limited literature depth."
        )

    def journal_framework(title: str) -> str:
        return (
            f"Journal Paper Framework for: {title}\n\n"
            "1. Title\n"
            "Purpose: Precise representation of the study.\n"
            "Content: Core variables, method, and domain.\n"
            "Guidelines: Formal and specific.\n"
            "Length: 8–20 words.\n"
            "Tips: Emphasize contribution.\n"
            "Common mistakes: Overly broad titles.\n\n"
            "2. Abstract (200–300 words)\n"
            "Purpose: Summarize study with more detail.\n"
            "Content: Background, problem, method, results, implications.\n"
            "Guidelines: Structured if needed; no citations.\n"
            "Length: 200–300 words.\n"
            "Tips: Include key quantitative results.\n"
            "Common mistakes: Missing implications.\n\n"
            "3. Keywords (5–8)\n"
            "Purpose: Indexing and discoverability.\n"
            "Content: Domain + methods.\n"
            "Guidelines: Specific terms.\n"
            "Length: 5–8 keywords.\n"
            "Tips: Include dataset/model.\n"
            "Common mistakes: Very general keywords.\n\n"
            "4. Introduction\n"
            "Purpose: Contextualize problem and contributions.\n"
            "Content: Background, gap, objectives, contributions.\n"
            "Guidelines: Deep context and scope.\n"
            "Length: 800–1200 words.\n"
            "Tips: End with roadmap.\n"
            "Common mistakes: Weak gap statement.\n\n"
            "5. Literature Review\n"
            "Purpose: Critical synthesis of prior work.\n"
            "Content: Themes, limitations, comparisons.\n"
            "Guidelines: Analytical, not just descriptive.\n"
            "Length: 1200–2000 words.\n"
            "Tips: Build toward research gap.\n"
            "Common mistakes: Unstructured summaries.\n\n"
            "6. Research Gap & Problem Statement\n"
            "Purpose: Formalize the unresolved issue.\n"
            "Content: Gap, objectives, hypotheses.\n"
            "Guidelines: Evidence-backed.\n"
            "Length: 300–600 words.\n"
            "Tips: Tie directly to method and evaluation.\n"
            "Common mistakes: Vague gap definition.\n\n"
            "7. Proposed Methodology\n"
            "Purpose: Present the approach in detail.\n"
            "Content: Model, algorithm, assumptions.\n"
            "Guidelines: Reproducible, formal.\n"
            "Length: 1000–1500 words.\n"
            "Tips: Use equations and workflow diagrams.\n"
            "Common mistakes: Missing implementation details.\n\n"
            "8. Theoretical Background (if applicable)\n"
            "Purpose: Provide formal foundations.\n"
            "Content: Theories, proofs, framework.\n"
            "Guidelines: Clear and precise.\n"
            "Length: 600–1000 words.\n"
            "Tips: Relate to methodology.\n"
            "Common mistakes: Unrelated theory.\n\n"
            "9. Experimental Setup\n"
            "Purpose: Describe data and evaluation.\n"
            "Content: Datasets, splits, preprocessing, metrics, baselines.\n"
            "Guidelines: Reproducible detail.\n"
            "Length: 600–1000 words.\n"
            "Tips: Justify metric selection.\n"
            "Common mistakes: Insufficient baseline detail.\n\n"
            "10. Results\n"
            "Purpose: Present findings clearly.\n"
            "Content: Tables, figures, quantitative metrics.\n"
            "Guidelines: Objective reporting.\n"
            "Length: 600–1000 words.\n"
            "Tips: Highlight statistical significance.\n"
            "Common mistakes: Reporting without context.\n\n"
            "11. Discussion\n"
            "Purpose: Deep analytical interpretation.\n"
            "Content: Implications, trade-offs, unexpected results.\n"
            "Guidelines: Critical and reflective.\n"
            "Length: 800–1200 words.\n"
            "Tips: Connect to literature.\n"
            "Common mistakes: Repeating results section.\n\n"
            "12. Implications (Practical + Theoretical)\n"
            "Purpose: Translate findings to impact.\n"
            "Content: Application, theory contribution.\n"
            "Guidelines: Specific and grounded.\n"
            "Length: 300–600 words.\n"
            "Tips: Highlight novelty.\n"
            "Common mistakes: Overgeneralization.\n\n"
            "13. Limitations\n"
            "Purpose: Acknowledge constraints.\n"
            "Content: Data, method, scope limitations.\n"
            "Guidelines: Honest and constructive.\n"
            "Length: 200–400 words.\n"
            "Tips: Suggest mitigation.\n"
            "Common mistakes: Missing or trivial limitations.\n\n"
            "14. Future Work\n"
            "Purpose: Suggest next steps.\n"
            "Content: Extensions, new experiments.\n"
            "Guidelines: Realistic and relevant.\n"
            "Length: 200–400 words.\n"
            "Tips: Link to limitations.\n"
            "Common mistakes: Vague directions.\n\n"
            "15. Conclusion\n"
            "Purpose: Summarize contributions.\n"
            "Content: Findings, significance, closing.\n"
            "Guidelines: Concise, no new data.\n"
            "Length: 200–400 words.\n"
            "Tips: Reinforce impact.\n"
            "Common mistakes: Overstated claims.\n\n"
            "16. References (APA/IEEE depending on field)\n"
            "Purpose: Cite sources.\n"
            "Content: All references used.\n"
            "Guidelines: Consistent style.\n"
            "Length: As needed.\n"
            "Tips: Ensure accuracy.\n"
            "Common mistakes: Missing citations.\n\n"
            "Journal notes: Detailed (10–25 pages), deep theory and literature synthesis, strong validation and comparison."
        )

    if phase == "start":
        next_state = {"phase": "await_title"}
        return WriterStepResponse(
            next_state=next_state,
            messages=[
                "To begin, please provide:\n1) Paper Title\n2) Mode of Paper (Conference / Journal)\n\nLet's start with the Paper Title."
            ],
        )

    if phase == "await_title":
        if not text:
            return WriterStepResponse(
                next_state=state,
                messages=["Please provide the Paper Title to proceed."],
            )
        next_state = {**state, "phase": "await_mode", "title": text}
        return WriterStepResponse(
            next_state=next_state,
            messages=["Great. Now specify the Mode of Paper (Conference / Journal)."],
        )

    if phase == "await_mode":
        mode = text.lower()
        if "conference" in mode:
            mode_value = "Conference"
        elif "journal" in mode:
            mode_value = "Journal"
        else:
            return WriterStepResponse(
                next_state=state,
                messages=["Please choose a valid mode: Conference or Journal."],
            )
        title = state.get("title", "Untitled Paper")
        framework = conference_framework(title) if mode_value == "Conference" else journal_framework(title)
        next_state = {**state, "phase": "await_proceed", "mode": mode_value}
        return WriterStepResponse(
            next_state=next_state,
            messages=[
                framework,
                "How do you want to proceed?\n(A) Write section-by-section\n(B) Generate full paper at once\n(C) Provide additional details first (dataset, method, domain)",
            ],
        )

    if phase == "await_proceed":
        choice = text.lower()
        if choice.startswith("a"):
            next_state = {**state, "phase": "await_section"}
            return WriterStepResponse(
                next_state=next_state,
                messages=["Which section should we write first? (e.g., Abstract, Introduction, Methodology)"],
            )
        if choice.startswith("b"):
            next_state = {**state, "phase": "await_full_details"}
            return WriterStepResponse(
                next_state=next_state,
                messages=[
                    "Before I generate the full paper, please provide:\n"
                    "- Research domain\n- Dataset used\n- Method/model\n- Tools/frameworks\n"
                    "- Evaluation metrics\n- Key results (if available)\n- Citation style preference (APA / IEEE)"
                ],
            )
        if choice.startswith("c"):
            next_state = {**state, "phase": "await_extra_details"}
            return WriterStepResponse(
                next_state=next_state,
                messages=["Please provide the additional details (dataset, method, domain, etc.)."],
            )
        return WriterStepResponse(
            next_state=state,
            messages=["Please choose A, B, or C to proceed."],
        )

    if phase == "await_section":
        if not text:
            return WriterStepResponse(
                next_state=state,
                messages=["Please specify the section name (e.g., Abstract, Introduction, Methodology)."],
            )
        next_state = {**state, "phase": "await_section_details", "section": text}
        return WriterStepResponse(
            next_state=next_state,
            messages=[
                f"Section noted: {text}. Please share any specific points, datasets, or constraints for this section.",
                "Once you provide details, I will draft the section in formal academic tone and ask whether to refine or proceed.",
            ],
        )

    if phase == "await_section_details":
        section = state.get("section", "Section")
        details = text or "No additional details provided."
        lower_section = section.strip().lower()

        def _claude_style_intro() -> str:
            return (
                "This section establishes the research context, articulates the gap, and "
                "motivates the proposed approach in a concise, evidence-driven manner."
            )

        def _draft_introduction() -> str:
            return (
                f"{section}\n"
                f"{'-' * len(section)}\n"
                f"{_claude_style_intro()}\n\n"
                "Paragraph 1 (Context): Provide a brief overview of the problem domain and "
                "why it matters in practice and research. Anchor claims with citations "
                "[Author, Year – Placeholder].\n\n"
                "Paragraph 2 (Gap): Summarize prior approaches and their limitations. "
                "Explicitly state the unresolved gap that motivates this work "
                "[Author, Year – Placeholder].\n\n"
                "Paragraph 3 (Approach): Present the proposed solution at a high level, "
                "highlighting novelty and expected impact. Include a compact contribution list.\n\n"
                f"Specific details provided: {details}\n\n"
                "Contributions (bullet points):\n"
                "- Contribution 1 (clearly measurable and novel).\n"
                "- Contribution 2 (methodological or empirical).\n"
                "- Contribution 3 (dataset, benchmark, or analysis).\n"
            )

        def _draft_abstract() -> str:
            return (
                f"{section}\n"
                f"{'-' * len(section)}\n"
                "Background: One sentence about the problem context.\n"
                "Problem: One sentence stating the research gap.\n"
                "Method: One sentence describing the proposed approach.\n"
                "Results: One sentence with key quantitative outcomes.\n"
                "Contributions: One sentence summarizing the main contributions.\n\n"
                f"Specific details provided: {details}\n\n"
                "Note: Replace placeholders with concrete numbers and citations where appropriate."
            )

        def _draft_methodology() -> str:
            return (
                f"{section}\n"
                f"{'-' * len(section)}\n"
                "Overview: Summarize the proposed method and its intuition.\n\n"
                "System Architecture: Describe components and their interactions. "
                "Include a workflow description and data flow narrative.\n\n"
                "Mathematical Formulation: Define key variables, objective functions, "
                "and constraints. Use equations where needed.\n\n"
                "Algorithm: Provide step-by-step pseudocode and note computational complexity.\n\n"
                f"Specific details provided: {details}\n\n"
                "Note: Add citations for baseline methods and theoretical foundations "
                "[Author, Year – Placeholder]."
            )

        def _draft_results() -> str:
            return (
                f"{section}\n"
                f"{'-' * len(section)}\n"
                "Experimental Setup Summary: Briefly mention datasets, splits, and metrics.\n\n"
                "Main Results: Present a table of results with baselines and highlight "
                "statistically significant improvements.\n\n"
                "Ablation/Analysis: Discuss which components contribute most.\n\n"
                "Discussion: Interpret trends and connect back to the research question.\n\n"
                f"Specific details provided: {details}\n\n"
                "Note: Include comparison tables and figure references."
            )

        def _draft_related_work() -> str:
            return (
                f"{section}\n"
                f"{'-' * len(section)}\n"
                "Theme 1: Summarize key approaches and their limitations "
                "[Author, Year – Placeholder].\n\n"
                "Theme 2: Compare methods with respect to data, scalability, and evaluation.\n\n"
                "Research Gap: Conclude with the specific gap your work addresses.\n\n"
                f"Specific details provided: {details}\n\n"
                "Suggestion: Include a comparison table (method, dataset, metric, limitations)."
            )

        def _draft_conclusion() -> str:
            return (
                f"{section}\n"
                f"{'-' * len(section)}\n"
                "Summary: Restate the problem and key findings in 2–3 sentences.\n\n"
                "Contributions: Briefly reiterate the main contributions.\n\n"
                "Limitations: Acknowledge constraints candidly.\n\n"
                "Future Work: Provide 2–3 concrete next steps.\n\n"
                f"Specific details provided: {details}"
            )

        if "abstract" in lower_section:
            draft = _draft_abstract()
        elif "introduction" in lower_section:
            draft = _draft_introduction()
        elif "method" in lower_section or "methodology" in lower_section:
            draft = _draft_methodology()
        elif "result" in lower_section or "discussion" in lower_section:
            draft = _draft_results()
        elif "related work" in lower_section:
            draft = _draft_related_work()
        elif "conclusion" in lower_section:
            draft = _draft_conclusion()
        else:
            draft = (
                f"{section}\n"
                f"{'-' * len(section)}\n"
                "Write a concise, formal academic section with clear structure, "
                "logical flow, and evidence-based statements. Use placeholder citations "
                "where sources are not yet specified.\n\n"
                f"Specific details provided: {details}"
            )
        next_state = {**state, "phase": "await_refine_choice", "last_draft": draft}
        return WriterStepResponse(
            next_state=next_state,
            messages=[
                draft,
                "Does any information need to be updated? (Yes/No)",
            ],
        )

    if phase == "await_refine_choice":
        answer = text.lower()
        if answer.startswith("y"):
            next_state = {**state, "phase": "await_refine_updates"}
            return WriterStepResponse(
                next_state=next_state,
                messages=["Please provide the updates or corrections for this section."],
            )
        if answer.startswith("n"):
            next_state = {**state, "phase": "await_section"}
            return WriterStepResponse(
                next_state=next_state,
                messages=["Which section should we write next? (e.g., Abstract, Methodology, Results)"],
            )
        return WriterStepResponse(
            next_state=state,
            messages=["Please reply with Yes or No. Does any information need to be updated?"],
        )

    if phase == "await_refine_updates":
        section = state.get("section", "Section")
        prev = state.get("last_draft", "")
        updates = text or ""
        updated = (
            f"{prev}\n\n"
            f"Updates Applied:\n{updates}\n\n"
            "Revised section reflects the updates above. Please verify accuracy."
        )
        next_state = {**state, "phase": "await_refine_choice", "last_draft": updated}
        return WriterStepResponse(
            next_state=next_state,
            messages=[
                updated,
                "Does any information need to be updated? (Yes/No)",
            ],
        )

    if phase == "await_full_details":
        return WriterStepResponse(
            next_state=state,
            messages=[
                "Details received. I will draft the full paper with placeholders for unknown citations. "
                "If you want to add or adjust any detail, please provide it now."
            ],
        )

    if phase == "await_extra_details":
        return WriterStepResponse(
            next_state=state,
            messages=[
                "Details received. Would you like to proceed section-by-section (A) or full paper (B)?"
            ],
        )

    return WriterStepResponse(
        next_state={"phase": "start"},
        messages=["Let's restart. Please provide Paper Title and Mode (Conference / Journal)."],
    )


if _FRONTEND_BUILD_DIR.exists():
    @app.get("/")
    def serve_root() -> FileResponse:
        return FileResponse(_FRONTEND_BUILD_DIR / "index.html")

    @app.get("/{full_path:path}")
    def serve_spa(full_path: str) -> FileResponse:
        file_path = _FRONTEND_BUILD_DIR / full_path
        if file_path.is_file():
            return FileResponse(file_path)
        return FileResponse(_FRONTEND_BUILD_DIR / "index.html")
