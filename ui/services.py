"""Workflow helpers that connect Streamlit UI actions to backend functions."""

from __future__ import annotations

import hashlib
import os
import re
from typing import Any

import requests
import streamlit as st
from fastapi import HTTPException

from backend.schemas import ResearchExplorerRequest, WriterStepRequest
from backend.services.response_templates import build_research_error_response
from backend.services.research_service import ResearchService

from .helpers import format_chat_history, save_uploaded_pdf
from .state import (
    current_session,
    replace_or_append_assistant,
    update_current_session,
)


def _is_local_only_mode() -> bool:
    return str(
        os.getenv("LOCAL_ONLY")
        or os.getenv("ASSISTANT_MODEL_ONLY")
        or st.secrets.get("LOCAL_ONLY")
        or st.secrets.get("ASSISTANT_MODEL_ONLY")
        or "false"
    ).strip().lower() == "true"


def _get_backend_url() -> str | None:
    if _is_local_only_mode():
        return None

    return (
        os.getenv("BACKEND_URL")
        or os.getenv("RESEARCH_BACKEND_URL")
        or st.secrets.get("BACKEND_URL")
        or st.secrets.get("RESEARCH_BACKEND_URL")
    )


def _backend_main():
    # If `BACKEND_URL` is set, use the remote FastAPI backend (Streamlit Cloud deployment).
    backend_url = _get_backend_url()
    if backend_url:
        base = backend_url.rstrip("/")

        class RestBackendClient:
            def __init__(self, base_url: str):
                self.base = base_url

            def _post(self, path: str, payload: dict[str, Any] | None = None, files: dict | None = None, timeout: int = 60):
                normalized_path = path
                if self.base.endswith("/api") and normalized_path.startswith("/api"):
                    normalized_path = normalized_path[len("/api") :]
                    if not normalized_path.startswith("/"):
                        normalized_path = "/" + normalized_path
                url = f"{self.base}{normalized_path}"
                try:
                    if files:
                        resp = requests.post(url, files=files, timeout=timeout)
                    else:
                        resp = requests.post(url, json=payload or {}, timeout=timeout)
                    resp.raise_for_status()
                    content_type = resp.headers.get("Content-Type", "") or ""
                    if "html" in content_type.lower() or resp.text.lstrip().startswith("<!doctype html>"):
                        body = resp.text[:500]
                        raise HTTPException(
                            status_code=500,
                            detail=(
                                "Remote backend returned HTML instead of JSON. "
                                "This often means BACKEND_URL is pointing at the wrong host or path. "
                                f"URL: {url}. Response body: {body}"
                            ),
                        )
                    try:
                        return resp.json()
                    except ValueError as exc:
                        body = resp.text[:500]
                        raise HTTPException(
                            status_code=500,
                            detail=(
                                "Invalid JSON response from backend. "
                                "Confirm BACKEND_URL is set to the backend API root and not a static web page. "
                                f"URL: {url}. Response body: {body}"
                            ),
                        ) from exc
                except requests.RequestException as exc:
                    raise HTTPException(status_code=500, detail=str(exc)) from exc

            def assistant_chat(self, prompt: str, chat_history: str | None = None):
                return self._post("/api/assistant/chat", {"prompt": prompt, "chat_history": chat_history})

            def run_research_explorer(self, topic: str, chat_history: str | None = None, use_live=None, focus_topic: str | None = None, previously_returned_titles: list | None = None, previously_returned_papers: list | None = None, force_refresh: bool = False):
                payload = {
                    "topic": topic,
                    "chat_history": chat_history,
                    "use_live": use_live,
                    "focus_topic": focus_topic,
                    "previously_returned_titles": previously_returned_titles or [],
                    "previously_returned_papers": previously_returned_papers or [],
                    "force_refresh": force_refresh,
                }
                return self._post("/api/research/explore", payload, timeout=120)

        return RestBackendClient(base)

    from backend import main as backend_main

    return backend_main


def _is_insufficient_research_result(result: dict[str, Any]) -> bool:
    if not isinstance(result, dict):
        return False
    assistant_reply = str(result.get("assistant_reply") or result.get("answer") or "").lower()
    return any(
        phrase in assistant_reply
        for phrase in [
            "couldn't find five closely relevant papers",
            "is too broad to turn into a trustworthy paper comparison yet",
            "try a narrower topic or add one concrete domain term",
        ]
    )


def ensure_writer_intro(session: dict[str, Any]) -> None:
    """Show the initial writer message if it's the first run for the session."""
    if session.get("mode") != "Research Paper Writer" or session.get("writer_intro_shown"):
        return

    from backend.app import writer_step

    try:
        response = writer_step(WriterStepRequest(user_text="", state={"phase": "start"}))
        intro_messages = [
            {"role": "assistant", "content": message, "type": "text", "display_text": message}
            for message in response.messages
        ]
        update_current_session(
            messages=[*session.get("messages", []), *intro_messages],
            writer_state=response.next_state,
            writer_intro_shown=True,
        )
    except Exception:
        # If the backend isn't ready, we can skip this and let the user initiate.
        pass


def _writer_step(request: WriterStepRequest):
    from backend.app import writer_step

    return writer_step(request)


def _maybe_schedule_assistant_retrain() -> None:
    if (os.getenv("ASSISTANT_AUTO_RETRAIN", "false") or "false").lower() != "true":
        return
    try:
        from backend.assistant_model import schedule_assistant_retrain

        schedule_assistant_retrain()
    except Exception:
        pass


def _normalize_title(value: Any) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip().lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _normalize_url(value: Any) -> str:
    text = re.sub(r"\s+", "", str(value or "").strip().lower())
    return text.rstrip("/")


def _paper_memory_key(ref: dict[str, Any]) -> str:
    url = _normalize_url(ref.get("url") or ref.get("paper_url") or ref.get("pdf_url"))
    if url:
        return f"url:{url}"
    title = _normalize_title(ref.get("title") or ref.get("paper_name"))
    if title:
        return f"title:{title}"
    return ""


def _paper_ref_from_row(row: dict[str, Any], topic: str = "") -> dict[str, Any] | None:
    if not isinstance(row, dict):
        return None
    title = str(row.get("paper_name", "") or row.get("title", "")).strip()
    url = str(row.get("paper_url", "") or row.get("url", "")).strip()
    pdf_url = str(row.get("pdf_url", "")).strip()
    if not title and not url and not pdf_url:
        return None
    ref = {
        "title": title,
        "url": url,
        "pdf_url": pdf_url,
        "source": str(row.get("source", "")).strip(),
        "topic": topic,
    }
    return ref if _paper_memory_key(ref) else None


def _extract_result_paper_refs(result: dict[str, Any], topic: str = "") -> list[dict[str, Any]]:
    refs: list[dict[str, Any]] = []
    for row in result.get("table", []) or []:
        ref = _paper_ref_from_row(row, topic=topic)
        if ref:
            refs.append(ref)
    return refs


def _legacy_seen_papers_from_messages(session: dict[str, Any]) -> list[dict[str, Any]]:
    refs: list[dict[str, Any]] = []
    for msg in session.get("messages", []):
        if msg.get("role") != "assistant" or msg.get("type") != "research":
            continue
        content = msg.get("content") or {}
        if not isinstance(content, dict):
            continue
        refs.extend(_extract_result_paper_refs(content))
    return refs


def _session_seen_papers(session: dict[str, Any]) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    seen_keys: set[str] = set()
    for ref in list(session.get("research_seen_papers") or []) + _legacy_seen_papers_from_messages(session):
        if not isinstance(ref, dict):
            continue
        key = _paper_memory_key(ref)
        if not key or key in seen_keys:
            continue
        seen_keys.add(key)
        merged.append(
            {
                "title": str(ref.get("title", "") or ref.get("paper_name", "")).strip(),
                "url": str(ref.get("url", "") or ref.get("paper_url", "")).strip(),
                "pdf_url": str(ref.get("pdf_url", "")).strip(),
                "source": str(ref.get("source", "")).strip(),
                "topic": str(ref.get("topic", "")).strip(),
            }
        )
    return merged


def _session_seen_titles(session: dict[str, Any]) -> list[str]:
    return [str(ref.get("title", "")).strip() for ref in _session_seen_papers(session) if str(ref.get("title", "")).strip()]


def _merge_session_seen_papers(session: dict[str, Any], result: dict[str, Any], topic: str) -> list[dict[str, Any]]:
    current_seen = _session_seen_papers(session)
    new_refs = _extract_result_paper_refs(result, topic=topic)
    
    seen_keys: set[str] = set()
    merged = list(current_seen)
    seen_keys.update(_paper_memory_key(ref) for ref in merged if _paper_memory_key(ref))

    for ref in new_refs:
        key = _paper_memory_key(ref)
        if not key or key in seen_keys or not ref.get("title"):
            continue
        seen_keys.add(key)
        merged.append(ref)
    return merged[-250:]


def _resolve_research_topic(prompt: str, history: str | None, session: dict[str, Any]) -> str:
    resolved = ResearchService.resolve_topic_from_history(prompt, history)
    if ResearchService.is_expansion_request(prompt) and ResearchService.should_resolve_topic_from_history(prompt):
        memory_topic = str(session.get("research_last_topic", "")).strip()
        if memory_topic:
            return memory_topic
    return resolved


def _looks_like_assistant_request(prompt: str) -> bool:
    normalized = " ".join(str(prompt or "").split()).lower()
    if not normalized:
        return False
    if ResearchService.is_expansion_request(normalized):
        return False
    if len(normalized.split()) >= 12:
        return True
    question_starters = (
        "what ",
        "why ",
        "how ",
        "when ",
        "where ",
        "which ",
        "who ",
        "can ",
        "could ",
        "would ",
        "should ",
        "is ",
        "are ",
        "do ",
        "does ",
        "explain ",
        "summarize ",
        "compare ",
        "tell me ",
        "give me ",
    )
    return "?" in normalized or normalized.startswith(question_starters)


def research_error_result(detail: str) -> dict[str, Any]:
    """Return a safe response shape when explorer generation fails."""
    return build_research_error_response(detail)


def _assistant_answer_text(result: Any) -> str:
    """Return visible text for assistant-like backend payloads."""
    if isinstance(result, dict):
        for key in ("answer", "assistant_reply", "message", "error", "detail"):
            value = result.get(key)
            if value:
                return str(value)
        return "The assistant returned a response, but it did not include displayable answer text."
    if result is None:
        return "The assistant did not return a response."
    return str(result)


def handle_upload(uploaded_file: Any) -> bool:
    """Process an uploaded PDF for the reviewer mode."""
    session = current_session()
    if not uploaded_file:
        return False

    file_bytes = uploaded_file.getvalue()
    signature = hashlib.sha256(file_bytes).hexdigest()
    if signature == session.get("last_uploaded_pdf_signature"):
        return False

    loading_message = {
        "role": "assistant",
        "content": "Processing PDF...",
        "type": "loading",
        "display_text": "Processing uploaded PDF...",
    }
    update_current_session(messages=[*session["messages"], loading_message])

    try:
        from backend.explorer_utils import format_review_reply
        from backend.pdf_utils import extract_text

        backend_url = _get_backend_url()
        file_bytes = uploaded_file.getvalue()
        if backend_url:
            # Send multipart upload to remote FastAPI backend
            url = backend_url.rstrip("/") + "/api/review/upload"
            files = {"file": (uploaded_file.name or "upload.pdf", file_bytes, "application/pdf")}
            try:
                resp = requests.post(url, files=files, timeout=120)
                resp.raise_for_status()
                try:
                    review_json = resp.json()
                except ValueError as exc:
                    raise RuntimeError(
                        f"Remote review upload failed: invalid JSON response from backend: {exc}. Body: {resp.text[:500]}"
                    ) from exc
            except requests.RequestException as exc:
                raise RuntimeError(f"Remote review upload failed: {exc}") from exc

            review_result = review_json.get("review") or review_json
            paper_text = review_json.get("paper_text") or extract_text(save_uploaded_pdf(uploaded_file))

        else:
            temp_path = save_uploaded_pdf(uploaded_file)
            paper_text = extract_text(temp_path)
            review_result = _backend_main().run_paper_reviewer(paper_text)

        if isinstance(review_result, dict) and review_result.get("error"):
            raise RuntimeError(str(review_result["error"]))

        review_text = format_review_reply(review_result if isinstance(review_result, dict) else {})
        final_msg = {"role": "assistant", "content": review_text, "type": "text", "display_text": review_text}
        update_current_session(
            messages=replace_or_append_assistant(session["messages"], final_msg),
            paper_text=paper_text,
            last_uploaded_pdf_signature=signature,
        )
        return True
    except Exception as exc:
        final_msg = {"role": "assistant", "content": str(exc), "type": "text", "display_text": str(exc)}
        update_current_session(messages=replace_or_append_assistant(session["messages"], final_msg))
        return False

def _show_running_notice(message: str):
    """Render a temporary visible notice while the assistant is working."""
    box = st.empty()
    with box.container():
        st.info(message)
    return box

def handle_send(prompt: str) -> None:
    """Handle the main chat input for all workspace modes."""
    session = current_session()
    mode = session["mode"]
    messages = session["messages"]

    trimmed = prompt.strip()
    if not trimmed:
        return

    title = session["title"]
    if not messages:
        title = trimmed[:40] + ("..." if len(trimmed) > 40 else "")

    user_message = {
        "role": "user",
        "content": trimmed,
        "display_text": trimmed,
        "effective_query": trimmed,
    }
    loading_text = "Working on your request..."
    if mode == "Research Explorer":
        loading_text = "Research assistant is analyzing your request..."
    elif mode == "Research Paper Reviewer":
        loading_text = "Reading the uploaded paper and preparing an answer..."
    elif mode == "Research Paper Writer":
        loading_text = "Advancing the writing workflow..."

    update_current_session(
        title=title or "New Workspace",
        messages=[
            *messages,
            user_message,
            {"role": "assistant", "content": "Loading...", "type": "loading", "display_text": loading_text},
        ],
    )

    session = current_session()
    history = format_chat_history(session["messages"], 100)
    spinner_text = "Processing request..."
    if mode == "Research Explorer":
        spinner_text = "Research assistant is working. This can take a bit while papers are retrieved and summarized."
    notice = _show_running_notice(spinner_text)

    try:
        with st.spinner(spinner_text):
            if mode == "Research Paper Writer":
                response = _writer_step(WriterStepRequest(user_text=trimmed, state=session.get("writer_state") or {"phase": "start"}))
                replies = [
                    {"role": "assistant", "content": message, "type": "text", "display_text": message}
                    for message in response.messages
                ]
                update_current_session(
                    messages=[*session["messages"][:-1], *replies],
                    writer_state=response.next_state or {"phase": "start"},
                )
                _maybe_schedule_assistant_retrain()
                return

            if mode == "Research Paper Reviewer":
                paper_text = session.get("paper_text") or ""
                if not paper_text:
                    final_msg = {"role": "assistant", "content": "Please upload a PDF first.", "type": "text", "display_text": "Please upload a PDF first."}
                else:
                    result = _backend_main().run_paper_reviewer_followup(trimmed, paper_text)
                    answer = _assistant_answer_text(result)
                    final_msg = {"role": "assistant", "content": answer, "type": "text", "display_text": answer}
                update_current_session(messages=replace_or_append_assistant(session["messages"], final_msg))
                _maybe_schedule_assistant_retrain()
                return

            if mode == "Research Explorer":
                if _looks_like_assistant_request(trimmed):
                    try:
                        result = _backend_main().assistant_chat(trimmed, chat_history=history)  # type: ignore
                    except HTTPException as exc:
                        detail = exc.detail if hasattr(exc, "detail") else str(exc)
                        result = {"answer": str(detail), "sources": [], "answer_source": "error"}

                    if isinstance(result, dict) and result.get("table") is not None:
                        display_text = result.get("assistant_reply", "Research result")
                        final_msg = {"role": "assistant", "content": result, "type": "research", "display_text": display_text}
                        update_current_session(
                            messages=replace_or_append_assistant(session["messages"], final_msg),
                            research_last_topic=trimmed,
                            research_seen_papers=_merge_session_seen_papers(session, result, trimmed),
                        )
                    else:
                        answer_text = _assistant_answer_text(result)
                        final_msg = {
                            "role": "assistant",
                            "content": result if isinstance(result, dict) else {"answer": str(result), "sources": []},
                            "type": "assistant",
                            "display_text": answer_text,
                        }
                        update_current_session(messages=replace_or_append_assistant(session["messages"], final_msg))
                    _maybe_schedule_assistant_retrain()
                    return

                resolved_topic = _resolve_research_topic(trimmed, history, session)
                is_expansion_request = ResearchService.is_expansion_request(trimmed)
                history_resolved_expansion = is_expansion_request and ResearchService.should_resolve_topic_from_history(trimmed)
                previously_returned_papers = _session_seen_papers(session) if is_expansion_request else []
                previously_returned_titles = _session_seen_titles(session) if is_expansion_request else []
                force_refresh = is_expansion_request
                focus_topic = resolved_topic
                try:
                    result = _backend_main().run_research_explorer( # type: ignore
                        topic=resolved_topic,
                        chat_history="",
                        use_live=None,
                        focus_topic=focus_topic,
                        previously_returned_titles=previously_returned_titles,
                        previously_returned_papers=previously_returned_papers,
                        force_refresh=force_refresh,
                    )
                    if _is_insufficient_research_result(result) and not force_refresh:
                        result = _backend_main().run_research_explorer( # type: ignore
                            topic=resolved_topic,
                            chat_history="",
                            use_live=True,
                            focus_topic=focus_topic,
                            previously_returned_titles=previously_returned_titles,
                            previously_returned_papers=previously_returned_papers,
                            force_refresh=True,
                        )
                except Exception as exc:
                    detail = exc.detail if hasattr(exc, "detail") else str(exc)
                    result = research_error_result(str(detail))
                if isinstance(result, dict) and result.get("error"):
                    result = research_error_result(str(result.get("error", "")))
                display_text = result.get("assistant_reply", "Research result")
                if history_resolved_expansion and isinstance(display_text, str) and not display_text.lower().startswith("here are additional papers on"):
                    display_text = f"Here are additional papers on {resolved_topic}. {display_text}"
                final_msg = {"role": "assistant", "content": result, "type": "research", "display_text": display_text}
                update_current_session(
                    messages=replace_or_append_assistant(session["messages"], final_msg),
                    research_last_topic=resolved_topic,
                    research_seen_papers=_merge_session_seen_papers(session, result, resolved_topic),
                )
                _maybe_schedule_assistant_retrain()
                return

    except Exception as exc:
        final_msg = {"role": "assistant", "content": str(exc), "type": "text", "display_text": str(exc)}
        update_current_session(messages=replace_or_append_assistant(session["messages"], final_msg))
        _maybe_schedule_assistant_retrain()
    finally:
        notice.empty()
