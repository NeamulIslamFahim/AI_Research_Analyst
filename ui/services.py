"""Workflow helpers that connect Streamlit UI actions to backend functions."""

from __future__ import annotations

import hashlib
import os
import re
from typing import Any

import streamlit as st
from fastapi import HTTPException

from backend.pdf_utils import extract_text
from backend.schemas import ResearchExplorerRequest, WriterStepRequest
from backend.services.response_templates import build_research_error_response
from backend.services.research_service import ResearchService

from .helpers import format_chat_history, save_uploaded_pdf
from .state import (
    current_session,
    replace_or_append_assistant,
    update_current_session,
)


def _backend_main():
    from backend import main as backend_main

    return backend_main


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
        if not key or key in seen_keys:
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


def research_error_result(detail: str) -> dict[str, Any]:
    """Return a safe response shape when explorer generation fails."""
    return build_research_error_response(detail)


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
        loading_text = "Searching the local VectorDB and generating a research summary..."
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
        spinner_text = "Research Explorer is working. This can take a bit while the local paper index is searched and summarized."
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
                    answer = result.get("answer") or "No answer found."
                    final_msg = {"role": "assistant", "content": answer, "type": "text", "display_text": answer}
                update_current_session(messages=replace_or_append_assistant(session["messages"], final_msg))
                _maybe_schedule_assistant_retrain()
                return

            if mode == "Research Explorer":
                resolved_topic = _resolve_research_topic(trimmed, history, session)
                is_expansion_request = ResearchService.is_expansion_request(trimmed)
                history_resolved_expansion = is_expansion_request and ResearchService.should_resolve_topic_from_history(trimmed)
                previously_returned_papers = _session_seen_papers(session) if is_expansion_request else []
                previously_returned_titles = _session_seen_titles(session) if is_expansion_request else []
                force_refresh = is_expansion_request
                focus_topic = resolved_topic if is_expansion_request else (trimmed if len(trimmed.split()) > 8 else None)
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
                except HTTPException as exc:
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
