"""Workflow helpers that connect Streamlit UI actions to backend functions."""

from __future__ import annotations

import hashlib
import os
import threading
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


def start_background_download(topic: str) -> None:
    """Download and index papers without blocking the UI."""

    def _worker() -> None:
        try:
            _backend_main().download_papers_for_topic(topic)
        except Exception:
            pass

    threading.Thread(target=_worker, daemon=True).start()


def _extract_previously_returned_titles(session: dict[str, Any]) -> list[str]:
    """Collect titles from the last research response to avoid duplicates in refresh/regeneration."""
    for msg in reversed(session.get("messages", [])):
        if msg.get("role") == "assistant" and msg.get("type") == "research":
            content = msg.get("content") or {}
            if isinstance(content, dict):
                return [
                    str(row.get("paper_name", "")).strip()
                    for row in content.get("table", [])
                    if isinstance(row, dict) and row.get("paper_name")
                ]
    return []


def research_error_result(detail: str) -> dict[str, Any]:
    """Return a safe response shape when explorer generation fails."""
    return build_research_error_response(detail)


def assistant_response_for_prompt(
    mode: str,
    prompt: str,
    session: dict[str, Any],
    history: str | None,
    force_refresh: bool = False,
    previously_returned_titles: list[str] | None = None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Build one assistant reply for reviewer or explorer modes."""
    if mode == "Research Paper Writer":
        raise RuntimeError("Writer mode is not supported by this helper.")

    if mode == "Research Paper Reviewer":
        paper_text = session.get("paper_text") or ""
        if not paper_text:
            message = "Please upload a PDF first."
            return (
                {"role": "assistant", "content": message, "type": "text", "display_text": message},
                None,
            )

        result = _backend_main().run_paper_reviewer_followup(prompt, paper_text)
        answer = result.get("answer") or "No answer found."
        return (
            {"role": "assistant", "content": answer, "type": "text", "display_text": answer},
            None,
        )

    if mode == "Research Explorer":
        resolved_topic = ResearchService.resolve_topic_from_history(prompt, history)
        is_follow_up = ResearchService.is_generic_explorer_prompt(prompt) or resolved_topic != prompt
        force_refresh = force_refresh or is_follow_up
        focus_topic = resolved_topic if is_follow_up else (prompt if len(prompt.split()) > 8 else None)
        previously_returned_titles = previously_returned_titles or _extract_previously_returned_titles(session)
        if force_refresh and not is_follow_up:
            previously_returned_titles = []
        try:
            result = _backend_main().run_research_explorer(
                topic=resolved_topic,
                chat_history=history,
                use_live=None,
                focus_topic=focus_topic,
                previously_returned_titles=previously_returned_titles,
                force_refresh=force_refresh,
            )
        except HTTPException as exc:
            detail = exc.detail if hasattr(exc, "detail") else str(exc)
            result = research_error_result(str(detail))
        if isinstance(result, dict) and result.get("error"):
            result = research_error_result(str(result.get("error", "")))

        display_text = result.get("assistant_reply", "Research result")
        if is_follow_up and isinstance(display_text, str) and not display_text.lower().startswith("here are additional papers on"):
            display_text = f"Here are additional papers on {resolved_topic}. {display_text}"

        return (
            {
                "role": "assistant",
                "content": result,
                "type": "research",
                "display_text": display_text,
            },
            {"background_download_topic": resolved_topic},
        )

    raise RuntimeError(f"Unsupported mode: {mode}")


def _show_running_notice(message: str):
    """Render a temporary visible notice while the assistant is working."""
    box = st.empty()
    box.info(message)
    return box


def ensure_writer_intro(session: dict[str, Any]) -> None:
    """Insert the initial writer assistant messages once per workspace."""
    if session["mode"] != "Research Paper Writer":
        return
    if session["messages"] or session.get("writer_intro_shown"):
        return

    response = _writer_step(WriterStepRequest(user_text="", state=session.get("writer_state") or {"phase": "start"}))
    update_current_session(
        messages=[
            {"role": "assistant", "content": message, "type": "text", "display_text": message}
            for message in response.messages
        ],
        writer_state=response.next_state or {"phase": "start"},
        writer_intro_shown=True,
    )


def handle_upload(uploaded_file: Any) -> bool:
    """Process a PDF upload and insert the review into the conversation."""
    if uploaded_file is None:
        return False

    session = current_session()
    raw_bytes = uploaded_file.getvalue()
    file_signature = hashlib.sha256(
        (uploaded_file.name or "").encode("utf-8") + b"::" + raw_bytes
    ).hexdigest()
    if session.get("last_uploaded_pdf_signature") == file_signature:
        return False

    update_current_session(last_uploaded_pdf_signature=file_signature)
    messages = [*session["messages"], {"role": "assistant", "content": "Loading...", "type": "loading"}]
    update_current_session(messages=messages)

    tmp_path = save_uploaded_pdf(uploaded_file)
    try:
        paper_text = extract_text(tmp_path)
        result = _backend_main().run_paper_reviewer(paper_text)
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass

    if result.get("error"):
        final_msg = {"role": "assistant", "content": result["error"], "type": "text", "display_text": result["error"]}
        update_current_session(messages=replace_or_append_assistant(current_session()["messages"], final_msg))
        _maybe_schedule_assistant_retrain()
        return False

    review_text = "\n\n".join(
        [
            "Here is a structured peer review of the paper:",
            f"Strengths: {result.get('strengths', '')}",
            f"Weaknesses: {result.get('weaknesses', '')}",
            f"Novelty: {result.get('novelty', '')}",
            f"Technical Correctness: {result.get('technical_correctness', '')}",
            f"Reproducibility: {result.get('reproducibility', '')}",
            f"Recommendation: {result.get('recommendation', '')}",
            f"Suggested Venue: {result.get('suggested_venue', '')}",
        ]
    )
    final_msg = {"role": "assistant", "content": review_text, "type": "review", "display_text": review_text}
    update_current_session(
        messages=replace_or_append_assistant(current_session()["messages"], final_msg),
        paper_text=paper_text,
    )
    _maybe_schedule_assistant_retrain()
    return True


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
        loading_text = "Searching papers, comparing sources, and generating a research summary..."
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
        spinner_text = "Research Explorer is working. This can take a bit while papers are retrieved and summarized."
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
                resolved_topic = ResearchService.resolve_topic_from_history(trimmed, history)
                is_follow_up = ResearchService.is_generic_explorer_prompt(trimmed) or resolved_topic != trimmed
                previously_returned_titles = _extract_previously_returned_titles(session)
                force_refresh = is_follow_up
                focus_topic = resolved_topic if is_follow_up else (trimmed if len(trimmed.split()) > 8 else None)
                try:
                    result = _backend_main().run_research_explorer(
                        topic=resolved_topic,
                        chat_history=history,
                        use_live=None,
                        focus_topic=focus_topic,
                        previously_returned_titles=previously_returned_titles,
                        force_refresh=force_refresh,
                    )
                except HTTPException as exc:
                    detail = exc.detail if hasattr(exc, "detail") else str(exc)
                    result = research_error_result(str(detail))
                if isinstance(result, dict) and result.get("error"):
                    result = research_error_result(str(result.get("error", "")))
                final_msg = {"role": "assistant", "content": result, "type": "research", "display_text": result.get("assistant_reply", "Research result")}
                update_current_session(messages=replace_or_append_assistant(session["messages"], final_msg))
                start_background_download(resolved_topic)
                _maybe_schedule_assistant_retrain()
                return

    except Exception as exc:
        final_msg = {"role": "assistant", "content": str(exc), "type": "text", "display_text": str(exc)}
        update_current_session(messages=replace_or_append_assistant(session["messages"], final_msg))
        _maybe_schedule_assistant_retrain()
    finally:
        notice.empty()
