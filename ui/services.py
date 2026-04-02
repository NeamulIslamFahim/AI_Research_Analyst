"""Workflow helpers that connect Streamlit UI actions to backend functions."""

from __future__ import annotations

import hashlib
import os
import threading
from typing import Any

import streamlit as st
from fastapi import HTTPException

from backend.app import writer_step
from backend.assistant_model import schedule_assistant_retrain
from backend.main import (
    download_papers_for_topic,
    run_paper_reviewer_followup,
    run_paper_reviewer,
    run_research_explorer,
)
from backend.pdf_utils import extract_text
from backend.schemas import ResearchExplorerRequest, WriterStepRequest
from backend.services.response_templates import build_research_error_response

from .helpers import format_chat_history, format_chat_history_up_to, save_uploaded_pdf
from .state import (
    current_session,
    replace_or_append_assistant,
    replace_or_insert_assistant_after_user,
    update_current_session,
)


def start_background_download(topic: str) -> None:
    """Download and index papers without blocking the UI."""

    def _worker() -> None:
        try:
            download_papers_for_topic(topic)
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

        result = run_paper_reviewer_followup(prompt, paper_text)
        answer = result.get("answer") or "No answer found."
        return (
            {"role": "assistant", "content": answer, "type": "text", "display_text": answer},
            None,
        )

    if mode == "Research Explorer":
        focus_topic = prompt if len(prompt.split()) > 8 else None
        previously_returned_titles = previously_returned_titles or []
        try:
            result = run_research_explorer(
                topic=prompt,
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

        return (
            {
                "role": "assistant",
                "content": result,
                "type": "research",
                "display_text": result.get("assistant_reply", "Research result"),
            },
            {"background_download_topic": prompt},
        )

    raise RuntimeError(f"Unsupported mode: {mode}")


def regenerate_from_user_message(user_idx: int) -> None:
    """Regenerate the answer for one earlier user prompt."""
    session = current_session()
    if session["mode"] == "Research Paper Writer":
        return

    messages = session["messages"]
    if user_idx < 0 or user_idx >= len(messages):
        return

    user_msg = messages[user_idx]
    prompt = (user_msg.get("effective_query") or user_msg.get("display_text") or user_msg.get("content") or "").strip()
    if not prompt:
        return

    history = format_chat_history_up_to(messages, user_idx, 100)
    spinner_text = "Regenerating response..."
    if session["mode"] == "Research Explorer":
        spinner_text = "Regenerating research response from external sources..."

    previously_returned_titles = _extract_previously_returned_titles(session)

    with st.spinner(spinner_text):
        assistant_msg, meta = assistant_response_for_prompt(
            session["mode"],
            prompt,
            session,
            history,
            force_refresh=True,
            previously_returned_titles=previously_returned_titles,
        )
        updated_messages = replace_or_insert_assistant_after_user(messages, user_idx, assistant_msg)
        update_current_session(messages=updated_messages)
        if meta and meta.get("background_download_topic"):
            start_background_download(meta["background_download_topic"])
        schedule_assistant_retrain()


def save_edited_user_message(user_idx: int) -> None:
    """Persist an edited user message and regenerate the matching response."""
    session = current_session()
    messages = list(session["messages"])
    if user_idx < 0 or user_idx >= len(messages):
        return

    new_text = (st.session_state.edit_message_text or "").strip()
    if not new_text:
        st.session_state.edit_message_index = None
        st.session_state.edit_message_text = ""
        return

    messages[user_idx] = {
        **messages[user_idx],
        "content": new_text,
        "display_text": new_text,
        "effective_query": new_text,
    }
    update_current_session(messages=messages)
    st.session_state.edit_message_index = None
    st.session_state.edit_message_text = ""
    regenerate_from_user_message(user_idx)


def submit_edited_user_message(user_idx: int) -> None:
    """Small wrapper so Streamlit callbacks stay readable."""
    save_edited_user_message(user_idx)


def ensure_writer_intro(session: dict[str, Any]) -> None:
    """Insert the initial writer assistant messages once per workspace."""
    if session["mode"] != "Research Paper Writer":
        return
    if session["messages"] or session.get("writer_intro_shown"):
        return

    response = writer_step(WriterStepRequest(user_text="", state=session.get("writer_state") or {"phase": "start"}))
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
        result = run_paper_reviewer(paper_text)
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass

    if result.get("error"):
        final_msg = {"role": "assistant", "content": result["error"], "type": "text", "display_text": result["error"]}
        update_current_session(messages=replace_or_append_assistant(current_session()["messages"], final_msg))
        schedule_assistant_retrain()
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
    schedule_assistant_retrain()
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

    with st.spinner(spinner_text):
        try:
            if mode == "Research Paper Writer":
                response = writer_step(WriterStepRequest(user_text=trimmed, state=session.get("writer_state") or {"phase": "start"}))
                replies = [
                    {"role": "assistant", "content": message, "type": "text", "display_text": message}
                    for message in response.messages
                ]
                update_current_session(
                    messages=[*session["messages"][:-1], *replies],
                    writer_state=response.next_state or {"phase": "start"},
                )
                schedule_assistant_retrain()
                return

            if mode == "Research Paper Reviewer":
                paper_text = session.get("paper_text") or ""
                if not paper_text:
                    final_msg = {"role": "assistant", "content": "Please upload a PDF first.", "type": "text", "display_text": "Please upload a PDF first."}
                else:
                    result = run_paper_reviewer_followup(trimmed, paper_text)
                    answer = result.get("answer") or "No answer found."
                    final_msg = {"role": "assistant", "content": answer, "type": "text", "display_text": answer}
                update_current_session(messages=replace_or_append_assistant(session["messages"], final_msg))
                schedule_assistant_retrain()
                return

            if mode == "Research Explorer":
                previously_returned_titles = _extract_previously_returned_titles(session)
                focus_topic = trimmed if len(trimmed.split()) > 8 else None
                try:
                    result = run_research_explorer(
                        topic=trimmed,
                        chat_history=history,
                        use_live=None,
                        focus_topic=focus_topic,
                        previously_returned_titles=previously_returned_titles,
                    )
                except HTTPException as exc:
                    detail = exc.detail if hasattr(exc, "detail") else str(exc)
                    result = research_error_result(str(detail))
                if isinstance(result, dict) and result.get("error"):
                    result = research_error_result(str(result.get("error", "")))
                final_msg = {"role": "assistant", "content": result, "type": "research", "display_text": result.get("assistant_reply", "Research result")}
                update_current_session(messages=replace_or_append_assistant(session["messages"], final_msg))
                start_background_download(trimmed)
                schedule_assistant_retrain()
                return

        except Exception as exc:
            final_msg = {"role": "assistant", "content": str(exc), "type": "text", "display_text": str(exc)}
            update_current_session(messages=replace_or_append_assistant(session["messages"], final_msg))
            schedule_assistant_retrain()
