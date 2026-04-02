"""Session state helpers for the Streamlit UI."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import streamlit as st


def default_sessions(default_mode: str) -> list[dict[str, Any]]:
    """Create the first empty workspace."""
    return [
        {
            "id": "chat-1",
            "title": "New Workspace",
            "mode": default_mode,
            "messages": [],
            "paper_text": "",
            "writer_state": {"phase": "start"},
            "writer_intro_shown": False,
        }
    ]


def _chat_logs_dir() -> Path:
    """Directory for persisted JSON chat logs."""
    path = Path("data/chat_logs")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _chat_log_path(session_id: str) -> Path:
    return _chat_logs_dir() / f"{session_id}.json"


def _persist_session_to_json(session: dict[str, Any]) -> None:
    record = {
        "id": session.get("id"),
        "title": session.get("title"),
        "mode": session.get("mode"),
        "updated_at": datetime.utcnow().isoformat() + "Z",
        "messages": [],
    }

    for msg in session.get("messages", []):
        record["messages"].append({
            "role": msg.get("role"),
            "type": msg.get("type"),
            "display_text": msg.get("display_text"),
            "content": msg.get("content"),
            "effective_query": msg.get("effective_query"),
        })

    try:
        with open(_chat_log_path(session.get("id", "unknown")), "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=2)
    except OSError:
        pass


def init_state(default_mode: str) -> None:
    """Populate Streamlit session state keys used by the app."""
    st.session_state.setdefault("sessions", default_sessions(default_mode))
    st.session_state.setdefault("current_session_id", "chat-1")
    st.session_state.setdefault("edit_message_index", None)
    st.session_state.setdefault("edit_message_text", "")


def current_session() -> dict[str, Any]:
    """Return the selected workspace session."""
    sessions = st.session_state.sessions
    current_id = st.session_state.current_session_id
    for session in sessions:
        if session["id"] == current_id:
            return session
    st.session_state.current_session_id = sessions[0]["id"]
    return sessions[0]


def save_sessions(sessions: list[dict[str, Any]]) -> None:
    """Replace the whole session list and persist the current workspace in JSON."""
    st.session_state.sessions = sessions
    try:
        _persist_session_to_json(current_session())
    except Exception:
        # Best effort persistence, do not break UI on write failure.
        pass


def update_current_session(**patch: Any) -> None:
    """Update just the active workspace session."""
    current_id = st.session_state.current_session_id
    updated_sessions = []
    for session in st.session_state.sessions:
        if session["id"] == current_id:
            updated_sessions.append({**session, **patch})
        else:
            updated_sessions.append(session)
    save_sessions(updated_sessions)


def new_chat(mode: str) -> None:
    """Create a new workspace at the top of the sidebar list."""
    sessions = st.session_state.sessions
    new_id = f"chat-{len(sessions) + 1}"
    session = {
        "id": new_id,
        "title": "New Workspace",
        "mode": mode,
        "messages": [],
        "paper_text": "",
        "writer_state": {"phase": "start"},
        "writer_intro_shown": False,
    }
    save_sessions([session, *sessions])
    st.session_state.current_session_id = new_id


def replace_or_append_assistant(messages: list[dict[str, Any]], assistant_msg: dict[str, Any]) -> list[dict[str, Any]]:
    """Swap the temporary loading message with the final assistant message."""
    if messages and messages[-1].get("role") == "assistant" and messages[-1].get("type") == "loading":
        updated_messages = list(messages)
        updated_messages[-1] = assistant_msg
        return updated_messages
    return [*messages, assistant_msg]


def replace_or_insert_assistant_after_user(
    messages: list[dict[str, Any]],
    user_idx: int,
    assistant_msg: dict[str, Any],
) -> list[dict[str, Any]]:
    """Replace the assistant reply that follows a user prompt."""
    next_assistant_idx = None
    for idx in range(user_idx + 1, len(messages)):
        if messages[idx].get("role") == "assistant":
            next_assistant_idx = idx
            break

    if next_assistant_idx is None:
        return [*messages, assistant_msg]

    updated_messages = list(messages)
    updated_messages[next_assistant_idx] = assistant_msg
    return updated_messages
