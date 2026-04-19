"""Session state helpers for the Streamlit UI."""

from __future__ import annotations

from typing import Any

import streamlit as st

from .persistence import get_or_create_owner_id, load_sessions, save_sessions as persist_sessions

def default_sessions(default_mode: str) -> list[dict[str, Any]]:
    """Create the first empty workspace."""
    return [
        {
            "id": "chat-1",
            "title": "New Workspace",
            "mode": default_mode,
            "messages": [],
            "paper_text": "",
            "last_uploaded_pdf_signature": "",
            "writer_state": {"phase": "start"},
            "writer_intro_shown": False,
            "research_last_topic": "",
            "research_seen_papers": [],
        }
    ]


def init_state(default_mode: str) -> None:
    """Populate Streamlit session state keys used by the app."""
    get_or_create_owner_id()
    if "sessions" not in st.session_state:
        loaded_sessions = load_sessions(default_mode)
        st.session_state["sessions"] = loaded_sessions or default_sessions(default_mode)

    sessions = st.session_state["sessions"]
    if not sessions:
        sessions = default_sessions(default_mode)
        st.session_state["sessions"] = sessions

    known_ids = {session.get("id") for session in sessions}
    current_id = st.session_state.get("current_session_id")
    if current_id not in known_ids:
        st.session_state["current_session_id"] = sessions[0]["id"]


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
    """Replace the whole session list and persist the workspaces."""
    st.session_state.sessions = sessions
    try:
        persist_sessions(sessions)
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
    used_numbers = []
    for session in sessions:
        session_id = str(session.get("id", ""))
        if session_id.startswith("chat-"):
            try:
                used_numbers.append(int(session_id.split("-", 1)[1]))
            except Exception:
                continue
    new_id = f"chat-{(max(used_numbers) + 1) if used_numbers else 1}"
    session = {
        "id": new_id,
        "title": "New Workspace",
        "mode": mode,
        "messages": [],
        "paper_text": "",
        "last_uploaded_pdf_signature": "",
        "writer_state": {"phase": "start"},
        "writer_intro_shown": False,
        "research_last_topic": "",
        "research_seen_papers": [],
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
