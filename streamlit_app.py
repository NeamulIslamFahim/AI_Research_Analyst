"""Simple Streamlit entrypoint for the AI Research Assistant."""

from __future__ import annotations

import streamlit as st

from ui.config import MODE_META, MODES, setup_page
from ui.rendering import render_header, render_message, render_reviewer_panel, render_sidebar
from ui.services import (
    ensure_writer_intro,
    handle_send,
    handle_upload,
)
from ui.state import current_session, init_state


setup_page()


def render_chat_thread(session: dict) -> None:
    """Render the full conversation."""
    for idx, msg in enumerate(session["messages"]):
        render_message(msg)


def main() -> None:
    """Run the Streamlit app."""
    init_state(MODES[0])
    render_sidebar()

    session = current_session()
    ensure_writer_intro(session)
    session = current_session()

    render_header(session)

    if session["mode"] == "Research Paper Reviewer":
        render_reviewer_panel(session, on_process_upload=handle_upload)

    render_chat_thread(session)

    prompt = st.chat_input(MODE_META[session["mode"]]["prompt"])
    if prompt:
        handle_send(prompt)
        st.rerun()


if __name__ == "__main__":
    main()
