"""Simple Streamlit entrypoint for the AI Research Assistant."""

from __future__ import annotations

import streamlit as st

from ui.config import MODE_META, MODES, setup_page
from ui.rendering import render_header, render_message, render_reviewer_panel, render_sidebar
from ui.services import (
    ensure_writer_intro,
    handle_send,
    handle_upload,
    regenerate_from_user_message,
    submit_edited_user_message,
)
from ui.state import current_session, init_state


setup_page()


def render_chat_thread(session: dict) -> None:
    """Render the full conversation and edit/regenerate controls."""
    for idx, msg in enumerate(session["messages"]):
        render_message(msg)
        if msg.get("role") == "user" and session["mode"] != "Research Paper Writer":
            toolbar_col1, toolbar_col2, toolbar_col3 = st.columns([1, 1.2, 6])
            with toolbar_col1:
                if st.button("Edit", key=f"edit-{session['id']}-{idx}", use_container_width=True):
                    st.session_state.edit_message_index = idx
                    st.session_state.edit_message_text = msg.get("display_text") or msg.get("content") or ""
                    st.rerun()
            with toolbar_col2:
                if st.button("Regenerate", key=f"regen-{session['id']}-{idx}", use_container_width=True):
                    regenerate_from_user_message(idx)
                    st.rerun()
            if st.session_state.edit_message_index == idx:
                st.text_input(
                    "Edit prompt",
                    key="edit_message_text",
                    on_change=submit_edited_user_message,
                    args=(idx,),
                    placeholder="Rewrite your prompt and press Enter...",
                )
                cancel_col, hint_col = st.columns([1, 5])
                with cancel_col:
                    if st.button("Cancel", key=f"cancel-edit-{session['id']}-{idx}", use_container_width=True):
                        st.session_state.edit_message_index = None
                        st.session_state.edit_message_text = ""
                        st.rerun()
                with hint_col:
                    st.caption("Press Enter to save the edit and regenerate the response.")


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
