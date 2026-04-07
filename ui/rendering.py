"""Rendering helpers for the Streamlit frontend."""

from __future__ import annotations

import html
from typing import Any

import streamlit as st

from .components import BulletListRenderer, IdeaCardRenderer, NumberedStepsRenderer, PaperTableRenderer, TextPreviewer
from .config import MODE_META, MODES
from .state import current_session, new_chat, update_current_session


def render_header(session: dict[str, Any]) -> None:
    """Render the large page header for the active workspace."""
    mode = session.get("mode", "Research Explorer")
    meta = MODE_META.get(mode, {})
    title = meta.get("title", mode)
    subtitle = meta.get("subtitle", "")
    status = meta.get("status", "")
    workspace_name = session.get("title", "New Workspace")
    st.markdown(
        f"""
        <div class="hero-card">
          <div class="eyebrow">AI Research Assistant Workspace</div>
          <div class="hero-title">{html.escape(title)}</div>
          <p class="hero-copy">{html.escape(subtitle)}</p>
          <span class="mode-chip">{html.escape(mode)}</span>
          <span class="mode-chip">{html.escape(status)}</span>
          <span class="mode-chip">{html.escape(workspace_name)}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_research_result(result: dict[str, Any]) -> None:
    """Render a Research Explorer response."""
    table = result.get("table") or []
    if table:
        st.markdown("#### Result Table")
        PaperTableRenderer.render(table)

    gaps = result.get("research_gaps") or []
    if gaps:
        st.markdown("#### Research Gaps")
        BulletListRenderer.render(gaps, max_chars=360)

    idea = result.get("generated_idea")
    if idea:
        IdeaCardRenderer.render(idea)

    steps = result.get("generated_idea_steps") or []
    if steps:
        st.markdown("#### Implementation Steps")
        NumberedStepsRenderer.render(steps, max_chars=320)

    if result.get("assistant_reply"):
        st.caption(str(result["assistant_reply"]))


def render_assistant_result(result: dict[str, Any]) -> None:
    """Render a generic assistant answer with optional sources."""
    source = result.get("answer_source")
    if source == "vectordb":
        st.caption("Source: Local VectorDB")
    elif source == "external_search":
        label = "Source: External search"
        if result.get("incremental_learning_started"):
            label += " | incremental learning started in background"
        st.caption(label)

    st.write(result.get("answer") or "No answer found.")
    sources = result.get("sources") or []
    if sources:
        st.markdown("**Sources**")
        for item in sources:
            title = item.get("title", "Untitled")
            url = item.get("url", "")
            snippet = item.get("snippet", "")
            if url:
                st.markdown(f"- [{title}]({url})")
            else:
                st.write(f"- {title}")
            if snippet:
                st.caption(snippet)


def render_message(msg: dict[str, Any]) -> None:
    """Render one chat message."""
    if msg.get("role") == "user":
        render_user_message(msg)
        return

    st.markdown('<div class="assistant-panel">', unsafe_allow_html=True)
    st.markdown('<div class="assistant-label">Assistant</div>', unsafe_allow_html=True)
    if msg.get("type") == "loading":
        st.write(msg.get("display_text") or "Loading...")
    elif msg.get("type") == "assistant":
        render_assistant_result(msg["content"])
    elif msg.get("type") == "research":
        render_research_result(msg["content"])
    else:
        st.write(msg.get("display_text") or msg.get("content") or "")
    st.markdown("</div>", unsafe_allow_html=True)


def render_user_message(msg: dict[str, Any]) -> None:
    """Render the user chat bubble."""
    text = html.escape(msg.get("display_text") or msg.get("content") or "")
    st.markdown(
        f"""
        <div class="user-message-wrap">
          <div class="user-message-card">{text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar() -> None:
    """Render the left sidebar and handle workspace switching."""
    with st.sidebar:
        st.markdown("### AI Research Assistant")
        st.caption("Small modules, reusable helpers, and a simpler beginner-friendly layout.")

        if st.button("New Workspace", width="stretch"):
            new_chat(MODES[0])
            st.rerun()

        session = current_session()
        mode = st.selectbox("Mode", MODES, index=MODES.index(session["mode"]))
        if mode != session["mode"]:
            if session["messages"]:
                new_chat(mode)
            else:
                update_current_session(mode=mode, paper_text="", writer_state={"phase": "start"}, writer_intro_shown=False)
            st.rerun()

        st.markdown("#### Workspaces")
        for session_item in st.session_state.sessions:
            button_type = "primary" if session_item["id"] == st.session_state.current_session_id else "secondary"
            if st.button(session_item["title"], key=f"session-{session_item['id']}", use_container_width=True, type=button_type):
                st.session_state.current_session_id = session_item["id"]
                st.rerun()
            st.caption(session_item["mode"])
        st.markdown("---")
        st.caption("All rights reserved by Neamul Islam Fahim")


def render_reviewer_panel(session: dict[str, Any], on_process_upload) -> None:
    """Render the PDF uploader panel for reviewer mode."""
    st.markdown(
        """
        <div class="panel-card">
          <div class="section-title">Reviewer Tools</div>
          <div class="muted-copy">Upload a PDF once and it will be processed automatically for a structured review, then use the conversation below for follow-up questions.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    uploaded_file = st.file_uploader("Upload PDF for review", type=["pdf"], key=f"upload-{session['id']}")
    if uploaded_file is not None:
        processed = on_process_upload(uploaded_file)
        if processed:
            st.rerun()
    if session.get("paper_text"):
        st.success("PDF loaded and ready for questions.")
    elif uploaded_file is not None:
        st.info("Processing uploaded PDF...")
    else:
        st.info("No PDF loaded yet.")
