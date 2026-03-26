"""Streamlit UI for the AI Research Assistant."""

from __future__ import annotations

import time
import os
import tempfile
from datetime import datetime, timezone
import hashlib
import streamlit as st

from helpers import safe_get, strip_html, load_env_var
from main import run_paper_reviewer, run_paper_qa, run_reference_generator, run_research_explorer, download_papers_for_topic
from pdf_utils import extract_text


st.set_page_config(page_title="AI Research Assistant", layout="wide", initial_sidebar_state="expanded")

st.title("AI Research Assistant")


def format_review_reply(review: dict) -> str:
    """Render a paper review dict into a chat-style reply."""
    if not isinstance(review, dict):
        return str(review)
    strengths = safe_get(review, "strengths", "")
    weaknesses = safe_get(review, "weaknesses", "")
    novelty = safe_get(review, "novelty", "")
    technical = safe_get(review, "technical_correctness", "")
    reproducibility = safe_get(review, "reproducibility", "")
    recommendation = safe_get(review, "recommendation", "")
    venue = safe_get(review, "suggested_venue", "")

    parts = [
        "Here is a structured peer review of the paper:",
        f"Strengths: {strengths}",
        f"Weaknesses: {weaknesses}",
        f"Novelty: {novelty}",
        f"Technical Correctness: {technical}",
        f"Reproducibility: {reproducibility}",
        f"Recommendation: {recommendation}",
        f"Suggested Venue: {venue}",
    ]
    return "\n\n".join([p for p in parts if p and not p.endswith(": ")])

# Initialize selected_mode if not present
if "selected_mode" not in st.session_state:
    st.session_state.selected_mode = "Research Explorer"

# Initialize other session state variables
if "uploaded_pdf" not in st.session_state:
    st.session_state.uploaded_pdf = None
if "uploaded_paper_text" not in st.session_state:
    st.session_state.uploaded_paper_text = None
if "review_file_hash" not in st.session_state:
    st.session_state.review_file_hash = None
if "review_result" not in st.session_state:
    st.session_state.review_result = None
if "writer_step" not in st.session_state:
    st.session_state.writer_step = 0
if "last_topic" not in st.session_state:
    st.session_state.last_topic = ""

# Mode selector at the top
mode_col1, mode_col2, mode_col3 = st.columns([1, 2, 1])
with mode_col2:
    mode_options = ["Research Explorer", "Research Paper Reviewer", "Research Paper Writer"]
    selected_mode = st.selectbox(
        "Select Mode",
        mode_options,
        index=mode_options.index(st.session_state.selected_mode),
        key="mode_selector",
        label_visibility="collapsed"
    )
    # If mode changed, start new chat only if current chat has messages
    if selected_mode != st.session_state.selected_mode:
        if len(st.session_state.chat_history) > 0:
            # Create new session only if there are messages
            new_id = f"chat-{len(st.session_state.sessions) + 1}"
            st.session_state.sessions[new_id] = {
                "title": f"New Chat ({selected_mode})",
                "chat_history": [],
                "memory": [],
                "created_at": time.time(),
            }
            st.session_state.current_session_id = new_id
        st.session_state.selected_mode = selected_mode
        # Reset mode-specific state
        st.session_state.writer_step = 0
        st.session_state.uploaded_pdf = None
        st.session_state.last_topic = ""
        st.rerun()

if st.session_state.selected_mode == "Research Paper Reviewer":
    st.header("Upload PDF for Review")
    uploaded_file = st.file_uploader("Upload PDF for review", type=["pdf"], key="main_uploader")
    if uploaded_file is None:
        st.session_state.uploaded_pdf = None
        st.session_state.uploaded_paper_text = None
        st.session_state.review_file_hash = None
        st.session_state.review_result = None
    else:
        st.session_state.uploaded_pdf = uploaded_file
        st.success(f"Uploaded: {uploaded_file.name}")
        file_bytes = uploaded_file.getvalue()
        file_hash = hashlib.md5(file_bytes).hexdigest()
        if st.session_state.review_file_hash != file_hash:
            st.session_state.review_file_hash = file_hash
            st.session_state.review_result = None
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(file_bytes)
                tmp_path = tmp.name
            
            try:
                with st.spinner("Analyzing PDF..."):
                    st.session_state.uploaded_paper_text = extract_text(tmp_path)
                    result = run_paper_reviewer(st.session_state.uploaded_paper_text)
                st.session_state.review_result = result
                if "error" in result:
                    st.error(result["error"])
                    st.session_state.chat_history.append({"role": "assistant", "content": result["error"]})
                else:
                    reply_text = format_review_reply(result)
                    st.session_state.chat_history.append({"role": "assistant", "content": reply_text})
                    st.session_state.memory.append({"role": "assistant", "content": "Reviewed the uploaded paper."})
            finally:
                os.remove(tmp_path)

st.markdown(
    """
    <style>
    /* ChatGPT-like layout */
    .block-container { max-width: 920px; padding-top: 1.5rem; }
    .stApp { background: #f7f7f8; color: #111827; font-family: "Söhne", "Inter", "Segoe UI", "Helvetica Neue", Arial, sans-serif; }
    footer { visibility: hidden; height: 0; }
    [data-testid="stSidebar"] { display: block !important; }
    section[data-testid="stSidebar"] { display: block !important; }
    [data-testid="stSidebar"] {
        background: #ffffff;
        border-right: 1px solid #e5e7eb;
        padding-top: 10px;
        font-family: "Söhne", "Inter", "Segoe UI", "Helvetica Neue", Arial, sans-serif;
    }
    [data-testid="stSidebar"] .stButton button {
        width: 100%;
        text-align: left;
        padding: 8px 10px;
        font-size: 13px;
        border-radius: 8px;
        background: transparent;
        border: 1px solid transparent;
    }
    [data-testid="stSidebar"] .stButton button:hover {
        background: #f3f4f6;
        border-color: #f3f4f6;
    }
    [data-testid="stSidebar"] .stCaption {
        font-size: 11px;
        color: #6b7280;
        margin: 10px 0 6px 6px;
        letter-spacing: 0.02em;
        text-transform: uppercase;
    }
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3, 
    [data-testid="stSidebar"] h4 {
        font-size: 13px !important;
        color: #6b7280 !important;
        font-weight: 600 !important;
        letter-spacing: 0.02em;
        text-transform: uppercase;
        margin-top: 12px !important;
    }
    .stDataFrame [data-testid="stDataFrameResizable"] div {
        white-space: normal !important;
        word-break: break-word !important;
    }
    /* Chat message styling */
    [data-testid="stChatMessage"] {
        display: flex;
        width: 100% !important;
        padding: 14px 16px;
        border-radius: 12px;
        margin-bottom: 12px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.04);
    }
    [data-testid="stChatMessage"] > div {
        max-width: 780px;
        width: fit-content;
    }
    [data-testid="stChatMessage"] .stMarkdown,
    [data-testid="stChatMessage"] .stText,
    [data-testid="stChatMessage"] p,
    [data-testid="stChatMessage"] span {
        white-space: normal !important;
        word-break: break-word !important;
        overflow-wrap: anywhere !important;
    }
    [data-testid="stChatMessage"][data-testid*="user"] {
        justify-content: flex-end;
        background: #e5e7eb;
        border: 1px solid #e5e7eb;
    }
    [data-testid="stChatMessage"][data-testid*="assistant"] {
        justify-content: flex-start;
        background: #ffffff;
        border: 1px solid #e5e7eb;
    }
    /* Chat input bar */
    [data-testid="stChatInput"] {
        position: sticky;
        bottom: 0;
        background: #f7f7f8;
        padding: 16px 0 8px 0;
    }
    [data-testid="stChatInput"] > div {
        max-width: 920px;
        width: 100%;
        margin: 0 auto;
    }
    [data-testid="stChatInput"] textarea {
        width: 100% !important;
        min-height: 52px;
        font-size: 14px;
        line-height: 1.4;
        border-radius: 999px !important;
        border: 1px solid #d1d5db !important;
        padding: 12px 16px !important;
        background: #ffffff !important;
    }
    /* Hide spinner text, keep subtle inline spinner */
    [data-testid="stSpinner"] span { display: none !important; }
    /* Action buttons inside chat bubbles (hover-only, icon style) */
    [data-testid="stChatMessage"] [data-testid="stButton"] {
        opacity: 0 !important;
        visibility: hidden !important;
        transition: opacity 0.15s ease;
    }
    [data-testid="stChatMessage"]:hover [data-testid="stButton"],
    [data-testid="stChatMessage"]:hover [data-testid="stButton"] * {
        opacity: 1 !important;
        visibility: visible !important;
    }
    [data-testid="stChatMessage"] [data-testid="stButton"] button {
        padding: 4px 6px;
        font-size: 12px;
        line-height: 1;
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 6px;
        min-width: 28px;
    }
    [data-testid="stChatMessage"] [data-testid="stButton"] button:hover {
        background: #f3f4f6;
        border-radius: 6px;
    }
    /* Sidebar menu (⋯) hidden by default, shown on row hover or when open */
    [data-testid="stSidebar"] button[title="menu"],
    [data-testid="stSidebar"] button[aria-label="menu"] {
        opacity: 0 !important;
        visibility: hidden !important;
        transition: opacity 0.15s ease;
    }
    [data-testid="stSidebar"] [data-testid="stHorizontalBlock"]:hover button[title="menu"],
    [data-testid="stSidebar"] [data-testid="stHorizontalBlock"]:hover button[aria-label="menu"],
    [data-testid="stSidebar"] button[aria-expanded="true"][title="menu"],
    [data-testid="stSidebar"] button[aria-expanded="true"][aria-label="menu"] {
        opacity: 1 !important;
        visibility: visible !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

if "sessions" not in st.session_state:
    st.session_state.sessions = {}
if "current_session_id" not in st.session_state:
    st.session_state.current_session_id = "chat-1"
if st.session_state.current_session_id not in st.session_state.sessions:
    st.session_state.sessions[st.session_state.current_session_id] = {
        "title": "New Chat",
        "chat_history": [],
        "memory": [],
        "created_at": time.time(),
    }
for sid, sess in st.session_state.sessions.items():
    if "created_at" not in sess:
        sess["created_at"] = time.time()

def _current_session() -> dict:
    return st.session_state.sessions[st.session_state.current_session_id]

st.sidebar.header("Chats")
new_row = st.sidebar.columns([7, 1])
with new_row[0]:
    if st.sidebar.button("New chat", key="new_chat_btn"):
        new_id = f"chat-{len(st.session_state.sessions) + 1}"
        st.session_state.sessions[new_id] = {
            "title": "New Chat",
            "chat_history": [],
            "memory": [],
            "created_at": time.time(),
        }
        st.session_state.current_session_id = new_id
        st.rerun()
with new_row[1]:
    with st.sidebar.popover("⋯", help="menu", use_container_width=True):
        if st.button("Start new chat", key="new_chat_menu_start"):
            new_id = f"chat-{len(st.session_state.sessions) + 1}"
            st.session_state.sessions[new_id] = {
                "title": "New Chat",
                "chat_history": [],
                "memory": [],
                "created_at": time.time(),
            }
            st.session_state.current_session_id = new_id
            st.rerun()
        if st.button("Rename current", key="new_chat_menu_rename"):
            st.session_state.rename_session_id = st.session_state.current_session_id
            st.rerun()
        if st.button("Delete current", key="new_chat_menu_delete"):
            st.session_state.delete_pending_id = st.session_state.current_session_id
            st.rerun()

if "rename_session_id" not in st.session_state:
    st.session_state.rename_session_id = None
if "delete_pending_id" not in st.session_state:
    st.session_state.delete_pending_id = None

def _is_today(ts: float) -> bool:
    return datetime.fromtimestamp(ts, tz=timezone.utc).date() == datetime.now(tz=timezone.utc).date()

sorted_sessions = sorted(
    st.session_state.sessions.items(),
    key=lambda kv: kv[1].get("created_at", 0),
    reverse=True,
)
today_sessions = [(sid, s) for sid, s in sorted_sessions if _is_today(s.get("created_at", 0))]
older_sessions = [(sid, s) for sid, s in sorted_sessions if not _is_today(s.get("created_at", 0))]

if today_sessions:
    st.sidebar.caption("Today")
for sid, session in today_sessions:
    title = session.get("title") or "New Chat"
    if st.session_state.rename_session_id == sid:
        row = st.sidebar.columns([6, 1])
        with row[0]:
            new_title = st.sidebar.text_input(
                "Rename",
                value=title,
                key=f"title_{sid}",
                label_visibility="collapsed",
            )
        with row[1]:
            if st.sidebar.button("Save", key=f"rename_save_{sid}"):
                session["title"] = new_title.strip() or "New Chat"
                st.session_state.rename_session_id = None
                st.session_state.delete_pending_id = None
                st.rerun()
        if st.sidebar.button("Cancel", key=f"rename_cancel_{sid}"):
            st.session_state.rename_session_id = None
            st.session_state.delete_pending_id = None
            st.rerun()
    else:
        row = st.sidebar.columns([7, 1])
        with row[0]:
            if st.sidebar.button(title, key=f"chat_select_{sid}"):
                st.session_state.current_session_id = sid
                st.session_state.delete_pending_id = None
                st.rerun()
        with row[1]:
            with st.sidebar.popover("⋯", help="menu", use_container_width=True):
                if st.button("Rename", key=f"menu_rename_{sid}"):
                    st.session_state.rename_session_id = sid
                    st.session_state.delete_pending_id = None
                    st.rerun()
                if st.button("Delete", key=f"menu_delete_{sid}"):
                    st.session_state.delete_pending_id = sid
                    st.rerun()
    if st.session_state.delete_pending_id == sid:
        confirm = st.sidebar.columns([3, 3, 4])
        with confirm[0]:
            if st.sidebar.button("Confirm", key=f"confirm_delete_{sid}"):
                st.session_state.sessions.pop(sid, None)
                if st.session_state.current_session_id == sid:
                    if st.session_state.sessions:
                        st.session_state.current_session_id = next(iter(st.session_state.sessions.keys()))
                    else:
                        new_id = "chat-1"
                        st.session_state.sessions[new_id] = {
                            "title": "New Chat",
                            "chat_history": [],
                            "memory": [],
                            "created_at": time.time(),
                        }
                        st.session_state.current_session_id = new_id
                st.session_state.rename_session_id = None
                st.session_state.delete_pending_id = None
                st.rerun()
        with confirm[1]:
            if st.sidebar.button("Cancel", key=f"cancel_delete_{sid}"):
                st.session_state.delete_pending_id = None
                st.rerun()

if older_sessions:
    st.sidebar.caption("Earlier")
for sid, session in older_sessions:
    title = session.get("title") or "New Chat"
    if st.session_state.rename_session_id == sid:
        row = st.sidebar.columns([6, 1])
        with row[0]:
            new_title = st.sidebar.text_input(
                "Rename",
                value=title,
                key=f"title_{sid}",
                label_visibility="collapsed",
            )
        with row[1]:
            if st.sidebar.button("Save", key=f"rename_save_{sid}"):
                session["title"] = new_title.strip() or "New Chat"
                st.session_state.rename_session_id = None
                st.session_state.delete_pending_id = None
                st.session_state.menu_open_id = None
                st.rerun()
        if st.sidebar.button("Cancel", key=f"rename_cancel_{sid}"):
            st.session_state.rename_session_id = None
            st.session_state.delete_pending_id = None
            st.session_state.menu_open_id = None
            st.rerun()
    else:
        row = st.sidebar.columns([7, 1])
        with row[0]:
            if st.sidebar.button(title, key=f"chat_select_{sid}"):
                st.session_state.current_session_id = sid
                st.session_state.delete_pending_id = None
                st.session_state.menu_open_id = None
                st.rerun()
        with row[1]:
            with st.sidebar.popover("⋯", help="menu", use_container_width=True):
                if st.button("Rename", key=f"menu_rename_{sid}"):
                    st.session_state.rename_session_id = sid
                    st.session_state.delete_pending_id = None
                    st.rerun()
                if st.button("Delete", key=f"menu_delete_{sid}"):
                    st.session_state.delete_pending_id = sid
                    st.rerun()
    if st.session_state.delete_pending_id == sid:
        confirm = st.sidebar.columns([3, 3, 4])
        with confirm[0]:
            if st.sidebar.button("Confirm", key=f"confirm_delete_{sid}"):
                st.session_state.sessions.pop(sid, None)
                if st.session_state.current_session_id == sid:
                    if st.session_state.sessions:
                        st.session_state.current_session_id = next(iter(st.session_state.sessions.keys()))
                    else:
                        new_id = "chat-1"
                        st.session_state.sessions[new_id] = {
                            "title": "New Chat",
                            "chat_history": [],
                            "memory": [],
                            "created_at": time.time(),
                        }
                        st.session_state.current_session_id = new_id
                st.session_state.rename_session_id = None
                st.session_state.delete_pending_id = None
                st.rerun()
        with confirm[1]:
            if st.sidebar.button("Cancel", key=f"cancel_delete_{sid}"):
                st.session_state.delete_pending_id = None
                st.rerun()

st.sidebar.header("Chat Commands")
st.sidebar.write("Use `/help` to see this list again.")



st.write(
    "Choose a mode above and start chatting. The assistant will help with research exploration, paper review, or writing."
)


def _format_chat_history(max_turns: int | None = None) -> str:
    """Format recent chat history into a compact text context."""
    if max_turns is None:
        max_turns = int(load_env_var("CONTEXT_WINDOW", "12") or "12")
    history = st.session_state.memory[-max_turns:]
    lines = []
    for msg in history:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if not content:
            continue
        lines.append(f"{role.upper()}: {content}")
    return "\n".join(lines)


def _rebuild_memory() -> None:
    """Rebuild memory from chat_history for continuous context."""
    memory = []
    for msg in st.session_state.chat_history:
        role = msg.get("role")
        content = msg.get("content")
        if role == "user":
            memory.append({"role": "user", "content": content})
        elif role == "assistant":
            if isinstance(content, dict):
                summary = safe_get(content, "generated_idea") or "Provided structured research results."
                memory.append({"role": "assistant", "content": summary})
            else:
                memory.append({"role": "assistant", "content": str(content)})
    st.session_state.memory = memory
    _current_session()["memory"] = memory


def _is_followup_prompt(text: str) -> bool:
    """Detect short follow-up prompts like 'more' or 'continue'."""
    t = text.strip().lower()
    if not t:
        return False
    follow_phrases = [
        "more",
        "continue",
        "tell me more",
        "more on it",
        "more on this",
        "more on that",
        "more research",
        "more research on it",
        "expand",
        "elaborate",
        "same topic",
        "same one",
        "go deeper",
        "go deeper on it",
        "add more",
    ]
    if t in follow_phrases:
        return True
    if len(t.split()) <= 4 and any(k in t for k in ["more", "continue", "expand", "elaborate", "same"]):
        return True
    return False


def render_user_actions(idx: int) -> None:
    """Render edit/regenerate controls for a user message."""
    cols = st.columns([8, 1, 1])
    with cols[1]:
        if st.button("Edit", key=f"edit_{idx}"):
            st.session_state.edit_mode = True
            st.session_state.edit_index = idx
            st.rerun()
    with cols[2]:
        if st.button("Regen", key=f"regen_{idx}"):
            user_text = st.session_state.chat_history[idx].get("effective_query") or st.session_state.chat_history[idx]["content"]
            with st.spinner("Loading"):
                if st.session_state.selected_mode == "Research Paper Reviewer":
                    if st.session_state.uploaded_paper_text:
                        result = run_paper_qa(
                            question=user_text,
                            paper_text=st.session_state.uploaded_paper_text,
                        )
                    else:
                        result = {"error": "Please upload a PDF first."}
                else:
                    history_text = _format_chat_history()
                    result = run_research_explorer(
                        user_text,
                        chat_history=history_text,
                        focus_topic=st.session_state.last_topic or None,
                    )
            if "error" in result:
                st.error(result["error"])
                new_msg = {"role": "assistant", "content": result["error"]}
            else:
                if st.session_state.selected_mode == "Research Paper Reviewer":
                    new_msg = {"role": "assistant", "content": result.get("answer", "No answer found.")}
                else:
                    new_msg = {"role": "assistant", "content": result}
            replaced = False
            for j in range(idx + 1, len(st.session_state.chat_history)):
                if st.session_state.chat_history[j]["role"] == "assistant":
                    st.session_state.chat_history[j] = new_msg
                    replaced = True
                    break
            if not replaced:
                st.session_state.chat_history.append(new_msg)
            _rebuild_memory()
            st.rerun()

if "show_review_uploader" not in st.session_state:
    st.session_state.show_review_uploader = False
if "edit_mode" not in st.session_state:
    st.session_state.edit_mode = False
if "edit_index" not in st.session_state:
    st.session_state.edit_index = None
if "last_topic" not in st.session_state:
    st.session_state.last_topic = ""

current = _current_session()
st.session_state.chat_history = current.get("chat_history", [])
st.session_state.memory = current.get("memory", [])

for idx, msg in enumerate(st.session_state.chat_history):
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant" and isinstance(msg.get("content"), dict):
            result = msg["content"]
            if "strengths" in result:
                st.markdown("**Strengths**")
                st.write(safe_get(result, "strengths"))
                st.markdown("**Weaknesses**")
                st.write(safe_get(result, "weaknesses"))
                st.markdown("**Novelty**")
                st.write(safe_get(result, "novelty"))
                st.markdown("**Technical Correctness**")
                st.write(safe_get(result, "technical_correctness"))
                st.markdown("**Reproducibility**")
                st.write(safe_get(result, "reproducibility"))
                st.markdown("**Recommendation**")
                st.write(safe_get(result, "recommendation"))
                st.markdown("**Suggested Venue**")
                st.write(safe_get(result, "suggested_venue"))
            else:
                assistant_reply = safe_get(result, "assistant_reply")
                if assistant_reply:
                    st.write(strip_html(assistant_reply))
                st.markdown("**Results Table**")
                table_rows = safe_get(result, "table", [])
                if isinstance(table_rows, list) and table_rows:
                    cleaned_rows = []
                    for row in table_rows:
                        if isinstance(row, dict):
                            cleaned_row = {
                                k: (strip_html(v) if isinstance(v, str) else v) for k, v in row.items()
                            }
                            cleaned_rows.append(cleaned_row)
                        else:
                            cleaned_rows.append(row)
                    st.dataframe(cleaned_rows, width="stretch")
                else:
                    st.write("No table rows returned.")

                warnings = safe_get(result, "warnings", [])
                if isinstance(warnings, list) and warnings:
                    st.warning(" | ".join(warnings))

                st.markdown("**Research Gaps (Per Paper)**")
                gaps = safe_get(result, "research_gaps", [])
                if isinstance(gaps, list) and gaps:
                    for g in gaps:
                        clean_gap = strip_html(g).replace("Gap not explicitly stated; inferred from abstract:", "").strip()
                        st.write(f"- {clean_gap}")
                else:
                    st.write("Not provided.")

                st.markdown("**Generated Idea to Solve the Gap**")
                idea_text = safe_get(result, "generated_idea")
                st.write(strip_html(idea_text) if idea_text else "Not provided.")
                steps = safe_get(result, "generated_idea_steps", [])
                if isinstance(steps, list) and steps:
                    st.markdown("**Implementation Steps**")
                    for step in steps:
                        st.write(f"- {strip_html(step)}")
                idea_cites = safe_get(result, "generated_idea_citations", [])
                if isinstance(idea_cites, list) and idea_cites:
                    st.caption("Cited papers: " + "; ".join(idea_cites))

                st.markdown("**Citations**")
                if isinstance(table_rows, list) and table_rows:
                    for row in table_rows:
                        title = row.get("paper_name", "")
                        url = row.get("paper_url", "")
                        if url:
                            st.markdown(f"[{strip_html(title)}]({url})")
                else:
                    st.write("No citations available.")
        elif msg["role"] == "user" and st.session_state.edit_mode and st.session_state.edit_index == idx:
            edited = st.text_area(
                "Edit message",
                value=msg["content"],
                height=80,
                key=f"edit_textarea_{idx}",
                label_visibility="collapsed",
            )
            cols = st.columns([8, 1, 1])
            with cols[1]:
                if st.button("Save", key=f"save_edit_{idx}"):
                    # Replace this message and regenerate in-place (no new chat)
                    st.session_state.edit_mode = False
                    st.session_state.edit_index = None
                    st.session_state.chat_history = st.session_state.chat_history[: idx + 1]
                    _current_session()["chat_history"] = st.session_state.chat_history
                    effective_query = edited
                    if _is_followup_prompt(edited) and st.session_state.last_topic:
                        effective_query = f"{st.session_state.last_topic} (continue with more detail)"
                    st.session_state.chat_history[idx]["content"] = edited
                    st.session_state.chat_history[idx]["effective_query"] = effective_query
                    if idx == 0:
                        title = edited.strip() or "New Chat"
                        if len(title) > 40:
                            title = title[:40] + "..."
                        _current_session()["title"] = title
                    with st.spinner(""):
                        if st.session_state.selected_mode == "Research Paper Reviewer":
                            if st.session_state.uploaded_paper_text:
                                result = run_paper_qa(
                                    question=effective_query,
                                    paper_text=st.session_state.uploaded_paper_text,
                                )
                            else:
                                result = {"error": "Please upload a PDF first."}
                        else:
                            history_text = _format_chat_history()
                            result = run_research_explorer(
                                effective_query,
                                chat_history=history_text,
                                focus_topic=st.session_state.last_topic or None,
                            )
                    if "error" in result:
                        st.error(result["error"])
                        st.session_state.chat_history.append({"role": "assistant", "content": result["error"]})
                    else:
                        if st.session_state.selected_mode == "Research Paper Reviewer":
                            st.session_state.chat_history.append(
                                {"role": "assistant", "content": result.get("answer", "No answer found.")}
                            )
                        else:
                            st.session_state.chat_history.append({"role": "assistant", "content": result})
                            _rebuild_memory()
            with cols[2]:
                if st.button("Cancel", key=f"cancel_edit_{idx}"):
                    st.session_state.edit_mode = False
                    st.session_state.edit_index = None
        else:
            st.write(msg["content"])
            if msg["role"] == "user":
                render_user_actions(idx)

# Inline edit handled inside the chat loop (ChatGPT-like)

# Chat input at the bottom
prompt = st.chat_input("Enter your message")

if prompt:
    user_prompt_raw = prompt.strip()
    effective_query = user_prompt_raw
    if _is_followup_prompt(user_prompt_raw) and st.session_state.last_topic:
        effective_query = f"{st.session_state.last_topic} (continue with more detail)"
    st.session_state.chat_history.append(
        {"role": "user", "content": user_prompt_raw, "effective_query": effective_query}
    )
    st.session_state.memory.append({"role": "user", "content": effective_query})
    if len(st.session_state.chat_history) == 1:
        title = user_prompt_raw.strip() or "New Chat"
        if len(title) > 40:
            title = title[:40] + "..."
        _current_session()["title"] = title
    with st.chat_message("user"):
        st.write(user_prompt_raw)
        render_user_actions(len(st.session_state.chat_history) - 1)

    with st.chat_message("assistant"):
        mode = st.session_state.selected_mode
        user_text = effective_query
        if user_text.lower().startswith("/help"):
            st.write("Select mode from dropdown. For Reviewer, upload PDF first.")
            st.session_state.chat_history.append(
                {"role": "assistant", "content": "Select mode from dropdown. For Reviewer, upload PDF first."}
            )
            st.session_state.memory.append({"role": "assistant", "content": "Select mode from dropdown. For Reviewer, upload PDF first."})
        elif mode == "Research Paper Reviewer":
            if st.session_state.uploaded_paper_text:
                with st.spinner("Answering question..."):
                    result = run_paper_qa(
                        question=user_text, paper_text=st.session_state.uploaded_paper_text
                    )
                if "error" in result:
                    st.error(result["error"])
                    st.session_state.chat_history.append(
                        {"role": "assistant", "content": result["error"]}
                    )
                else:
                    st.write(result.get("answer", "No answer found."))
                    st.session_state.chat_history.append(
                        {"role": "assistant", "content": result.get("answer", "No answer found.")}
                    )
            else:
                st.error("Please upload a PDF first.")
                st.session_state.chat_history.append(
                    {"role": "assistant", "content": "Please upload a PDF first."}
                )
        elif mode == "Research Paper Writer":
            if "writer_step" not in st.session_state:
                st.session_state.writer_step = 0
            if st.session_state.writer_step == 0:
                st.write("What type of paper? Conference or Journal? Please specify Q1, Q2, Q3.")
                st.session_state.chat_history.append({"role": "assistant", "content": "What type of paper? Conference or Journal? Please specify Q1, Q2, Q3."})
                st.session_state.memory.append({"role": "assistant", "content": "Asked for paper type."})
                st.session_state.writer_step = 1
            elif st.session_state.writer_step == 1:
                paper_type = user_text
                st.write(f"Paper type: {paper_type}. Please provide the paper name.")
                st.session_state.chat_history.append({"role": "assistant", "content": f"Paper type: {paper_type}. Please provide the paper name."})
                st.session_state.memory.append({"role": "assistant", "content": f"Set paper type to {paper_type}."})
                st.session_state.writer_step = 2
            elif st.session_state.writer_step == 2:
                paper_name = user_text
                st.write(f"Paper name: {paper_name}. Now writing the paper step by step based on standard outline.")
                st.write("Step 1: Introduction...")
                st.session_state.chat_history.append({"role": "assistant", "content": f"Starting to write paper: {paper_name}."})
                st.session_state.memory.append({"role": "assistant", "content": f"Writing paper: {paper_name}."})
                st.session_state.writer_step = 0
        else:
            with st.spinner("Retrieving papers and generating analysis..."):
                history_text = _format_chat_history()
                result = run_research_explorer(
                    user_text,
                    chat_history=history_text,
                    focus_topic=(st.session_state.last_topic if _is_followup_prompt(user_prompt_raw) else None),
                )
            if "error" in result:
                st.error(result["error"])
                st.session_state.chat_history.append(
                    {"role": "assistant", "content": result["error"]}
                )
            else:
                st.markdown("**Results Table**")
                assistant_reply = safe_get(result, "assistant_reply")
                if assistant_reply:
                    st.write(strip_html(assistant_reply))
                table_rows = safe_get(result, "table", [])
                if isinstance(table_rows, list) and table_rows:
                    cleaned_rows = []
                    for row in table_rows:
                        if isinstance(row, dict):
                            cleaned_row = {
                                k: (strip_html(v) if isinstance(v, str) else v) for k, v in row.items()
                            }
                            cleaned_rows.append(cleaned_row)
                        else:
                            cleaned_rows.append(row)
                    st.dataframe(cleaned_rows, width="stretch")
                else:
                    st.write("No table rows returned.")

                warnings = safe_get(result, "warnings", [])
                if isinstance(warnings, list) and warnings:
                    st.warning(" | ".join(warnings))

                st.markdown("**Research Gaps**")
                gaps = safe_get(result, "research_gaps", [])
                if isinstance(gaps, list) and gaps:
                    for g in gaps:
                        clean_gap = strip_html(g).replace("Gap not explicitly stated; inferred from abstract:", "").strip()
                        st.write(f"- {clean_gap}")
                else:
                    st.write("Not provided.")

                st.markdown("**Generated Idea to Solve the Gap**")
                idea_text = safe_get(result, "generated_idea")
                st.write(strip_html(idea_text) if idea_text else "Not provided.")
                steps = safe_get(result, "generated_idea_steps", [])
                if isinstance(steps, list) and steps:
                    st.markdown("**Implementation Steps**")
                    for step in steps:
                        st.write(f"- {strip_html(step)}")
                idea_cites = safe_get(result, "generated_idea_citations", [])
                if isinstance(idea_cites, list) and idea_cites:
                    st.caption("Cited papers: " + "; ".join(idea_cites))

                st.markdown("**Citations**")
                if isinstance(table_rows, list) and table_rows:
                    for row in table_rows:
                        title = row.get("paper_name", "")
                        url = row.get("paper_url", "")
                        if url:
                            st.markdown(f"[{strip_html(title)}]({url})")
                else:
                    st.write("No citations available.")

                st.session_state.chat_history.append({"role": "assistant", "content": result})
                if not _is_followup_prompt(user_prompt_raw):
                    st.session_state.last_topic = user_prompt_raw
