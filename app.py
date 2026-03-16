"""Streamlit UI for the AI Research Assistant."""

from __future__ import annotations

import streamlit as st

from helpers import safe_get, strip_html
from main import run_paper_reviewer, run_reference_generator, run_research_explorer, download_papers_for_topic


st.set_page_config(page_title="AI Research Assistant", layout="wide")

st.title("AI Research Assistant")

st.markdown(
    """
    <style>
    /* ChatGPT-like layout */
    .block-container { max-width: 900px; }
    .stDataFrame [data-testid="stDataFrameResizable"] div {
        white-space: normal !important;
        word-break: break-word !important;
    }
    /* Chat message styling */
    [data-testid="stChatMessage"] {
        padding: 12px 16px;
        border-radius: 12px;
        margin-bottom: 10px;
    }
    [data-testid="stChatMessage"][data-testid*="user"] {
        background: #f3f4f6;
    }
    [data-testid="stChatMessage"][data-testid*="assistant"] {
        background: #ffffff;
        border: 1px solid #e5e7eb;
    }
    /* Icon-only buttons inside chat bubbles (hover-only) */
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
        padding: 2px 6px;
        font-size: 12px;
        line-height: 1;
        background: transparent;
        border: none;
    }
    [data-testid="stChatMessage"] [data-testid="stButton"] button:hover {
        background: #f3f4f6;
        border-radius: 6px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.sidebar.header("Chat Commands")
st.sidebar.write("Use `/review` to analyze the uploaded PDF.")
st.sidebar.write("Use `/refs <topic>` to generate 10 APA references.")
st.sidebar.write("Use `/help` to see this list again.")
uploaded_pdf = st.sidebar.file_uploader("Upload a PDF for /review", type=["pdf"])
if uploaded_pdf is not None:
    st.session_state["uploaded_pdf"] = uploaded_pdf

st.write(
    "Ask a research question below. The assistant will retrieve papers and return a structured table, "
    "followed by a research gap and a generated idea."
)


def _format_chat_history(max_turns: int = 12) -> str:
    """Format recent chat history into a compact text context."""
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


def render_user_actions(idx: int) -> None:
    """Render edit/regenerate controls for a user message."""
    cols = st.columns([8, 1, 1])
    with cols[1]:
        if st.button("🖊️", key=f"edit_{idx}"):
            st.session_state.edit_mode = True
            st.session_state.edit_index = idx
            st.rerun()
    with cols[2]:
        if st.button("🔁", key=f"regen_{idx}"):
            user_text = st.session_state.chat_history[idx]["content"]
            with st.spinner("Regenerating response..."):
                history_text = _format_chat_history()
                result = run_research_explorer(user_text, chat_history=history_text)
            if "error" in result:
                st.error(result["error"])
                st.session_state.chat_history.append({"role": "assistant", "content": result["error"]})
            else:
                st.session_state.chat_history.append({"role": "assistant", "content": result})
                _rebuild_memory()
            st.rerun()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "show_review_uploader" not in st.session_state:
    st.session_state.show_review_uploader = False
if "memory" not in st.session_state:
    st.session_state.memory = []
if "edit_mode" not in st.session_state:
    st.session_state.edit_mode = False
if "edit_index" not in st.session_state:
    st.session_state.edit_index = None

for idx, msg in enumerate(st.session_state.chat_history):
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant" and isinstance(msg.get("content"), dict):
            result = msg["content"]
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
                    st.write(f"- {strip_html(g)}")
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
        else:
            if msg["role"] == "user" and st.session_state.edit_mode and st.session_state.edit_index == idx:
                edited = st.text_area(
                    "Edit message",
                    value=msg["content"],
                    height=80,
                    key=f"edit_textarea_{idx}",
                    label_visibility="collapsed",
                )
                cols = st.columns([8, 1, 1])
                with cols[1]:
                    if st.button("✅", key=f"save_edit_{idx}"):
                        # Replace this message and regenerate in-place (no new chat)
                        st.session_state.edit_mode = False
                        st.session_state.edit_index = None
                        st.session_state.chat_history = st.session_state.chat_history[: idx + 1]
                        st.session_state.chat_history[idx]["content"] = edited
                        with st.spinner("Regenerating response..."):
                            history_text = _format_chat_history()
                            result = run_research_explorer(edited, chat_history=history_text)
                        if "error" in result:
                            st.error(result["error"])
                            st.session_state.chat_history.append({"role": "assistant", "content": result["error"]})
                        else:
                            st.session_state.chat_history.append({"role": "assistant", "content": result})
                            _rebuild_memory()
                with cols[2]:
                    if st.button("✖", key=f"cancel_edit_{idx}"):
                        st.session_state.edit_mode = False
                        st.session_state.edit_index = None
            else:
                st.write(msg["content"])
                if msg["role"] == "user":
                    render_user_actions(idx)

# Inline edit handled inside the chat loop (ChatGPT-like)

prompt = st.chat_input("Enter a research topic or command (/review, /refs <topic>)")
if prompt:
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    st.session_state.memory.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)
        render_user_actions(len(st.session_state.chat_history) - 1)

    with st.chat_message("assistant"):
        user_text = prompt.strip()
        if user_text.lower().startswith("/help"):
            st.write("Commands: /review, /refs <topic>")
            st.session_state.chat_history.append(
                {"role": "assistant", "content": "Commands: /review, /refs <topic>"}
            )
            st.session_state.memory.append({"role": "assistant", "content": "Commands: /review, /refs <topic>"})
        elif user_text.lower().startswith("/review"):
            pdf_file = st.session_state.get("uploaded_pdf")
            if pdf_file is None:
                st.session_state.show_review_uploader = True
                st.error("Please upload a PDF below and then run /review again.")
                st.session_state.chat_history.append(
                    {"role": "assistant", "content": "Please upload a PDF below."}
                )
            else:
                with st.spinner("Analyzing PDF..."):
                    result = run_paper_reviewer(pdf_file)
                if "error" in result:
                    st.error(result["error"])
                    st.session_state.chat_history.append(
                        {"role": "assistant", "content": result["error"]}
                    )
                else:
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
                    st.session_state.chat_history.append({"role": "assistant", "content": result})
                    st.session_state.memory.append({"role": "assistant", "content": "Reviewed the uploaded paper."})
        elif user_text.lower().startswith("/refs"):
            parts = user_text.split(maxsplit=1)
            if len(parts) < 2 or not parts[1].strip():
                st.error("Usage: /refs <topic>")
                st.session_state.chat_history.append(
                    {"role": "assistant", "content": "Usage: /refs <topic>"}
                )
            else:
                topic = parts[1].strip()
                with st.spinner("Generating references..."):
                    result = run_reference_generator(topic)
                if "error" in result:
                    st.error(result["error"])
                    st.session_state.chat_history.append(
                        {"role": "assistant", "content": result["error"]}
                    )
                else:
                    st.markdown("**APA References**")
                    references = result.get("references", [])
                    if isinstance(references, list):
                        for ref in references:
                            st.write(ref)
                    else:
                        st.write(references)
                    st.session_state.chat_history.append({"role": "assistant", "content": result})
                    st.session_state.memory.append(
                        {"role": "assistant", "content": f"Generated references for: {topic}"}
                    )
        else:
            with st.spinner("Retrieving papers and generating analysis..."):
                history_text = _format_chat_history()
                result = run_research_explorer(user_text, chat_history=history_text)
            if "error" in result:
                st.error(result["error"])
                st.session_state.chat_history.append(
                    {"role": "assistant", "content": result["error"]}
                )
            else:
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

                st.markdown("**Research Gaps**")
                gaps = safe_get(result, "research_gaps", [])
                if isinstance(gaps, list) and gaps:
                    for g in gaps:
                        st.write(f"- {strip_html(g)}")
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

# Inline uploader for /review when requested
if st.session_state.show_review_uploader:
    st.markdown("**Upload PDF for /review**")
    chat_pdf = st.file_uploader("Choose a PDF file", type=["pdf"], key="chat_review_uploader")
    if chat_pdf is not None:
        st.session_state["uploaded_pdf"] = chat_pdf
        st.session_state.show_review_uploader = False
