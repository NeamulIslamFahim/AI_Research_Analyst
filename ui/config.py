"""Configuration and shared styling for the Streamlit UI."""

from __future__ import annotations

import os

import streamlit as st
from dotenv import load_dotenv


load_dotenv()

ASSISTANT_ONLY = (os.getenv("ASSISTANT_ONLY", "false") or "false").lower() == "true"
MODES = ["Research Explorer"] if ASSISTANT_ONLY else [
    "Research Explorer",
    "Research Paper Reviewer",
    "Research Paper Writer",
]

MODE_META = {
    "Research Explorer": {
        "title": "Research Explorer",
        "subtitle": "Search scholarly sources, compare papers, surface gaps, and shape a grounded research direction.",
        "prompt": "Ask about a topic, method, dataset, or literature gap...",
        "status": "Multi-source retrieval",
    },
    "Research Paper Reviewer": {
        "title": "Paper Reviewer",
        "subtitle": "Upload a PDF, generate a structured review, and ask follow-up questions grounded in the paper.",
        "prompt": "Ask a question about the uploaded paper...",
        "status": "PDF-grounded analysis",
    },
    "Research Paper Writer": {
        "title": "Paper Writer",
        "subtitle": "Move through a guided academic writing workflow for framing, sections, and drafting.",
        "prompt": "Continue the writing workflow...",
        "status": "Guided drafting",
    },
}


def setup_page() -> None:
    """Configure the Streamlit page and inject shared CSS."""
    st.set_page_config(
        page_title="AI Research Assistant",
        page_icon="A",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(211, 107, 51, 0.18), transparent 26%),
                radial-gradient(circle at top right, rgba(34, 122, 112, 0.14), transparent 24%),
                linear-gradient(180deg, #f8f1e6 0%, #f1eadf 100%);
        }
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #17231e 0%, #23372d 100%);
            border-right: 1px solid rgba(255, 255, 255, 0.08);
        }
        [data-testid="stSidebar"] > div:first-child {
            display: flex;
            flex-direction: column;
            height: auto;
            max-height: none;
        }
        [data-testid="stSidebar"] > div:first-child > div[data-testid="stVerticalBlock"] {
            flex-grow: 1;
            overflow-y: visible;
        }
        [data-testid="stSidebar"] * {
            color: #f7f2e8;
        }
        [data-testid="stSidebar"] .stButton > button {
            border-radius: 14px;
            border: 1px solid rgba(255, 255, 255, 0.12);
            background: rgba(255, 255, 255, 0.08);
        }
        [data-testid="stSidebar"] [data-baseweb="select"] > div {
            background: rgba(255, 250, 242, 0.96);
            color: #1f2d25;
            border-radius: 14px;
        }
        [data-testid="stSidebar"] [data-baseweb="select"] div {
            color: #1f2d25 !important;
        }
        [data-testid="stSidebar"] [data-baseweb="select"] span {
            color: #1f2d25 !important;
        }
        [data-testid="stSidebar"] [role="listbox"] *,
        [data-testid="stSidebar"] [role="option"] * {
            color: #1f2d25 !important;
        }
        [data-testid="stSidebar"] [data-baseweb="select"] svg {
            fill: #1f2d25;
        }
        .hero-card,
        .panel-card,
        .assistant-panel,
        .result-card {
            background: rgba(255, 251, 245, 0.88);
            border: 1px solid rgba(31, 47, 40, 0.10);
            border-radius: 24px;
            box-shadow: 0 20px 50px rgba(31, 47, 40, 0.08);
            backdrop-filter: blur(12px);
        }
        .hero-card {
            padding: 1.35rem 1.4rem;
            margin-bottom: 1rem;
        }
        .panel-card,
        .assistant-panel,
        .result-card {
            padding: 1rem 1.1rem;
        }
        .eyebrow {
            text-transform: uppercase;
            letter-spacing: 0.12em;
            color: #68756b;
            font-size: 0.72rem;
            margin-bottom: 0.4rem;
        }
        .hero-title {
            margin: 0 0 0.45rem 0;
            font-size: 2.1rem;
            line-height: 1.05;
            color: #1b2a22;
        }
        .hero-copy {
            margin: 0;
            line-height: 1.65;
            color: #546257;
            max-width: 60rem;
        }
        .mode-chip {
            display: inline-block;
            margin-top: 0.9rem;
            margin-right: 0.45rem;
            padding: 0.42rem 0.75rem;
            border-radius: 999px;
            background: rgba(255, 255, 255, 0.72);
            border: 1px solid rgba(31, 47, 40, 0.08);
            color: #324238;
            font-size: 0.82rem;
        }
        .section-title {
            font-size: 1rem;
            font-weight: 700;
            color: #1e3026;
            margin-bottom: 0.55rem;
        }
        .muted-copy {
            color: #58665c;
            line-height: 1.6;
        }
        .user-message-wrap {
            display: flex;
            justify-content: flex-end;
            margin: 0.25rem 0 0.45rem 0;
        }
        .user-message-card {
            max-width: 82%;
            background: linear-gradient(135deg, #203129 0%, #2b4438 100%);
            color: #fffaf2;
            border-radius: 22px 22px 8px 22px;
            padding: 0.9rem 1rem;
            line-height: 1.6;
            white-space: pre-wrap;
            word-break: break-word;
            box-shadow: 0 16px 34px rgba(32, 49, 41, 0.18);
        }
        .assistant-label {
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: #6a766d;
            font-size: 0.76rem;
            margin-bottom: 0.45rem;
        }
        .stTextInput input,
        .stTextArea textarea,
        div[data-testid="stChatInput"] {
            border-radius: 16px;
        }
        .result-table-scroll {
            width: 100%;
            overflow-x: auto;
            overflow-y: visible;
            -webkit-overflow-scrolling: touch;
            touch-action: pan-x pan-y;
            border: 1px solid rgba(31, 47, 40, 0.10);
            border-radius: 20px;
            background: rgba(255, 251, 245, 0.82);
            box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.5);
        }
        .result-table {
            width: 100%;
            min-width: 1200px;
            border-collapse: separate;
            border-spacing: 0;
        }
        .result-table thead th {
            position: sticky;
            top: 0;
            z-index: 1;
            background: #f3eadc;
            color: #213228;
            text-align: left;
            font-weight: 700;
        }
        .result-table th,
        .result-table td {
            padding: 0.9rem 0.95rem;
            border-bottom: 1px solid rgba(31, 47, 40, 0.08);
            vertical-align: top;
            white-space: normal;
            word-break: break-word;
            overflow-wrap: anywhere;
            line-height: 1.55;
            color: #24362b;
        }
        .result-table tbody tr:nth-child(even) td {
            background: rgba(255, 255, 255, 0.42);
        }
        .result-table .paper-col {
            min-width: 240px;
        }
        .result-table .url-col {
            min-width: 120px;
        }
        .result-table .authors-col {
            min-width: 220px;
        }
        .result-table .summary-col,
        .result-table .approach-col {
            min-width: 360px;
        }
        .result-table .source-col {
            min-width: 120px;
        }
        .result-table a {
            color: #0f6d64;
            text-decoration: none;
            font-weight: 600;
        }
        .result-table a:hover {
            text-decoration: underline;
        }
        @media (max-width: 768px) {
            .result-table {
                min-width: 1380px;
            }
            .result-table th,
            .result-table td {
                padding: 0.8rem 0.85rem;
                font-size: 0.95rem;
            }
            .result-table .summary-col,
            .result-table .approach-col {
                min-width: 420px;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
