"""Card and list UI components for research outputs."""

from __future__ import annotations

import html

import streamlit as st

from .text import TextPreviewer


class BulletListRenderer:
    @staticmethod
    def render(items: list[str], max_chars: int = 360) -> None:
        for item in items:
            st.markdown(f"- {TextPreviewer.preview(item, max_chars=max_chars)}")


class IdeaCardRenderer:
    @staticmethod
    def render(text: str) -> None:
        st.markdown(
            f"""
            <div class="result-card">
              <div class="section-title">New Idea</div>
              <div class="muted-copy">{html.escape(TextPreviewer.preview(text, max_chars=760))}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


class NumberedStepsRenderer:
    @staticmethod
    def render(steps: list[str], max_chars: int = 320) -> None:
        for idx, step in enumerate(steps, start=1):
            st.markdown(f"{idx}. {TextPreviewer.preview(step, max_chars=max_chars)}")

