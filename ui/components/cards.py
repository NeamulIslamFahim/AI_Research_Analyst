"""Card and list UI components for research outputs."""

from __future__ import annotations

import html

import streamlit as st


class BulletListRenderer:
    @staticmethod
    def render(items: list[str], max_chars: int = 360) -> None:
        for item in items:
            st.markdown(f"- {html.escape(str(item or ''))}", unsafe_allow_html=True)


class IdeaCardRenderer:
    @staticmethod
    def render(text: str) -> None:
        st.markdown(
            f"""
            <div class="result-card">
              <div class="section-title">New Idea</div>
              <div class="muted-copy">{html.escape(str(text or ''))}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


class NumberedStepsRenderer:
    @staticmethod
    def render(steps: list[str], max_chars: int = 320) -> None:
        for idx, step in enumerate(steps, start=1):
            st.markdown(f"{idx}. {html.escape(str(step or ''))}", unsafe_allow_html=True)
