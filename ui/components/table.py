"""Tabular UI components for research result rendering."""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from ..helpers import safe_paper_url
from .text import TextPreviewer


class PaperTableRenderer:
    @staticmethod
    def render(table: list[dict[str, Any]]) -> None:
        display_rows = []
        for row in table:
            title = row.get("paper_name", "Untitled")
            display_rows.append(
                {
                    "Paper Name": title,
                    "Paper URL": safe_paper_url(row.get("paper_url", ""), title),
                    "Authors": TextPreviewer.preview(row.get("authors_name", ""), max_chars=140),
                    "Summary": TextPreviewer.preview(row.get("summary_full_paper", ""), max_chars=520),
                    "Approach": TextPreviewer.preview(row.get("proposed_model_or_approach", ""), max_chars=420),
                    "Source": row.get("source", ""),
                }
            )
        st.dataframe(
            pd.DataFrame(display_rows),
            width="stretch",
            hide_index=True,
            column_config={"Paper URL": st.column_config.LinkColumn("Paper URL", display_text="Open paper")},
        )

