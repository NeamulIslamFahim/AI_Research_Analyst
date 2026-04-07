"""Tabular UI components for research result rendering."""

from __future__ import annotations

import html
from typing import Any

import streamlit as st

from ..helpers import safe_paper_url


class PaperTableRenderer:
    @staticmethod
    def render(table: list[dict[str, Any]]) -> None:
        html_rows: list[str] = []
        for row in table:
            title = row.get("paper_name", "Untitled")
            paper_url = safe_paper_url(row.get("paper_url", ""), title)
            authors = str(row.get("authors_name", "") or "")
            summary = str(row.get("summary_full_paper", "") or "")
            approach = str(row.get("proposed_model_or_approach", "") or "")
            source = str(row.get("source", "") or "")

            paper_link = (
                f'<a href="{html.escape(paper_url, quote=True)}" target="_blank" rel="noopener noreferrer">Open paper</a>'
                if paper_url
                else ""
            )

            html_rows.append(
                "".join(
                    [
                        "<tr>",
                        f'<td class="paper-col">{html.escape(title)}</td>',
                        f'<td class="url-col">{paper_link}</td>',
                        f'<td class="authors-col">{html.escape(authors)}</td>',
                        f'<td class="summary-col">{html.escape(summary)}</td>',
                        f'<td class="approach-col">{html.escape(approach)}</td>',
                        f'<td class="source-col">{html.escape(source)}</td>',
                        "</tr>",
                    ]
                )
            )

        table_html = """
            <div class="result-table-scroll" tabindex="0">
              <table class="result-table">
                <thead>
                  <tr>
                    <th class="paper-col">Paper Name</th>
                    <th class="url-col">Paper URL</th>
                    <th class="authors-col">Authors</th>
                    <th class="summary-col">Summary</th>
                    <th class="approach-col">Approach</th>
                    <th class="source-col">Source</th>
                  </tr>
                </thead>
                <tbody>
        """
        table_html += "".join(html_rows)
        table_html += """
                </tbody>
              </table>
            </div>
        """
        st.markdown(table_html, unsafe_allow_html=True)
