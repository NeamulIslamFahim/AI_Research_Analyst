"""Shared response templates for consistent safe fallback messages."""

from __future__ import annotations

from typing import Any


def build_research_error_response(detail: str) -> dict[str, Any]:
    return {
        "table": [],
        "research_gaps": [],
        "assistant_reply": (
            "I couldn't build a full paper comparison right now, so I’m giving you a broad starting point instead.\n\n"
            f"Details: {detail}"
        ),
        "generated_idea": (
            "Start with a broader query, inspect the strongest sources, then narrow to one concrete research question."
        ),
        "generated_idea_steps": [
            "Broaden the topic phrasing slightly.",
            "Check the top papers and note the common theme.",
            "Pick one specific subproblem, dataset, or language.",
            "Run the query again with that narrower focus.",
        ],
        "generated_idea_citations": [],
        "error_recovered": True,
    }


def build_reviewer_error_response(detail: str) -> dict[str, Any]:
    return {
        "strengths": "The paper addresses a relevant research problem, but the review could not be completed automatically.",
        "weaknesses": f"Reviewer fallback activated: {detail}",
        "novelty": "The available evidence is insufficient to assess novelty confidently.",
        "technical_correctness": "The automated path could not verify the method details reliably.",
        "reproducibility": "More details about the dataset, setup, and implementation are needed.",
        "recommendation": "Major Revision",
        "suggested_venue": "Conference",
    }

