"""Response composition helpers for research and review workflows."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from backend.explorer_utils import fix_paper_url
from backend.helpers import authors_to_str, extract_proposed_approach

from .text_utils import (
    clean_text,
    collapse_text,
    human_summary_from_text,
    normalize_output_text,
    row_matches_topic,
    topic_is_specific,
    topic_tokens,
)


@dataclass(slots=True)
class TopicPolicy:
    topic: str

    def tokens(self) -> list[str]:
        return topic_tokens(self.topic)

    def strong_tokens(self) -> list[str]:
        return [t for t in self.tokens() if len(t) >= 4]

    def is_specific(self) -> bool:
        return topic_is_specific(self.topic)

    def matches_row(self, row: dict) -> bool:
        return row_matches_topic(row, self.topic)


class ResearchResponseComposer:
    """Compose a short, readable research response without relying on the LLM."""

    def __init__(self, topic: str) -> None:
        self.topic = topic.strip()
        self.policy = TopicPolicy(self.topic)

    def select_rows(self, rows: list[dict], limit: int = 5) -> list[dict]:
        matched = [r for r in rows if self.policy.matches_row(r)]
        if self.policy.is_specific() and not matched:
            return []
        pool = matched or rows
        seen: set[tuple[str, str]] = set()
        selected: list[dict] = []
        for row in pool:
            title = str(row.get("title", "")).strip().lower()
            url = str(row.get("url", "")).strip().lower()
            key = (title, url)
            if key in seen:
                continue
            seen.add(key)
            selected.append(row)
            if len(selected) >= limit:
                break
        return selected

    def _summary(self, row: dict, fulltext_snippet: str = "", abstract: str = "") -> str:
        title = clean_text(row.get("title", "")) or "Untitled paper"
        text = clean_text(fulltext_snippet or abstract or row.get("abstract", ""))
        if text:
            summary = human_summary_from_text(text, title, max_chars=420) or text
            summary = summary.replace(title, "").strip(" :-â€“â€”")
            return collapse_text(summary or text, 420)
        return collapse_text(f"This record is centered on {title} and lacks a visible abstract.", 420)

    def _problem(self, row: dict, fulltext_snippet: str = "", abstract: str = "") -> str:
        title = clean_text(row.get("title", "")) or "Untitled paper"
        text = clean_text(fulltext_snippet or abstract or row.get("abstract", ""))
        if text:
            first = re.split(r"(?<=[.!?])\s+", text)[0].strip()
            return collapse_text(first, 360)
        return collapse_text(f"The paper addresses the topic suggested by its title: {title}.", 360)

    def _approach(self, row: dict, fulltext_snippet: str = "", abstract: str = "") -> str:
        title = clean_text(row.get("title", "")) or "Untitled paper"
        inferred = extract_proposed_approach(clean_text(fulltext_snippet or abstract or row.get("abstract", "")))
        if inferred:
            return collapse_text(inferred, 420)
        return collapse_text(
            f"The source metadata does not expose a distinct new method; the paper appears centered on {title}.",
            420,
        )

    def _topic_theme(self) -> str:
        blob = f"{self.topic} {self.topic}".lower()
        if "sentiment" in blob:
            return "sentiment analysis"
        if "phishing" in blob:
            return "phishing detection"
        if "space" in blob or "exploration" in blob:
            return "space exploration"
        if "analysis" in blob or "data" in blob:
            return "data analysis"
        return "the topic"

    def _gaps(self, rows: list[dict]) -> list[str]:
        gaps: list[str] = []
        for row in rows[:5]:
            title = clean_text(row.get("title", "")) or "Paper"
            abstract = clean_text(row.get("abstract", ""))
            blob = f"{title} {abstract}".lower()
            if "twitter" in blob or "social media" in blob:
                gap = f"{title}: test the approach on noisier social-media text and cross-domain data."
            elif "multilingual" in blob or "low-resource" in blob:
                gap = f"{title}: expand evaluation to more languages and report transfer clearly."
            elif "space" in blob or "planet" in blob:
                gap = f"{title}: connect the method to a concrete scientific or operational use case with clearer validation."
            else:
                gap = f"{title}: strengthen the evaluation with stronger baselines and clearer error analysis."
            gaps.append(collapse_text(gap, 360))
        return gaps

    def _idea(self, gaps: list[str]) -> str:
        theme = self._topic_theme()
        if not gaps:
            return f"Build a stronger benchmark for {theme} that tests robustness, transfer, and interpretability."
        gap_summary = " ".join(g.split(":", 1)[-1].strip() for g in gaps[:3])
        return collapse_text(
            f"Build a stronger {theme} pipeline that addresses the recurring weaknesses across these papers, especially {gap_summary}.",
            700,
        )

    def _implementation_steps(self, gaps: list[str], rows: list[dict], idea: str) -> list[str]:
        titles = [clean_text(row.get("paper_name", "")) for row in rows if clean_text(row.get("paper_name", ""))]
        primary_title = titles[0] if titles else self._topic_theme().title()
        secondary_title = titles[1] if len(titles) > 1 else primary_title

        gap_focus = []
        for gap in gaps[:3]:
            focus = gap.split(":", 1)[-1].strip() if ":" in gap else gap.strip()
            if focus:
                gap_focus.append(focus.rstrip("."))

        steps: list[str] = []
        steps.append(f"Use {primary_title} and {secondary_title} as the baseline comparison set.")
        if gap_focus:
            steps.append(f"Target the main weakness: {gap_focus[0]}.")
        else:
            steps.append("Target the main weakness shared across the selected papers.")

        if len(gap_focus) > 1:
            steps.append(f"Design the first experiment to check whether the method still works when {gap_focus[1]}.")
        else:
            steps.append("Design the first experiment to test the method on a harder or more diverse dataset.")

        steps.append("Add strong baselines from the same problem area and compare them under one shared evaluation setup.")
        steps.append("Run an ablation or sensitivity study so the contribution of each component is clear.")
        steps.append("Finish with an error analysis that explains where the approach fails and what the next refinement should be.")

        # Keep the steps concise and aligned with the generated idea instead of returning a fixed template.
        cleaned_steps = [collapse_text(step, 220) for step in steps if step]
        return cleaned_steps[:6]

    def build(self, rows: list[dict], fulltext_map: dict[tuple[str, str], str], fulltext_by_title: dict[str, str]) -> dict[str, Any]:
        selected = self.select_rows(rows, limit=5)
        if not selected:
            return self.build_insufficient()

        table: list[dict[str, Any]] = []
        for row in selected:
            title = row.get("title", "")
            fulltext_snippet = fulltext_map.get((title, row.get("url", ""))) or fulltext_by_title.get(title, "")
            abstract = clean_text(row.get("abstract", ""))
            table.append(
                {
                    "paper_name": title,
                    "paper_url": fix_paper_url(str(row.get("url", "") or row.get("pdf_url", "") or row.get("doi", ""))),
                    "authors_name": authors_to_str(row.get("authors", "")),
                    "summary_full_paper": self._summary(row, fulltext_snippet, abstract),
                    "problem_solved": self._problem(row, fulltext_snippet, abstract),
                    "proposed_model_or_approach": self._approach(row, fulltext_snippet, abstract),
                    "source": row.get("source", ""),
                    "score_relevance": 8,
                    "score_quality": 7,
                }
            )

        gaps = self._gaps(selected)
        idea = self._idea(gaps)
        return {
            "table": table,
            "research_gaps": gaps,
            "assistant_reply": (
                f"I found {len(table)} relevant papers on {self.topic}. "
                f"The common thread is {self._topic_theme()}, and the main open problems are "
                f"{' '.join(g.split(':', 1)[-1].strip() for g in gaps[:2])}. "
                "A good next step is to test whether these methods still hold up on harder, more diverse data."
            ),
            "generated_idea": idea,
            "generated_idea_steps": self._implementation_steps(gaps, table, idea),
            "generated_idea_citations": [row.get("paper_name", "") for row in table],
            "used_broader_fallback": True,
        }

    def build_insufficient(self) -> dict[str, Any]:
        return {
            "table": [],
            "research_gaps": [],
            "assistant_reply": (
                f"I could not find five closely relevant papers for '{self.topic}'. "
                "Try a narrower topic or more concrete domain terms so the retrieval can stay focused."
            ),
            "generated_idea": (
                "Narrow the topic, retrieve five close matches, then derive the research gaps from those papers before proposing a new idea."
            ),
            "generated_idea_steps": [
                f"Add one concrete keyword that narrows the topic to {self._topic_theme()}.",
                "Retrieve the top five close-match papers first.",
                "Filter out broad fallback results that do not match the topic.",
                "Generate the idea only after the paper set is clearly on-topic and comparable.",
            ],
            "generated_idea_citations": [],
            "used_broader_fallback": True,
        }


class ReviewResponseComposer:
    """Build review fallbacks when the LLM is unavailable."""

    def _extract_source_text(self, paper_text: str) -> str:
        text = " ".join(str(paper_text or "").split())
        if not text:
            return ""
        return text[:24000]

    def heuristic_review(self, paper_text: str) -> dict[str, str]:
        text = self._extract_source_text(paper_text)
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
        if not sentences:
            return {
                "strengths": "The paper addresses a relevant research problem.",
                "weaknesses": "The evidence is too limited to assess the method fully.",
                "novelty": "The paper seems to present an applied combination of existing techniques.",
                "technical_correctness": "The method appears plausible, but the validation details are not sufficient.",
                "reproducibility": "Reproducibility would be stronger with clearer dataset and training details.",
                "recommendation": "Major Revision",
                "suggested_venue": "Conference",
            }

        strengths = sentences[:2]
        weaknesses = sentences[2:4] if len(sentences) > 2 else sentences[:1]
        novelty = sentences[0]
        technical = sentences[1] if len(sentences) > 1 else sentences[0]
        reproducibility = sentences[2] if len(sentences) > 2 else "The paper should report more implementation detail."

        return {
            "strengths": collapse_text(" ".join(strengths), 280),
            "weaknesses": collapse_text(" ".join(weaknesses), 280),
            "novelty": collapse_text(novelty, 240),
            "technical_correctness": collapse_text(technical, 260),
            "reproducibility": collapse_text(reproducibility, 260),
            "recommendation": "Major Revision",
            "suggested_venue": "Conference",
        }

    def sanitize(self, review: dict[str, Any], paper_text: str) -> dict[str, Any]:
        cleaned = dict(review or {})
        fallback = self.heuristic_review(paper_text)
        for key, value in fallback.items():
            current = str(cleaned.get(key, "") or "").strip()
            if not current or len(current) < 20:
                cleaned[key] = value
        cleaned["suggested_venue"] = "Journal" if len(clean_text(paper_text).split()) > 5000 else "Conference"
        return cleaned
