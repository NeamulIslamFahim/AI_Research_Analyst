"""Response composition helpers for research and review workflows."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from backend.explorer_utils import fix_paper_url
from backend.helpers import authors_to_str, extract_datasets, extract_models, extract_proposed_approach

from .text_utils import (
    _unique_sentences,
    clean_text,
    collapse_text,
    normalize_output_text,
    row_matches_topic,
    strip_front_matter,
    titles_look_equivalent,
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

    def _ensure_sentence(self, text: str) -> str:
        text = re.sub(r"\s+", " ", str(text or "")).strip()
        if not text:
            return ""
        if text[-1] not in ".!?":
            text += "."
        return text

    def _lower_first(self, text: str) -> str:
        text = str(text or "").strip()
        if not text:
            return ""
        return text[0].lower() + text[1:] if text[0].isalpha() else text

    def _clean_fragment(self, text: str, title: str = "", max_words: int = 28) -> str:
        cleaned = re.sub(r"(\w)-\s+(\w)", r"\1\2", clean_text(text))
        if title:
            title_clean = re.escape(clean_text(title))
            cleaned = re.sub(rf"^{title_clean}\s*[:\-\u2013\u2014]?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.split(r"(?<=[.!?])\s+", cleaned, maxsplit=1)[0].strip(" ,;:-")
        cleaned = re.sub(
            r"^(?:this paper|the paper|this study|the study|the authors|authors|it)\s+"
            r"(?:is about|focuses on|addresses|examines|explores|studies|proposes|presents|introduces|develops|designs|uses|shows|reports|finds|suggests|argues that)\s+",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )
        words = cleaned.split()
        if len(words) > max_words:
            cleaned = " ".join(words[:max_words]).rstrip(" ,;:-")
        return cleaned.strip(" ,;:-")

    def _natural_list(self, items: list[str], limit: int = 3) -> str:
        cleaned_items: list[str] = []
        seen: set[str] = set()
        for item in items:
            cleaned = re.sub(r"\s+", " ", str(item or "")).strip(" ,;:-.")
            key = cleaned.lower()
            if not cleaned or key in seen:
                continue
            seen.add(key)
            cleaned_items.append(cleaned)
            if len(cleaned_items) >= limit:
                break
        if not cleaned_items:
            return ""
        if len(cleaned_items) == 1:
            return cleaned_items[0]
        if len(cleaned_items) == 2:
            return f"{cleaned_items[0]} and {cleaned_items[1]}"
        return f"{', '.join(cleaned_items[:-1])}, and {cleaned_items[-1]}"

    def select_rows(self, rows: list[dict], limit: int = 5, excluded_titles: list[str] | None = None) -> list[dict]:
        matched = [r for r in rows if self.policy.matches_row(r)]
        # Prefer the strict match set whenever it exists. Only fall back to the full
        # pool when no rows match at all, so follow-ups stay on-topic instead of
        # filling the table with unrelated papers.
        pool = matched if matched else rows
        excluded = [clean_text(title) for title in (excluded_titles or []) if clean_text(title)]
        seen_urls: set[str] = set()
        seen_titles: list[str] = []
        selected: list[dict] = []
        for row in pool:
            title = clean_text(row.get("title", ""))
            url = str(row.get("url", "") or row.get("pdf_url", "") or row.get("doi", "")).strip().lower()
            if excluded and any(titles_look_equivalent(title, old_title) for old_title in excluded):
                continue
            if url and url in seen_urls:
                continue
            if title and any(titles_look_equivalent(title, seen_title) for seen_title in seen_titles):
                continue
            selected.append(row)
            if url:
                seen_urls.add(url)
            if title:
                seen_titles.append(title)
            if len(selected) >= limit:
                break
        return selected

    def _problem(self, row: dict, fulltext_snippet: str = "", abstract: str = "") -> str:
        title = clean_text(row.get("title", "")) or "Untitled paper"
        text = clean_text(fulltext_snippet or abstract or row.get("abstract", ""))
        if not text:
            return collapse_text(f"The paper centers on the question suggested by its title, {title}.", 360)

        sentences = [
            piece.strip()
            for piece in re.split(r"(?<=[.!?])\s+", text)
            if len(piece.strip().split()) >= 6
        ]
        keywords = ["problem", "challenge", "gap", "need", "barrier", "issue", "limitation", "address", "improve", "support"]
        candidate = ""
        for sentence in sentences:
            lowered = sentence.lower()
            if any(keyword in lowered for keyword in keywords):
                candidate = sentence
                break
        if not candidate and sentences:
            candidate = sentences[0]

        fragment = self._clean_fragment(candidate or text, title=title, max_words=26)
        if not fragment:
            return collapse_text(f"The paper focuses on {title}.", 360)
        return collapse_text(self._ensure_sentence(f"The paper focuses on {self._lower_first(fragment)}"), 360)

    def _topic_theme(self) -> str:
        blob = f"{self.topic} {self.topic}".lower()
        if "generative ai" in blob or "genai" in blob:
            return "generative AI"
        if "sentiment" in blob:
            return "sentiment analysis"
        if "phishing" in blob:
            return "phishing detection"
        if "space" in blob or "exploration" in blob:
            return "space exploration"
        if "analysis" in blob or "data" in blob:
            return "data analysis"
        cleaned = clean_text(self.topic)
        if cleaned:
            return cleaned
        return "this topic"

    def _paper_label(self, title: str) -> str:
        label = clean_text(title) or "Paper"
        if ":" in label:
            prefix, suffix = [part.strip() for part in label.split(":", 1)]
            if prefix and suffix and len(prefix) <= 60:
                return prefix
        return label

    def _gap_focus(self, gap: str) -> str:
        focus = gap.rsplit(":", 1)[-1].strip() if ":" in gap else gap.strip()
        focus = re.sub(r"^(?:in|for)\s+.+?,\s*", "", focus, flags=re.IGNORECASE)
        focus = re.sub(
            r"^(?:the paper would benefit from|the safest next step is to|the next step would be to|the next step is to|a strong follow-up would)\s+",
            "",
            focus,
            flags=re.IGNORECASE,
        )
        focus = focus.rstrip(".")
        focus = re.sub(r"\s+", " ", focus).strip()
        return focus

    def _gaps(self, rows: list[dict]) -> list[str]:
        gaps: list[str] = []
        for idx, row in enumerate(rows[:5]):
            title = clean_text(row.get("title", "")) or "Paper"
            abstract = clean_text(row.get("abstract", ""))
            blob = f"{title} {abstract}".lower()
            if any(term in blob for term in ["twitter", "tweet", "microblog", "social media"]):
                gap = f"{title}: test the approach on noisier social-media text and cross-domain data."
            elif any(term in blob for term in ["students", "education", "higher education", "learning", "classroom"]):
                gap = f"{title}: test the claims on a larger and more diverse student sample, not just one institution."
            elif any(term in blob for term in ["work", "workplace", "organization", "enterprise", "employee"]):
                gap = f"{title}: evaluate the method in a real workflow and measure how it affects human-AI collaboration."
            elif any(term in blob for term in ["chatgpt", "llm", "large language model", "foundation model"]):
                gap = f"{title}: report stronger safety, hallucination, and robustness checks under prompt variation."
            elif any(term in blob for term in ["retrieval", "search", "genir", "retriever", "retrieval-augmented"]):
                gap = f"{title}: separate retrieval quality from generation quality and compare against stronger retrieval baselines."
            elif any(term in blob for term in ["survey", "review", "overview", "foundations", "framework"]):
                gap = f"{title}: the work is primarily descriptive; a valuable next step would be to add empirical comparisons or a benchmark-style evaluation."
            elif "multilingual" in blob or "low-resource" in blob:
                gap = f"{title}: the evaluation could be strengthened by expanding to more languages and reporting cross-lingual transfer performance."
            elif "space" in blob or "planet" in blob:
                gap = f"{title}: the method's practical value would be clearer if connected to a concrete scientific or operational use case with more direct validation."
            elif "generative ai" in blob or "genai" in blob:
                if idx == 0:
                    gap = f"{title}: the approach should be benchmarked against stronger baseline systems with a detailed task-level error analysis."
                elif idx == 1:
                    gap = f"{title}: the claims' robustness is unclear; testing across different user groups, prompts, or deployment settings is needed."
                elif idx == 2:
                    gap = f"{title}: the study needs robustness checks for hallucination, safety, and prompt sensitivity to be considered reliable."
                else:
                    gap = f"{title}: an ablation study is needed to clarify which components of the pipeline contribute most to the reported gains."
            else:
                gap = f"{title}: the evaluation lacks strong baselines and a clear error analysis, making it difficult to assess the method's true performance."
            gaps.append(collapse_text(gap, 360))
        return gaps

    def _idea(self, gaps: list[str]) -> str:
        theme = self._topic_theme()
        if not gaps:
            return f"Build a stronger benchmark for {theme} that tests robustness, transfer, and interpretability."
        gap_summary = " ".join(dict.fromkeys(self._gap_focus(g) for g in gaps[:3]))
        return collapse_text(f"A promising research direction is to develop a more robust {theme} pipeline that directly addresses the recurring weaknesses identified, such as {gap_summary}.", 700)

    def _implementation_steps(self, gaps: list[str], rows: list[dict], idea: str) -> list[str]:
        titles = [self._paper_label(row.get("paper_name", "")) for row in rows if clean_text(row.get("paper_name", ""))]
        primary_title = titles[0] if titles else self._topic_theme().title()
        secondary_title = titles[1] if len(titles) > 1 else primary_title

        gap_focus = list(dict.fromkeys(self._gap_focus(g) for g in gaps[:3] if self._gap_focus(g)))

        steps: list[str] = []
        if len(titles) >= 2:
            steps.append(f"Use {primary_title} and {secondary_title} as the baseline comparison set.")
        elif titles:
            steps.append(f"Use {primary_title} as the main comparison anchor.")
        else:
            steps.append(f"Use the selected papers as the baseline comparison set for {self._topic_theme()}.")
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

    def _metadata_limited(self, row: dict) -> bool:
        return len(clean_text(row.get("abstract", "")).split()) < 25

    def _domain_gap(self, row: dict) -> str:
        blob = f"{clean_text(row.get('title', ''))} {clean_text(row.get('abstract', ''))}".lower()
        if any(term in blob for term in ["protocol", "systematic review", "survey", "viewpoint", "framework", "overview"]):
            return (
                "the next step is to turn the conceptual contribution into a comparative empirical study with a shared evaluation setup, "
                "so readers can assess how the proposed ideas perform in practice"
            )
        if any(term in blob for term in ["chatbot", "llm", "chatgpt", "assistant", "conversational agent"]):
            return (
                "a strong follow-up would test the system in a real user workflow, measure sustained behavior change, and report safety, "
                "reliability, and human-AI interaction outcomes together"
            )
        if any(term in blob for term in ["diet", "nutrition", "physical activity", "exercise", "lifestyle"]):
            return (
                "the evidence would be stronger with longitudinal evaluation across more diverse populations, clearer adherence measures, "
                "and direct comparison against non-AI intervention baselines"
            )
        return (
            "the paper's conclusions would be more credible with stronger empirical comparisons, clearer evaluation criteria, and a more explicit analysis of where the approach succeeds or fails"
        )

    def _summary(self, row: dict, fulltext_snippet: str = "", abstract: str = "") -> str:
        title = clean_text(row.get("title", "")) or "Untitled paper"
        fulltext = clean_text(fulltext_snippet)
        abstract_text = clean_text(abstract or row.get("abstract", ""))
        source_text = fulltext or abstract_text
        if fulltext and abstract_text and (len(fulltext.split()) < 80 or len(abstract_text.split()) > len(fulltext.split()) * 1.2):
            source_text = abstract_text
        if not source_text:
            return collapse_text(
                (
                    f"While '{title}' appears relevant to {self._topic_theme()}, the available metadata is too limited to support a trustworthy paper-level summary. "
                    "The source does not clearly expose the problem setting, the concrete method, or the main findings, so this row should be treated as metadata-only evidence."
                ),
                1200,
            )

        def _sentences(text: str) -> list[str]:
            cleaned = re.sub(r"(\w)-\s+(\w)", r"\1\2", text.strip())
            cleaned = re.sub(r"\s+", " ", cleaned).strip()
            return [piece.strip() for piece in re.split(r"(?<=[.!?])\s+", cleaned) if len(piece.split()) >= 6]

        def _pick(sentences: list[str], keywords: list[str]) -> str:
            for sentence in sentences:
                lowered = sentence.lower()
                if any(keyword in lowered for keyword in keywords):
                    candidate = re.sub(r"\s+", " ", sentence).strip()
                    if len(candidate.split()) >= 6:
                        return candidate
            return ""

        sentence_list = _unique_sentences(_sentences(strip_front_matter(source_text, title) or source_text))
        if not sentence_list:
            sentence_list = _unique_sentences(_sentences(source_text))

        overview = sentence_list[0] if sentence_list else source_text
        problem = _pick(sentence_list, ["problem", "challenge", "gap", "need", "barrier", "promote", "improve", "support", "address"])
        approach = clean_text(extract_proposed_approach(source_text))
        if len(approach.split()) < 6:
            approach = _pick(sentence_list, ["propose", "present", "introduce", "framework", "model", "approach", "protocol", "review", "survey", "method"])
        impact = _pick(sentence_list, ["result", "find", "show", "suggest", "improve", "effective", "feasible", "impact", "outperform", "performance"])

        def _sentence_impact() -> str:
            impact_candidate = impact
            if not impact_candidate or impact_candidate.strip().lower() == problem.strip().lower():
                impact_candidate = _pick(sentence_list, ["guide", "future", "evaluation", "effective", "intervention", "support", "implication"])
            if impact_candidate:
                normalized = self._clean_fragment(impact_candidate, title=title, max_words=26)
                return self._ensure_sentence(f"The analysis suggests that {self._lower_first(normalized)}")
            return self._ensure_sentence("The paper frames its method as a meaningful contribution, though the visible text lacks precise quantitative results")

        def _sentence_close() -> str:
            if fulltext:
                return "Overall, the paper provides a grounded account of the problem, method, and reported outcomes."
            return "Overall, this summary is based on the abstract and metadata, not a full reconstruction of the paper."

        about_fragment = self._clean_fragment(overview or title, title=title, max_words=24)
        if not about_fragment:
            short_title = title.split(":", 1)[0].strip() or title
            about_sentence = self._ensure_sentence(f"The paper looks at {short_title}")
        elif len(about_fragment.split()) < 6:
            about_sentence = self._ensure_sentence(f"This paper investigates {self._lower_first(about_fragment)}")
        else:
            about_sentence = self._ensure_sentence(f"This paper investigates {self._lower_first(about_fragment)}")

        problem_fragment = self._clean_fragment(problem, title=title, max_words=26) if problem else ""
        if problem_fragment:
            problem_sentence = self._ensure_sentence(
                f"addressing the core challenge of {self._lower_first(problem_fragment)}"
            )
        else:
            problem_sentence = self._ensure_sentence(
                "addressing the core challenge described in the paper"
            )

        approach_fragment = self._clean_fragment(approach, title=title, max_words=28) if approach else ""
        about_key = re.sub(r"\s+", " ", about_fragment).strip().lower()
        problem_key = re.sub(r"\s+", " ", problem_fragment).strip().lower()
        approach_key = re.sub(r"\s+", " ", approach_fragment).strip().lower()
        if approach_fragment and approach_key not in {about_key, problem_key} and approach_key not in about_key:
            approach_sentence = self._ensure_sentence(
                f"by proposing a method centered on {self._lower_first(approach_fragment)}"
            )
        else:
            approach_sentence = ""

        parts = [
            f"{about_sentence.rstrip('.')} {problem_sentence.rstrip('.')} {approach_sentence.rstrip('.') if approach_sentence else ''}.".replace("  ", " ").strip(),
            _sentence_impact(),
            self._ensure_sentence(_sentence_close()),
        ]
        parts = [part for part in parts if part]
        final_summary = " ".join(parts)
        # Final cleanup for flow
        final_summary = final_summary.replace(" .", ".").replace("  ", " ")
        return collapse_text(final_summary, 1600 if fulltext else 1100)

    def _approach(self, row: dict, fulltext_snippet: str = "", abstract: str = "") -> str:
        title = clean_text(row.get("title", "")) or "Untitled paper"
        fulltext = clean_text(fulltext_snippet)
        abstract_text = clean_text(abstract or row.get("abstract", ""))
        source_text = fulltext or abstract_text
        if fulltext and abstract_text and len(abstract_text.split()) > len(fulltext.split()) * 1.2:
            source_text = abstract_text

        def _sentences(text: str) -> list[str]:
            cleaned = re.sub(r"(\w)-\s+(\w)", r"\1\2", text.strip())
            cleaned = re.sub(r"\s+", " ", cleaned).strip()
            return [piece.strip() for piece in re.split(r"(?<=[.!?])\s+", cleaned) if len(piece.split()) >= 6]

        def _pick(sentence_list: list[str], keywords: list[str]) -> str:
            for sentence in sentence_list:
                lowered = sentence.lower()
                if any(keyword in lowered for keyword in keywords):
                    return re.sub(r"\s+", " ", sentence).strip()
            return ""

        def _title_method_hint() -> str:
            short_title = title.split(":", 1)[0].strip() or title
            return re.sub(r"\s+", " ", short_title).strip(" ,;:-")

        if not source_text:
            title_hint = _title_method_hint()
            return collapse_text(
                " ".join(
                    _unique_sentences([
                        self._ensure_sentence(
                            f"The available metadata suggests a method centered on {self._lower_first(title_hint) if title_hint else title}"
                        ),
                        self._ensure_sentence(
                            "However, the visible record does not expose enough detail to reconstruct the full pipeline with confidence"
                        ),
                        self._ensure_sentence(
                            "The dataset, benchmark, and evaluation setup are also unclear from the metadata alone"
                        ),
                    ])
                ),
                1000,
            )

        method_text = strip_front_matter(source_text, title) or source_text
        sentence_list = _unique_sentences(_sentences(method_text))
        if not sentence_list:
            sentence_list = _unique_sentences(_sentences(source_text))

        model_hits = extract_models(f"{title} {method_text} {abstract_text}")
        dataset_hits = extract_datasets(f"{title} {method_text} {abstract_text}")
        explicit_approach = clean_text(extract_proposed_approach(method_text))
        cue_match = re.search(
            r"(?:this paper|the paper|this study|the study)\s+(?:proposes|presents|introduces|develops|designs|applies|uses)\s+(.*)",
            explicit_approach,
            flags=re.IGNORECASE,
        )
        if cue_match:
            explicit_approach = cue_match.group(1).strip(" ,;:-")
        explicit_approach = re.sub(
            r"^(?:we|this paper|the paper|this study|the study)\s+(?:propose|present|introduce|develop|design|apply|use|uses)\s+",
            "",
            explicit_approach,
            flags=re.IGNORECASE,
        ).strip(" ,;:-")
        if len(explicit_approach.split()) < 6:
            explicit_approach = _pick(sentence_list, ["propose", "present", "introduce", "framework", "model", "approach", "method", "pipeline", "protocol", "review", "survey"])
            cue_match = re.search(
                r"(?:this paper|the paper|this study|the study)\s+(?:proposes|presents|introduces|develops|designs|applies|uses)\s+(.*)",
                explicit_approach,
                flags=re.IGNORECASE,
            )
            if cue_match:
                explicit_approach = cue_match.group(1).strip(" ,;:-")
            explicit_approach = re.sub(
                r"^(?:we|this paper|the paper|this study|the study)\s+(?:propose|present|introduce|develop|design|apply|use|uses)\s+",
                "",
                explicit_approach,
                flags=re.IGNORECASE,
            ).strip(" ,;:-")
        setup_sentence = _pick(sentence_list, ["dataset", "data set", "corpus", "sample", "participants", "trial", "benchmark", "evaluation"])
        eval_sentence = _pick(sentence_list, ["evaluate", "evaluation", "experiment", "results", "performance", "accuracy", "effectiveness", "feasible"])

        survey_like = any(
            term in f"{title} {method_text} {abstract_text}".lower()
            for term in ["systematic review", "survey", "literature review", "scoping review", "meta-analysis", "viewpoint", "protocol"]
        )

        sentences_out: list[str] = []
        missing_sentences: list[str] = []
        if explicit_approach:
            normalized = self._clean_fragment(explicit_approach, title=title, max_words=30)
            sentences_out.append(self._ensure_sentence(f"The paper's core contribution is a method based on {self._lower_first(normalized)}"))
        elif survey_like:
            sentences_out.append(
                self._ensure_sentence(
                    "Rather than introducing a single predictive model, the paper mainly synthesizes prior work into a review or framework-style contribution"
                )
            )
        else:
            title_hint = _title_method_hint()
            sentences_out.append(
                self._ensure_sentence(
                    f"The methodology appears to be built around {self._lower_first(title_hint)}"
                )
            )
            missing_sentences.append(
                "The extracted text does not expose enough detail to restate every methodological step precisely"
            )

        if model_hits:
            normalized_explicit = explicit_approach.lower()
            if len(model_hits) == 1 and model_hits[0].lower() not in normalized_explicit:
                sentences_out.append(self._ensure_sentence(f"This approach is framed around {model_hits[0]}"))
            elif len(model_hits) > 1:
                model_list = self._natural_list(model_hits[:4])
                sentences_out.append(self._ensure_sentence(f"It integrates several components, including {model_list}"))
        else:
            missing_sentences.append(
                "The available text does not clearly name a more specific model family beyond the general methodology described in the paper"
            )

        if dataset_hits:
            if len(dataset_hits) == 1:
                sentences_out.append(self._ensure_sentence(f"The method is evaluated using the {dataset_hits[0]} dataset"))
            else:
                dataset_list = self._natural_list(dataset_hits[:4])
                sentences_out.append(self._ensure_sentence(f"The evaluation draws on several datasets, including {dataset_list}"))
        elif survey_like:
            sentences_out.append(
                self._ensure_sentence(
                    "The paper does not revolve around a named benchmark dataset and instead reads as a conceptual or review-oriented study"
                )
            )
        elif setup_sentence:
            normalized = self._clean_fragment(setup_sentence, title=title, max_words=26)
            sentences_out.append(self._ensure_sentence(f"Its evaluation setup is centered on {self._lower_first(normalized)}"))
        else:
            missing_sentences.append(
                "The dataset is not clearly named in the available text, so the safest summary is that the paper uses the study setting described in its abstract or extracted sections"
            )

        if eval_sentence and not survey_like:
            normalized = self._clean_fragment(eval_sentence, title=title, max_words=26)
            sentences_out.append(self._ensure_sentence(f"Analysis of the results suggests that {self._lower_first(normalized)}"))
        elif survey_like:
            sentences_out.append(
                self._ensure_sentence(
                    "Its contribution is framed more as synthesis and design guidance than as a benchmark-style comparison on one fixed dataset"
                )
            )
        else:
            missing_sentences.append(
                "The methodology is presented together with an evaluation process, but the extracted text does not expose all of the experimental details clearly"
            )

        if survey_like:
            sentences_out.append(
                self._ensure_sentence(
                    "Taken together, the methodology should be interpreted as a conceptual synthesis rather than a single deployable model"
                )
            )
        else:
            sentences_out.append(
                self._ensure_sentence(
                    "Taken together, the methodology connects the modeling choice, data, and evaluation into a coherent workflow"
                )
            )

        cleaned = [sentence for sentence in sentences_out if sentence]
        for sentence in missing_sentences:
            if len(cleaned) >= 3:
                break
            cleaned.append(self._ensure_sentence(sentence))
        final_approach = " ".join(cleaned[:5])
        final_approach = final_approach.replace(" .", ".").replace("  ", " ")
        return collapse_text(final_approach, 1400 if fulltext else 1100)

    def _gaps(self, rows: list[dict]) -> list[str]:
        gaps: list[str] = []
        for row in rows[:5]:
            title = clean_text(row.get("title", "")) or "Paper"
            if self._metadata_limited(row):
                gap = (
                    f"For {title}, the metadata is still too thin to state a paper-specific gap confidently, "
                    "so the safest next step is to inspect the full text before turning it into a concrete design recommendation."
                )
            else:
                gap = f"In {title}, {self._domain_gap(row)}."
            gaps.append(collapse_text(gap, 500))
        return gaps

    def _idea(self, gaps: list[str]) -> str:
        theme = self._topic_theme()
        if not gaps:
            return (
                f"A promising research direction would be to construct a more rigorous research pipeline for {theme} that combines focused retrieval, "
                "paper-level evidence tracking, and evaluation in real workflows, so the final recommendation rests on comparable studies rather than generic summaries."
            )
        return collapse_text(
            f"A strong follow-up project would be to design a {theme} pipeline that directly addresses the recurring weaknesses shared across these papers. "
            "This new study should compare closely related methods on the same task, keep the evaluation population and metrics explicit, "
            "and separate evidence-backed conclusions from metadata-only guesses.",
            900,
        )

    def _implementation_steps(self, gaps: list[str], rows: list[dict], idea: str) -> list[str]:
        titles = [self._paper_label(row.get("paper_name", "")) for row in rows if clean_text(row.get("paper_name", ""))]
        primary_title = titles[0] if titles else self._topic_theme().title()
        secondary_title = titles[1] if len(titles) > 1 else primary_title
        gap_focus = list(dict.fromkeys(self._gap_focus(g) for g in gaps[:3] if self._gap_focus(g)))

        steps: list[str] = []
        if len(titles) >= 2:
            steps.append(f"Use {primary_title} and {secondary_title} as the first comparison anchors, then add the remaining selected papers only if they still match the same task and population.")
        elif titles:
            steps.append(f"Use {primary_title} as the main comparison anchor and keep every later comparison tied to the same problem setting.")
        else:
            steps.append(f"Use the selected papers as the initial comparison set for {self._topic_theme()}, but remove any paper that is only loosely related before drawing conclusions.")

        if gap_focus:
            steps.append(f"Target the main weakness explicitly: {gap_focus[0]}.")
        else:
            steps.append("Target the main weakness shared across the selected papers and define it in measurable terms before implementation starts.")

        if len(gap_focus) > 1:
            steps.append(f"Design the first experiment to test whether the proposed method still holds when {gap_focus[1]}.")
        else:
            steps.append("Design the first experiment on a harder and more diverse evaluation set so the results are useful beyond one narrow setting.")

        steps.append("Add strong baselines from the same problem area and compare every system under one shared evaluation setup with the same metrics, population, and reporting rules.")
        steps.append("Run an ablation or sensitivity study so the contribution of each component is clear and the final paper can explain which part of the pipeline is actually responsible for any gain.")
        steps.append("Finish with an error analysis that explains where the approach fails, which users or cases are most affected, and what the next refinement should be.")
        return [collapse_text(step, 360) for step in steps[:6]]

    def build(
        self,
        rows: list[dict],
        fulltext_map: dict[tuple[str, str], str],
        fulltext_by_title: dict[str, str],
        excluded_titles: list[str] | None = None,
    ) -> dict[str, Any]:
        selected = self.select_rows(rows, limit=5, excluded_titles=excluded_titles)
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
                f"I found {len(table)} papers on {self.topic}. "
                f"The common thread across them is {self._topic_theme()}."
                + (
                    " The papers point to recurring weaknesses around evaluation depth, the quality of baseline comparisons, and the confidence with which results can be interpreted."
                    if gaps
                    else ""
                )
                + " A sensible next step is to focus only on the most relevant papers, compare them within a shared evaluation framework, and avoid making strong claims based on thin metadata."
            ),
            "generated_idea": idea,
            "generated_idea_steps": self._implementation_steps(gaps, table, idea),
            "generated_idea_citations": [row.get("paper_name", "") for row in table],
        }

    def build_insufficient(self) -> dict[str, Any]:
        return {
            "table": [],
            "research_gaps": [],
            "assistant_reply": (
                f"I couldn't find five closely relevant papers for '{self.topic}'. "
                "Try a narrower topic or add one concrete domain term so the retrieval can stay focused."
            ),
            "generated_idea": (
                "A better next move is to narrow the topic, retrieve a tighter paper set, and derive the research gaps from those papers before proposing a new idea."
            ),
            "generated_idea_steps": [
                f"Add one concrete keyword that narrows the topic to {self._topic_theme()}.",
                "Retrieve the top five close-match papers first.",
                "Filter out broad fallback results that do not match the topic.",
                "Generate the idea only after the paper set is clearly on-topic and comparable.",
            ],
            "generated_idea_citations": [],
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
