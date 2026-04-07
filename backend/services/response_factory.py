"""Response composition helpers for research and review workflows."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from backend.explorer_utils import fix_paper_url
from backend.helpers import authors_to_str, extract_models, extract_proposed_approach

from .text_utils import (
    _unique_sentences,
    clean_text,
    collapse_text,
    normalize_output_text,
    row_matches_topic,
    strip_front_matter,
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
        # Prefer the strict match set whenever it exists. Only fall back to the full
        # pool when no rows match at all, so follow-ups stay on-topic instead of
        # filling the table with unrelated papers.
        pool = matched if matched else rows
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
        fulltext = clean_text(fulltext_snippet)
        abstract_text = clean_text(abstract or row.get("abstract", ""))
        source_text = fulltext or abstract_text
        if fulltext and abstract_text:
            # Prefer the cleaner source when the extracted PDF text is too thin or noisy.
            if len(fulltext.split()) < 80 or len(abstract_text.split()) > len(fulltext.split()) * 1.2:
                source_text = abstract_text
        if not source_text:
            return collapse_text(
                (
                    f"This paper is about {title}. "
                    "It addresses a research problem that is not clearly specified in the available metadata. "
                    "The available metadata does not clearly describe the approach used to solve the problem. "
                    "The reported impact is not clearly stated in the available metadata. "
                    "Overall, the source suggests the work is relevant, but the evidence available here is limited."
                ),
                1200,
            )

        def _split_sentences(text: str) -> list[str]:
            cleaned = re.sub(r"(\w)-\s+(\w)", r"\1\2", text.strip())
            cleaned = re.sub(r"\s+", " ", cleaned).strip()
            pieces = re.split(r"(?<=[.!?])\s+", cleaned)
            out: list[str] = []
            for piece in pieces:
                piece = piece.strip()
                if len(piece.split()) < 5:
                    continue
                out.append(piece)
            return out

        def _normalize_sentence(sentence: str) -> str:
            sentence = re.sub(r"\s+", " ", sentence.strip())
            sentence = re.sub(r"(\w)-\s+(\w)", r"\1\2", sentence)
            sentence = collapse_text(sentence, 420)
            sentence = sentence.replace(title, "").strip(" :-â€“â€”")
            sentence = re.sub(
                r"(?:[\s,;:\-]+(?:an|a|and|or|of|to|the|for|with|in|on|at|by|from|as))+\.*$",
                "",
                sentence,
                flags=re.IGNORECASE,
            )
            sentence = re.sub(r"\s+", " ", sentence).strip(" ,;:-")
            return sentence

        def _shorten_clause(text: str, max_words: int = 24) -> str:
            words = [w for w in re.sub(r"\s+", " ", text).strip().split() if w]
            if len(words) <= max_words:
                clause = " ".join(words)
            else:
                clause = " ".join(words[:max_words]).rstrip(" ,;:")
            clause = re.sub(
                r"(?:[\s,;:\-]+(?:an|a|and|or|of|to|the|for|with|in|on|at|by|from|as))+\.*$",
                "",
                clause,
                flags=re.IGNORECASE,
            )
            clause = re.sub(r"\s+", " ", clause).strip(" ,;:-")
            return clause

        def _has_dangling_tail(text: str) -> bool:
            return bool(
                re.search(
                    r"(?:^|[\s,;:\-])(?:an|a|and|or|of|to|the|for|with|in|on|at|by|from|as)\.?$",
                    text.strip(),
                    flags=re.IGNORECASE,
                )
            )

        def _is_meaningful(text: str, min_words: int = 6) -> bool:
            words = [w for w in re.sub(r"\s+", " ", text).split() if w]
            alpha_chars = len(re.findall(r"[A-Za-z]", text))
            return len(words) >= min_words and alpha_chars >= 24 and not _has_dangling_tail(text)

        def _find_sentence(sentences: list[str], keywords: list[str]) -> str:
            for sentence in sentences:
                low = sentence.lower()
                if any(keyword in low for keyword in keywords):
                    normalized = _normalize_sentence(sentence)
                    if _is_meaningful(normalized):
                        return normalized
                    clauses = re.split(r",\s+|;\s+|:\s+", normalized)
                    for clause in clauses:
                        clause = _normalize_sentence(clause)
                        if _is_meaningful(clause, min_words=5):
                            return clause
                    if normalized:
                        return normalized
            return ""

        def _drop_front_matter(text: str) -> str:
            cleaned = re.sub(r"\s+", " ", text).strip()
            if title and cleaned.lower().startswith(title.lower()):
                cleaned = cleaned[len(title):].lstrip(" :-—–")
            for marker in [r"\babstract\b[:\-\s]*", r"\bintroduction\b[:\-\s]*", r"\bmethodology\b[:\-\s]*", r"\bmethods?\b[:\-\s]*"]:
                match = re.search(marker, cleaned, flags=re.IGNORECASE)
                if match:
                    cleaned = cleaned[match.end():].strip()
                    break
            return cleaned

        method_source = _drop_front_matter(strip_front_matter(source_text, title) or source_text)
        sentences = _unique_sentences(_split_sentences(method_source) or _split_sentences(source_text))
        first = _normalize_sentence(sentences[0]) if sentences else ""
        if not _is_meaningful(first):
            first = ""
        if not first:
            first = _normalize_sentence(method_source[:360])
        if not _is_meaningful(first):
            first = ""
        if not first and abstract_text:
            abstract_first = _normalize_sentence(abstract_text.split(".")[0] if "." in abstract_text else abstract_text[:280])
            if _is_meaningful(abstract_first, min_words=7):
                first = abstract_first

        problem = _find_sentence(
            sentences,
            ["problem", "challenge", "gap", "need", "difficult", "demand", "issue", "limitation"],
        )
        if not _is_meaningful(problem):
            problem = ""
        if not problem and abstract_text:
            problem = _normalize_sentence(abstract_text.split(".")[0] if "." in abstract_text else abstract_text[:280])

        approach = ""
        if method_source:
            approach = _normalize_sentence(extract_proposed_approach(method_source))
        if not approach:
            approach = _find_sentence(
                sentences,
                ["propose", "present", "introduce", "method", "approach", "framework", "model", "algorithm", "system"],
            )
        if not approach and abstract_text:
            approach = _normalize_sentence(extract_proposed_approach(abstract_text))
        if not _is_meaningful(approach):
            approach = ""

        impact = _find_sentence(
            sentences,
            [
                "improve",
                "improves",
                "improved",
                "increase",
                "reduces",
                "reduce",
                "outperform",
                "outperforms",
                "better",
                "effect",
                "impact",
                "result",
                "performance",
                "accuracy",
                "feasible",
                "effective",
            ],
        )
        if not _is_meaningful(impact):
            impact = ""

        topic_hint = first or f"This paper is about {title}."
        topic_hint = _shorten_clause(topic_hint, max_words=28)
        if not topic_hint.endswith((".", "!", "?")):
            topic_hint += "."

        problem_clause = problem or (
            "It addresses the central research challenge described in the paper."
            if "paper" not in title.lower()
            else "It addresses the main problem described in the paper."
        )
        problem_clause = _shorten_clause(problem_clause, max_words=28)
        if not problem_clause.endswith((".", "!", "?")):
            problem_clause += "."

        method_clause = approach or "It solves the problem by applying the method described in the paper."
        method_clause = _shorten_clause(method_clause, max_words=30)
        if not method_clause.endswith((".", "!", "?")):
            method_clause += "."

        impact_clause = impact or "The reported impact is that the solution makes the target task more effective or practical."
        impact_clause = _shorten_clause(impact_clause, max_words=28)
        if not impact_clause.endswith((".", "!", "?")):
            impact_clause += "."

        about_sentence = topic_hint
        if not about_sentence.lower().startswith("this paper"):
            about_sentence = f"This paper is about {about_sentence[0].lower() + about_sentence[1:]}" if about_sentence else f"This paper is about {title}."
        if not about_sentence.endswith((".", "!", "?")):
            about_sentence += "."

        problem_sentence = problem_clause
        if not problem_sentence.lower().startswith("it "):
            problem_sentence = f"It solves the problem of {problem_sentence[0].lower() + problem_sentence[1:]}" if problem_sentence else "It solves the main problem described in the paper."
        if not problem_sentence.endswith((".", "!", "?")):
            problem_sentence += "."

        method_sentence = method_clause
        if not method_sentence.lower().startswith("it "):
            method_sentence = f"It takes the approach of {method_sentence[0].lower() + method_sentence[1:]}" if method_sentence else "It takes the approach described in the paper."
        if not method_sentence.endswith((".", "!", "?")):
            method_sentence += "."

        impact_sentence = impact_clause
        if not impact_sentence.lower().startswith("the "):
            impact_sentence = f"The impact of the approach is that {impact_sentence[0].lower() + impact_sentence[1:]}" if impact_sentence else "The impact of the approach is not clearly quantified in the available text."
        if not impact_sentence.endswith((".", "!", "?")):
            impact_sentence += "."

        closing_sentence = "Overall, the approach appears meaningful for the target task based on the evidence reported in the paper."

        structured_summary = " ".join(
            [
                about_sentence,
                problem_sentence,
                method_sentence,
                impact_sentence,
                closing_sentence,
            ]
        )
        return collapse_text(structured_summary, 1600 if fulltext else 900)

    def _problem(self, row: dict, fulltext_snippet: str = "", abstract: str = "") -> str:
        title = clean_text(row.get("title", "")) or "Untitled paper"
        text = clean_text(fulltext_snippet or abstract or row.get("abstract", ""))
        if text:
            first = re.split(r"(?<=[.!?])\s+", text)[0].strip()
            return collapse_text(first, 360)
        return collapse_text(f"The paper addresses the topic suggested by its title: {title}.", 360)

    def _approach(self, row: dict, fulltext_snippet: str = "", abstract: str = "") -> str:
        title = clean_text(row.get("title", "")) or "Untitled paper"
        fulltext = clean_text(fulltext_snippet)
        abstract_text = clean_text(abstract or row.get("abstract", ""))
        source_text = fulltext or abstract_text
        if fulltext and abstract_text:
            if len(abstract_text.split()) > len(fulltext.split()) * 1.2:
                source_text = abstract_text
        if not source_text:
            return collapse_text(
                f"The source metadata does not expose a clear methodology for {title}.",
                420,
            )

        def _normalize(text: str) -> str:
            text = re.sub(r"(?is)^.*?\babstract\b\s*[:-]?\s*", "", text)
            text = re.sub(r"\s+", " ", text).strip()
            text = text.replace(title, "")
            return text.strip(" ,;:-")

        def _sentences(text: str) -> list[str]:
            cleaned = re.sub(r"(\w)-\s+(\w)", r"\1\2", text.strip())
            cleaned = re.sub(r"\s+", " ", cleaned).strip()
            raw = [piece.strip() for piece in re.split(r"(?<=[.!?])\s+", cleaned) if len(piece.split()) >= 5]
            return _unique_sentences(raw)

        def _compact(text: str, max_words: int = 26) -> str:
            words = [w for w in re.sub(r"\s+", " ", text).split() if w]
            if len(words) > max_words:
                text = " ".join(words[:max_words]).rstrip(" ,;:")
            return _normalize(text)

        def _pick(sentence_list: list[str], keywords: list[str], min_words: int = 5) -> str:
            for sentence in sentence_list:
                low = sentence.lower()
                if any(keyword in low for keyword in keywords):
                    normalized = _compact(sentence)
                    if len(normalized.split()) >= min_words:
                        return normalized
            return ""

        def _sanitize_candidate(text: str) -> str:
            cleaned = _compact(text, max_words=40)
            cleaned = re.sub(r"^(?:this paper|the paper|this study|the study)\s+(?:is|was)\s+about\s+", "", cleaned, flags=re.IGNORECASE)
            cleaned = re.sub(
                r"^(?:the source metadata does not expose|the paper describes|the paper studies|it proposes or applies)\s+",
                "",
                cleaned,
                flags=re.IGNORECASE,
            )
            cleaned = cleaned.strip(" ,;:-")
            return cleaned

        def _is_generic(text: str) -> bool:
            lowered = text.lower()
            generic_phrases = [
                "source metadata",
                "paper describes its empirical setting",
                "methodology should be read cautiously",
                "the methodology described in the paper text",
                "this paper is about",
                "the paper studies",
            ]
            return any(phrase in lowered for phrase in generic_phrases)

        def _methods_block(text: str) -> str:
            cleaned = re.sub(r"\s+", " ", text).strip()
            patterns = [
                r"(?is)\b(?:methodology|materials and methods|methods?|approach|proposed method|proposed approach|system overview)\b\s*[:\-]?\s*(.*?)(?=\b(?:experiments?|results?|discussion|evaluation|conclusion|conclusions|references)\b|$)",
                r"(?is)\b(?:we propose|we present|we introduce|in this paper, we propose|in this work, we propose)\b\s*(.*?)(?=\b(?:experiments?|results?|discussion|evaluation|conclusion|conclusions|references)\b|$)",
            ]
            for pattern in patterns:
                match = re.search(pattern, cleaned)
                if match and match.group(1).strip():
                    return match.group(1).strip()
            return cleaned

        method_source = strip_front_matter(source_text, title) or source_text
        method_focus = _methods_block(method_source)
        sentence_list = _sentences(method_focus) or _sentences(method_source) or _sentences(source_text)
        abstract_sentences = _sentences(abstract_text)

        survey_like = any(
            term in f"{title} {method_source} {abstract_text}".lower()
            for term in ["systematic review", "survey", "literature review", "scoping review", "meta-analysis", "overview"]
        )

        candidates: list[str] = []
        for candidate in [method_focus, method_source, abstract_text, source_text]:
            if not candidate:
                continue
            extracted = _sanitize_candidate(extract_proposed_approach(candidate))
            extracted_low = extracted.lower()
            if extracted and len(extracted.split()) >= 6 and any(
                cue in extracted_low for cue in ["propose", "present", "introduce", "develop", "design", "framework", "method", "approach", "algorithm", "system", "review"]
            ):
                candidates.append(extracted)

        for keyword_group in [
            ["we propose", "we present", "we introduce", "we develop", "we design"],
            ["framework", "method", "approach", "algorithm", "system", "pipeline"],
            ["review", "survey", "systematic review", "literature review"],
        ]:
            picked = _pick(sentence_list or abstract_sentences, keyword_group, min_words=6)
            if picked:
                candidates.append(_sanitize_candidate(picked))

        named_model_sentence = ""
        model_hits = extract_models(f"{title} {method_focus} {method_source} {abstract_text} {fulltext}")
        if model_hits:
            model_sentence = _pick(
                sentence_list or abstract_sentences,
                [hit.lower() for hit in model_hits],
                min_words=5,
            )
            if model_sentence:
                named_model_sentence = _sanitize_candidate(model_sentence)

        if named_model_sentence:
            candidates.append(named_model_sentence)

        approach_text = ""
        seen_candidates: set[str] = set()
        for candidate in candidates:
            key = re.sub(r"\s+", " ", candidate).strip().lower()
            if not key or key in seen_candidates or _is_generic(candidate):
                continue
            seen_candidates.add(key)
            approach_text = candidate
            break

        if survey_like and approach_text:
            return collapse_text(
                f"The paper takes a survey-based approach: {approach_text}.",
                1200 if fulltext else 850,
            )
        if survey_like:
            return collapse_text(
                "The paper is a survey or systematic review that synthesizes prior work from the paper text rather than introducing a single new executable method.",
                1200 if fulltext else 850,
            )
        if approach_text:
            return collapse_text(approach_text, 1200 if fulltext else 850)
        return collapse_text(
            "The paper text does not clearly expose one distinct proposed method, but the approach should be read from the methods and experiment sections of the source PDF.",
            1200 if fulltext else 850,
        )

        def _normalize(text: str) -> str:
            text = re.sub(r"(?is)^.*?\babstract\b\s*[–-]\s*", "", text)
            text = re.sub(r"\s+", " ", text).strip()
            text = text.replace(title, "")
            return text.strip(" :-ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â€šÂ¬Ã…â€œÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ,;")

        def _sentences(text: str) -> list[str]:
            cleaned = re.sub(r"(\w)-\s+(\w)", r"\1\2", text.strip())
            cleaned = re.sub(r"\s+", " ", cleaned).strip()
            return [piece.strip() for piece in re.split(r"(?<=[.!?])\s+", cleaned) if len(piece.split()) >= 5]

        def _compact(text: str, max_words: int = 20) -> str:
            words = [w for w in re.sub(r"\s+", " ", text).split() if w]
            if len(words) > max_words:
                text = " ".join(words[:max_words]).rstrip(" ,;:")
            return _normalize(text)

        def _methods_block(text: str) -> str:
            cleaned = re.sub(r"\s+", " ", text).strip()
            for pattern in [
                r"(?is)(?:methodology|methods?|experimental setup|approach)\s*[:\-]?\s*(.*?)(?=\b(?:results?|discussion|conclusion|conclusions|references)\b|$)",
                r"(?is)(?:materials and methods|methodology and data|data and methods)\s*[:\-]?\s*(.*?)(?=\b(?:results?|discussion|conclusion|references)\b|$)",
            ]:
                match = re.search(pattern, cleaned)
                if match and match.group(1).strip():
                    return match.group(1).strip()
            return cleaned

        def _drop_front_matter(text: str) -> str:
            cleaned = re.sub(r"\s+", " ", text).strip()
            title_norm = re.sub(r"\s+", " ", title).strip().lower()
            lower = cleaned.lower()
            if title_norm and lower.startswith(title_norm):
                cleaned = cleaned[len(title):].lstrip(" :-—–")
                lower = cleaned.lower()
            for marker in [
                r"\babstract\b[:\-\s]*",
                r"\bintroduction\b[:\-\s]*",
                r"\b1\.\s*introduction\b[:\-\s]*",
                r"\bmethodology\b[:\-\s]*",
                r"\bmethods?\b[:\-\s]*",
            ]:
                match = re.search(marker, cleaned, flags=re.IGNORECASE)
                if match:
                    cleaned = cleaned[match.end():].strip()
                    break
            return cleaned

        def _pick(sentence_list: list[str], keywords: list[str], min_words: int = 5) -> str:
            for sentence in sentence_list:
                low = sentence.lower()
                if any(keyword in low for keyword in keywords):
                    norm = _normalize(sentence)
                    if len(norm.split()) >= min_words:
                        return norm
            return ""

        def _article(word: str) -> str:
            return "an" if word[:1].lower() in {"a", "e", "i", "o", "u"} else "a"

        def _clause_after(text: str, cue: str) -> str:
            match = re.search(rf"(?is)\b{re.escape(cue)}\b\s+(.*?)(?:,|;|:|\.|$)", text)
            if match and match.group(1).strip():
                return _compact(match.group(1).strip(), max_words=18)
            return ""

        method_source = strip_front_matter(source_text, title) or source_text
        method_source = _drop_front_matter(method_source)
        method_source = _methods_block(method_source)
        method_source = _drop_front_matter(method_source)
        sentence_list = _sentences(method_source) or _sentences(source_text) or [source_text]

        dataset_sentence = _pick(
            sentence_list,
            ["dataset", "data set", "corpus", "benchmark", "samples", "sample", "collection", "data", "experiment"],
            min_words=5,
        ) or _pick(
            _sentences(abstract_text),
            ["dataset", "data set", "corpus", "benchmark", "samples", "sample", "collection", "data"],
            min_words=5,
        )
        if dataset_sentence:
            for cue in ["requires", "using", "based on", "built on", "collected from", "produced by"]:
                clause = _clause_after(dataset_sentence, cue)
                if clause:
                    dataset_sentence = clause
                    break

        model_hits = extract_models(f"{title} {method_source} {abstract_text} {fulltext}")
        if model_hits:
            model_phrase = f"{_article(model_hits[0])} {model_hits[0]}"
        else:
            model_phrase = _pick(
                sentence_list,
                ["model", "method", "approach", "framework", "algorithm", "network", "classifier", "transformer"],
                min_words=4,
            ) or "the main model described in the paper"

        approach_sentence = _pick(
            sentence_list,
            ["propose", "present", "introduce", "use", "train", "fine-tune", "optimize", "evaluate", "compare", "pipeline", "workflow", "process", "combine"],
            min_words=5,
        ) or "the methodology described in the methods section"
        if approach_sentence:
            for cue in ["using", "through", "by", "with", "apply", "applies", "carried out through", "based on"]:
                clause = _clause_after(approach_sentence, cue)
                if clause:
                    approach_sentence = clause
                    break
        if len(approach_sentence.split()) < 5:
            approach_sentence = ""
        if not approach_sentence:
            approach_sentence = "the methodology described in the paper text"

        parts = [
            f"The paper uses {dataset_sentence}." if dataset_sentence else "The paper describes its dataset or experimental setup in the methods section.",
            f"It uses {model_phrase} as the main model.",
            f"The approach is carried out through {approach_sentence}.",
        ]
        return collapse_text(" ".join(parts), 1200 if fulltext else 850)

        def _methodology_sentences(text: str) -> list[str]:
            cleaned = re.sub(r"(\w)-\s+(\w)", r"\1\2", text.strip())
            cleaned = re.sub(r"\s+", " ", cleaned).strip()
            pieces = re.split(r"(?<=[.!?])\s+", cleaned)
            return [piece.strip() for piece in pieces if len(piece.split()) >= 5]

        def _compact(text: str, max_words: int = 20) -> str:
            words = [w for w in re.sub(r"\s+", " ", text).split() if w]
            if len(words) > max_words:
                text = " ".join(words[:max_words]).rstrip(" ,;:")
            text = re.sub(r"\s+", " ", text).strip()
            text = text.replace(title, "").strip(" :-ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Å“ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â")
            return text.strip(" ,;:-")

        def _find_phrase(text: str, patterns: list[str], default: str = "") -> str:
            lowered = text.lower()
            for pattern in patterns:
                match = re.search(pattern, lowered, flags=re.IGNORECASE)
                if match:
                    return _compact(text[match.start() : match.end()], max_words=14)
            return default

        method_source = strip_front_matter(source_text, title) or source_text
        sentences = _methodology_sentences(method_source)
        if not sentences:
            sentences = _methodology_sentences(source_text) or [source_text]

        dataset_sentence = ""
        for sentence in sentences:
            low = sentence.lower()
            if any(term in low for term in ["dataset", "data set", "corpus", "benchmark", "samples", "sample", "collection"]):
                dataset_sentence = _compact(sentence, 22)
                break

        detected_models = extract_models(f"{title} {method_source or abstract_text or fulltext}")
        model_phrase = f"a {detected_models[0]} model" if detected_models else ""
        if not model_phrase:
            model_phrase = _find_phrase(
                method_source,
                [
                    r"one-class support vector machine(?: \(oc-svm\))?",
                    r"\bsvm\b",
                    r"convolutional neural network(?:s)?(?: \(cnn\))?",
                    r"recurrent neural network(?:s)?(?: \(rnn\))?",
                    r"long short-term memory(?: \(lstm\))?",
                    r"transformer(?:-based)?",
                    r"graph neural network(?:s)?(?: \(gnn\))?",
                    r"random forest",
                    r"xgboost",
                    r"lightgbm",
                    r"catboost",
                    r"rule-based system",
                    r"fuzzy logic",
                ],
            )
        if not model_phrase:
            for sentence in sentences:
                low = sentence.lower()
                if any(term in low for term in ["model", "method", "approach", "framework", "algorithm", "network", "classifier"]):
                    model_phrase = ""
                    break

        parts: list[str] = []
        if dataset_sentence:
            parts.append(f"The methodology uses the dataset or experimental setup described as {dataset_sentence}.")
        else:
            parts.append("The methodology is described in the paper's methods section.")

        if model_phrase:
            parts.append(f"It uses {model_phrase} as the main model.")
        else:
            parts.append("It states the main model used in the workflow.")

        parts.append("It applies the methodology described in the methods section, including the preprocessing, training, and evaluation steps.")

        return collapse_text(" ".join(parts[:3]), 1200 if fulltext else 850)

        def _split_sentences(text: str) -> list[str]:
            cleaned = re.sub(r"(\w)-\s+(\w)", r"\1\2", text.strip())
            cleaned = re.sub(r"\s+", " ", cleaned).strip()
            pieces = re.split(r"(?<=[.!?])\s+", cleaned)
            return [piece.strip() for piece in pieces if len(piece.split()) >= 5]

        def _normalize(sentence: str) -> str:
            sentence = re.sub(r"\s+", " ", sentence.strip())
            if title:
                sentence = sentence.replace(title, "")
            sentence = sentence.strip(" :-Ã¢â‚¬â€œÃ¢â‚¬â€")
            sentence = re.sub(r"\s+(?:an|a|and|or|of|to|the|for|with|in|on|at|by|from|as)\.?$", "", sentence, flags=re.IGNORECASE)
            return re.sub(r"\s+", " ", sentence).strip(" ,;:-")

        def _pick_sentence(sentences: list[str], keywords: list[str], min_words: int = 5) -> str:
            for sentence in sentences:
                low = sentence.lower()
                if any(keyword in low for keyword in keywords):
                    normalized = _normalize(sentence)
                    if len(normalized.split()) >= min_words:
                        return normalized
            return ""

        def _clamp_sentence(text: str, max_words: int = 30) -> str:
            words = [w for w in re.sub(r"\s+", " ", text).split() if w]
            if len(words) > max_words:
                text = " ".join(words[:max_words]).rstrip(" ,;:")
            return _normalize(text)

        def _model_phrase(text: str) -> str:
            lowered = text.lower()
            if "one-class support vector machine" in lowered or "oc-svm" in lowered:
                return "a one-class support vector machine (OC-SVM) model"
            if re.search(r"\bsvm\b|\bsvm-based\b", lowered, flags=re.IGNORECASE):
                return "a support vector machine (SVM) model"
            if "convolutional neural network" in lowered or "cnn" in lowered:
                return "a convolutional neural network (CNN)"
            if "recurrent neural network" in lowered or "rnn" in lowered:
                return "a recurrent neural network (RNN)"
            if "long short-term memory" in lowered or "lstm" in lowered:
                return "a long short-term memory (LSTM) model"
            if "transformer" in lowered:
                return "a transformer-based model"
            if "graph neural network" in lowered or "gnn" in lowered:
                return "a graph neural network (GNN)"
            if "random forest" in lowered:
                return "a random forest model"
            if "xgboost" in lowered:
                return "XGBoost"
            if "lightgbm" in lowered:
                return "LightGBM"
            if "catboost" in lowered:
                return "CatBoost"
            if "rule-based system" in lowered:
                return "a rule-based system"
            if "fuzzy logic" in lowered:
                return "fuzzy logic"
            return ""

        sentences = _split_sentences(source_text)
        if not sentences:
            sentences = [source_text]

        dataset_info = _pick_sentence(
            sentences,
            ["dataset", "data set", "corpus", "benchmark", "sample", "samples", "collection", "study", "experiments"],
            min_words=6,
        )
        model_info = _pick_sentence(
            sentences,
            ["model", "models", "method", "methods", "approach", "framework", "algorithm", "network", "classifier", "transformer", "svm", "llm"],
            min_words=6,
        )
        method_info = _pick_sentence(
            sentences,
            ["propose", "present", "introduce", "train", "fine-tune", "optimize", "evaluate", "compare", "combine", "use", "describe", "process"],
            min_words=6,
        )
        if method_info and ("abstract" in method_info.lower() or len(method_info.split()) > 18):
            method_info = ""
        impact_info = _pick_sentence(
            sentences,
            ["improve", "improves", "improved", "reduce", "reduces", "outperform", "outperforms", "accuracy", "performance", "effective", "practical", "result"],
            min_words=6,
        )
        model_phrase = _model_phrase(source_text) or _model_phrase(abstract_text) or _model_phrase(fulltext)
        if model_info and ("abstract" in model_info.lower() or len(model_info.split()) > 14):
            model_info = ""
        if model_phrase:
            model_info = model_phrase
        if method_info and method_info == model_info:
            method_info = ""

        parts: list[str] = []
        if dataset_info:
            parts.append(_clamp_sentence(f"The methodology is built around the dataset or experimental setting described as {dataset_info}.", 34))
        else:
            parts.append(_clamp_sentence("The methodology is centered on the dataset and experimental setup described in the paper.", 34))

        if model_info:
            parts.append(_clamp_sentence(f"It uses {model_info} as the main model in the workflow.", 34))
        elif method_info:
            parts.append(_clamp_sentence(f"It uses {method_info} as the main approach in the workflow.", 34))
        else:
            parts.append(_clamp_sentence("It uses the paper's described model or algorithm as the main approach.", 34))

        if method_info and method_info not in parts[-1]:
            parts.append(_clamp_sentence(f"The approach is implemented through {method_info}.", 34))
        if impact_info:
            parts.append(_clamp_sentence(f"The paper evaluates the methodology through experiments and reports {impact_info}.", 34))
        else:
            parts.append(_clamp_sentence("The paper evaluates the approach through the experiments and comparisons reported in the source text.", 34))

        if len(parts) < 3:
            parts.append(_clamp_sentence("The paper combines preprocessing, training, and evaluation into one workflow.", 34))

        summary = " ".join(parts[:4])
        return collapse_text(summary, 1600 if fulltext else 900)

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
                gap = f"{title}: move beyond description and add stronger empirical comparisons or benchmark-style evaluation."
            elif "multilingual" in blob or "low-resource" in blob:
                gap = f"{title}: expand evaluation to more languages and report transfer clearly."
            elif "space" in blob or "planet" in blob:
                gap = f"{title}: connect the method to a concrete scientific or operational use case with clearer validation."
            elif "generative ai" in blob or "genai" in blob:
                if idx == 0:
                    gap = f"{title}: benchmark the approach against stronger baseline systems and report task-level error analysis."
                elif idx == 1:
                    gap = f"{title}: test whether the claims hold across different user groups, prompts, or deployment settings."
                elif idx == 2:
                    gap = f"{title}: add robustness checks for hallucination, safety, and prompt sensitivity."
                else:
                    gap = f"{title}: clarify which part of the pipeline contributes most to the reported gains."
            else:
                gap = f"{title}: strengthen the evaluation with stronger baselines and clearer error analysis."
            gaps.append(collapse_text(gap, 360))
        return gaps

    def _idea(self, gaps: list[str]) -> str:
        theme = self._topic_theme()
        if not gaps:
            return f"Build a stronger benchmark for {theme} that tests robustness, transfer, and interpretability."
        gap_summary = " ".join(dict.fromkeys(self._gap_focus(g) for g in gaps[:3]))
        return collapse_text(
            f"Build a stronger {theme} pipeline that addresses the recurring weaknesses across these papers, especially {gap_summary}.",
            700,
        )

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
                f"I found {len(table)} papers related to {self.topic}. "
                f"The common thread is {self._topic_theme()}, and the main open problems are "
                f"{' '.join(dict.fromkeys(self._gap_focus(g) for g in gaps[:2]))}. "
                "A good next step is to test whether these methods still hold up on harder, more diverse data."
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
