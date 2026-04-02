
"""LangChain prompt templates and chains."""

from __future__ import annotations

from langchain_classic.prompts import PromptTemplate
from langchain_core.language_models import BaseLLM
from langchain_core.runnables import Runnable


RESEARCH_PROMPT = PromptTemplate(
    input_variables=["context", "question", "papers_json"],
    template=(
        "You are a conversational research assistant. Use the provided context, chat history, and papers list. "
        "If the user asks for 'more' or similar, acknowledge it and provide additional papers from the list "
        "that were not emphasized in previous responses. "
        "Do NOT invent papers, titles, authors, venues, metrics, datasets, or claims. "
        "Every row MUST correspond to a paper from the papers list. "
        "Do NOT mention missing full text in any field. "
        "Return ONLY valid JSON (no markdown, no code fences, no extra text). "
        "Use double quotes for all strings and no trailing commas. "
        "Use the following schema:\n"
        "{{\n"
        '  "table": [\n'
        "    {{\n"
        '      "paper_name": "...",\n'
        '      "paper_url": "...",\n'
        '      "authors_name": "...",\n'
        '      "summary_full_paper": "...",\n'
        '      "problem_solved": "...",\n'
        '      "proposed_model_or_approach": "...",\n'
        '      "source": "arxiv | semantic_scholar | google_scholar | researchgate | web | sciencedirect | openalex | core | doaj | europe_pmc",\n'
        '      "score_relevance": 0,\n'
        '      "score_quality": 0\n'
        "    }}\n"
        "  ],\n"
        '  "research_gaps": ["Paper name: gap text"],\n'
        '  "assistant_reply": "...",\n'
        '  "generated_idea": "...",\n'
        '  "generated_idea_steps": ["Step 1 ...", "Step 2 ..."],\n'
        '  "generated_idea_citations": ["paper_name"]\n'
        "}}\n\n"
        "Rules:\n"
        "- Choose only relevant papers to the user's question; if none are relevant, return an empty table and explain in assistant_reply.\n"
        "- From the papers list, select exactly 5 of the most relevant papers for the user's question.\n"
        "- The output table MUST have exactly 5 rows, each corresponding to one of the selected papers.\n"
        "- Present the output in this order: table, research_gaps, generated_idea, generated_idea_steps.\n"
        "- For each row, copy paper_name, paper_url, authors_name, and source from the papers list.\n"
        "- If a DOI is present in the papers list, prefer https://doi.org/DOI as paper_url.\n"
        "- problem_solved MUST describe the main problem addressed by the paper.\n"
        "- score_relevance and score_quality MUST be integers from 0 to 10.\n"
        "- If fulltext_available is true, summary_full_paper MUST reflect the full paper content.\n"
        "- If fulltext_available is false, summary_full_paper MUST be an abstract-based summary without stating that full text was missing.\n"
        "- proposed_model_or_approach MUST describe what the paper explicitly proposes (method/model/approach/algorithm). If the paper does not propose a new method, say so clearly using cautious language such as 'The source metadata does not expose a distinct new method'.\n"
        "- research_gaps MUST contain one specific, actionable gap per selected paper (list format).\n"
        "- assistant_reply MUST be a concise researcher-style response (neutral, evidence-based, no fluff) that introduces the findings.\n"
        "- generated_idea MUST synthesize the identified research_gaps into one concrete and novel research direction.\n"
        "- generated_idea_steps MUST be 6-8 detailed, actionable steps to implement the generated_idea, referencing specific methods or datasets from the context where appropriate.\n"
        "- NEVER leave any field blank. If information is missing, summarize cautiously from the title/metadata instead of repeating a placeholder.\n"
        '- generated_idea_citations MUST list paper_name values used in the idea.\n\n'
        "Example (format only, use real content):\n"
        '{{"table":[{{"paper_name":"...","paper_url":"...","authors_name":"...",'
        '"summary_full_paper":"...","problem_solved":"...",'
        '"proposed_model_or_approach":"...","source":"arxiv","score_relevance":8,"score_quality":7}}],'
        '"research_gaps":["Paper A: ...","Paper B: ..."],"generated_idea":"...",'
        '"generated_idea_steps":["Step 1 ...","Step 2 ..."],'
        '"generated_idea_citations":["Paper A","Paper B"]}}\n\n'
        "Papers list (JSON):\n{papers_json}\n\n"
        "Context:\n{context}\n\nQuestion: {question}"
    ),
)

REVIEW_PROMPT = PromptTemplate(
    input_variables=["paper"],
    template=(
        "You are a peer reviewer. Use ONLY the provided paper text. "
        "Ignore the title page, author list, affiliation block, conference header, page numbers, and reference list. "
        "Do NOT copy the paper title or venue into the review fields. "
        "Do NOT invent details not stated in the text. "
        "Write concise, evidence-based critique grounded in the paper's abstract, introduction, method, experiments, and conclusion. "
        "Return ONLY valid JSON with fields:\n"
        "{{\n"
        '  "strengths": "...",\n'
        '  "weaknesses": "...",\n'
        '  "novelty": "...",\n'
        '  "technical_correctness": "...",\n'
        '  "reproducibility": "...",\n'
        '  "recommendation": "Accept | Minor Revision | Major Revision | Reject",\n'
        '  "suggested_venue": "Conference | Journal"\n'
        "}}\n\n"
        "Guidelines:\n"
        "- Strengths should cite concrete contributions, not the paper title or author names.\n"
        "- Weaknesses should mention missing experiments, unclear baselines, limited evaluation, or scope issues when supported by the text.\n"
        "- Novelty should explain what is actually new relative to the described method.\n"
        "- Technical correctness should focus on method clarity, assumptions, and evaluation soundness.\n"
        "- Reproducibility should assess whether datasets, hyperparameters, code, or training details are sufficient.\n"
        "- Suggested venue must be exactly one label: Conference or Journal. Choose Conference for shorter, incremental, or application-focused papers; choose Journal for more mature, comprehensive, and deeply validated work.\n"
        "- Keep each field to 1-3 sentences and avoid generic filler.\n\n"
        "Paper text:\n{paper}"
    ),
)

PAPER_QA_PROMPT = PromptTemplate(
    input_variables=["paper_text", "question"],
    template=(
        "You are a research assistant. Answer the user's question based ONLY on the provided paper text. "
        "If the answer is not in the text, say \"Not specified in the paper.\" Do not guess.\n\n"
        "Paper text:\n{paper_text}\n\n"
        "Question: {question}"
    ),
)

REVIEW_FOLLOWUP_PROMPT = PromptTemplate(
    input_variables=["paper_text", "question"],
    template=(
        "You are continuing a peer review discussion. "
        "Answer the user's question as a reviewer, not as a generic QA assistant. "
        "Keep the response critique-oriented and grounded in the paper text. "
        "Focus on strengths, weaknesses, novelty, technical correctness, reproducibility, or recommendation when relevant. "
        "Do not repeat the paper title, authors, or venue unless directly relevant. "
        "If the paper text does not support a confident answer, say what evidence is missing.\n\n"
        "Paper text:\n{paper_text}\n\n"
        "Question: {question}"
    ),
)

PAPER_CHUNK_SUMMARIZER_PROMPT = PromptTemplate(
    input_variables=["chunk"],
    template=(
        "You are a research assistant. Summarize the following chunk of a research paper. "
        "Be factual and concise. Do not add external information.\n\n"
        "Chunk:\n{chunk}"
    ),
)

REFERENCE_PROMPT = PromptTemplate(
    input_variables=["topic", "seed_references"],
    template=(
        "You are a scholarly assistant. Using the provided seed references, return ONLY a JSON array of 10 APA references. "
        "Only real references that match the topic. Use the seed references as a base and do not invent titles.\n\n"
        "Topic: {topic}\n"
        "Seed references:\n{seed_references}\n"
    ),
)

JSON_REPAIR_PROMPT = PromptTemplate(
    input_variables=["bad_json", "schema_hint"],
    template=(
        "You are a strict JSON repair tool. Convert the input into valid JSON ONLY. "
        "Do not add commentary or code fences. Use the schema hint as guidance.\n\n"
        "Schema hint:\n{schema_hint}\n\n"
        "Input:\n{bad_json}"
    ),
)

GAP_IDEA_PROMPT = PromptTemplate(
    input_variables=["context", "papers_json", "question"],
    template=(
        "You are a research assistant. Use ONLY the provided context and papers list. "
        "Do NOT invent papers or claims.\n"
        "Return exactly two lines (no JSON, no bullets):\n"
        "Research Gap: <one concise paragraph>\n"
        "Generated Idea: <concrete solution + brief procedure>\n\n"
        "Papers list (JSON):\n{papers_json}\n\n"
        "Context:\n{context}\n\nQuestion: {question}"
    ),
)

GAP_LIST_PROMPT = PromptTemplate(
    input_variables=["context", "papers_json", "question"],
    template=(
        "You are a research assistant. Use ONLY the provided context and papers list. "
        "Do NOT invent papers or claims.\n"
        "Return ONLY valid JSON with this schema:\n"
        "{{\n"
        '  "research_gaps": ["Paper name: gap text"],\n'
        '  "generated_idea": "...",\n'
        '  "generated_idea_steps": ["Step 1 ...", "Step 2 ..."],\n'
        '  "generated_idea_citations": ["paper_name"]\n'
        "}}\n\n"
        "Rules:\n"
        "- research_gaps MUST contain one gap per selected paper (list format).\n"
        "- generated_idea MUST synthesize all gaps into one concrete solution.\n"
        "- generated_idea_steps MUST be 6-8 detailed steps, including tools/datasets where relevant.\n\n"
        "Papers list (JSON):\n{papers_json}\n\n"
        "Context:\n{context}\n\nQuestion: {question}"
    ),
)

ASSISTANT_QA_PROMPT = PromptTemplate(
    input_variables=["prompt", "chat_history", "context"],
    template=(
        "You are the default assistant for a research system trained on the user's local corpus. "
        "Answer the user's exact prompt using ONLY the supplied context. "
        "If the context is insufficient, say what is missing instead of guessing. "
        "Be direct, accurate, and grounded in the retrieved material.\n\n"
        "Chat history:\n{chat_history}\n\n"
        "Retrieved context:\n{context}\n\n"
        "User prompt: {prompt}"
    ),
)


def research_explainer_chain(llm: BaseLLM) -> Runnable:
    """Build a chain that outputs a strict JSON table and narrative."""
    try:
        return RESEARCH_PROMPT | llm
    except Exception as exc:
        raise RuntimeError(f"Failed to build research explainer chain: {exc}") from exc


def paper_reviewer_chain(llm: BaseLLM) -> Runnable:
    """Build a reviewer chain that outputs structured JSON."""
    try:
        return REVIEW_PROMPT | llm
    except Exception as exc:
        raise RuntimeError(f"Failed to build paper reviewer chain: {exc}") from exc


def paper_qa_chain(llm: BaseLLM) -> Runnable:
    """Build a chain that answers questions about a paper."""
    try:
        return PAPER_QA_PROMPT | llm
    except Exception as exc:
        raise RuntimeError(f"Failed to build paper QA chain: {exc}") from exc


def paper_reviewer_followup_chain(llm: BaseLLM) -> Runnable:
    """Build a chain that answers reviewer follow-up questions in critique mode."""
    try:
        return REVIEW_FOLLOWUP_PROMPT | llm
    except Exception as exc:
        raise RuntimeError(f"Failed to build paper reviewer follow-up chain: {exc}") from exc


def paper_chunk_summarizer_chain(llm: BaseLLM) -> Runnable:
    """Build a chain that summarizes a chunk of a paper."""
    try:
        return PAPER_CHUNK_SUMMARIZER_PROMPT | llm
    except Exception as exc:
        raise RuntimeError(f"Failed to build paper chunk summarizer chain: {exc}") from exc


def reference_generator_chain(llm: BaseLLM) -> Runnable:
    """Build a reference generation chain for APA references."""
    try:
        return REFERENCE_PROMPT | llm
    except Exception as exc:
        raise RuntimeError(f"Failed to build reference generator chain: {exc}") from exc


def json_repair_chain(llm: BaseLLM) -> Runnable:
    """Build a chain that repairs invalid JSON into valid JSON."""
    try:
        return JSON_REPAIR_PROMPT | llm
    except Exception as exc:
        raise RuntimeError(f"Failed to build JSON repair chain: {exc}") from exc


def gap_idea_chain(llm: BaseLLM) -> Runnable:
    """Build a chain that returns plain-text gap + idea lines."""
    try:
        return GAP_IDEA_PROMPT | llm
    except Exception as exc:
        raise RuntimeError(f"Failed to build gap/idea chain: {exc}") from exc


def gap_list_chain(llm: BaseLLM) -> Runnable:
    """Build a chain that returns JSON gaps + idea + steps."""
    try:
        return GAP_LIST_PROMPT | llm
    except Exception as exc:
        raise RuntimeError(f"Failed to build gap list chain: {exc}") from exc


def assistant_answer_chain(llm: BaseLLM) -> Runnable:
    """Build a grounded assistant chain for the trained local corpus."""
    try:
        return ASSISTANT_QA_PROMPT | llm
    except Exception as exc:
        raise RuntimeError(f"Failed to build assistant answer chain: {exc}") from exc
