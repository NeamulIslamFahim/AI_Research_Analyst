
"""LangChain prompt templates and chains."""

from __future__ import annotations

from langchain_classic.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_core.language_models import BaseLLM


RESEARCH_PROMPT = PromptTemplate(
    input_variables=["context", "question", "papers_json"],
    template=(
        "You are a research assistant. Use ONLY the provided context and papers list. "
        "Do NOT mention missing full text in any field. "
        "Return ONLY valid JSON. Do NOT wrap in markdown or code fences. "
        "Do NOT include any extra text. Use double quotes for all strings and no trailing commas. "
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
        "- Choose up to 10-20 relevant papers from the papers list across different sources; if fewer are available, use all.\n"
        "- The output table MUST have the same number of rows as the papers list you select.\n"
        "- For each row, copy paper_name, paper_url, authors_name, and source from the papers list.\n"
        "- If a DOI is present in the papers list, prefer https://doi.org/DOI as paper_url.\n"
        "- problem_solved MUST describe the main problem addressed by the paper.\n"
        "- score_relevance and score_quality MUST be integers from 0 to 10.\n"
        "- If fulltext_available is true, summary_full_paper MUST reflect the full paper content.\n"
        "- If fulltext_available is false, summary_full_paper MUST be an abstract-based summary without stating that full text was missing.\n"
        "- proposed_model_or_approach MUST come from the selected paper's content and include datasets/models if present.\n"
        "- research_gaps MUST contain one gap per selected paper (list format).\n"
        "- assistant_reply MUST be a concise researcher-style response (neutral, evidence-based, no fluff).\n"
        "- generated_idea MUST synthesize all gaps into one concrete solution.\n"
        "- generated_idea_steps MUST be 6-8 detailed steps, including tools/datasets where relevant.\n"
        "- NEVER leave any field blank. If information is missing, write \"Not specified in paper\".\n"
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
        "You are a peer reviewer. Read the paper text and return ONLY valid JSON with fields:\n"
        "{{\n"
        '  "strengths": "...",\n'
        '  "weaknesses": "...",\n'
        '  "novelty": "...",\n'
        '  "technical_correctness": "...",\n'
        '  "reproducibility": "...",\n'
        '  "recommendation": "Accept | Minor Revision | Major Revision | Reject",\n'
        '  "suggested_venue": "Journal | Conference"\n'
        "}}\n\n"
        "Paper text:\n{paper}"
    ),
)

PAPER_QA_PROMPT = PromptTemplate(
    input_variables=["paper_text", "question"],
    template=(
        "You are a research assistant. Answer the user's question based ONLY on the provided paper text.\n\n"
        "Paper text:\n{paper_text}\n\n"
        "Question: {question}"
    ),
)

PAPER_CHUNK_SUMMARIZER_PROMPT = PromptTemplate(
    input_variables=["chunk"],
    template=(
        "You are a research assistant. Summarize the following chunk of a research paper.\n\n"
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
        "You are a research assistant. Use ONLY the provided context and papers list.\n"
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
        "You are a research assistant. Use ONLY the provided context and papers list.\n"
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
def research_explainer_chain(llm: BaseLLM) -> RunnableSequence:
    """Build a chain that outputs a strict JSON table and narrative."""
    try:
        return RESEARCH_PROMPT | llm
    except Exception as exc:
        raise RuntimeError(f"Failed to build research explainer chain: {exc}") from exc


def paper_reviewer_chain(llm: BaseLLM) -> RunnableSequence:
    """Build a reviewer chain that outputs structured JSON."""
    try:
        return REVIEW_PROMPT | llm
    except Exception as exc:
        raise RuntimeError(f"Failed to build paper reviewer chain: {exc}") from exc


def paper_qa_chain(llm: BaseLLM) -> RunnableSequence:
    """Build a chain that answers questions about a paper."""
    try:
        return PAPER_QA_PROMPT | llm
    except Exception as exc:
        raise RuntimeError(f"Failed to build paper QA chain: {exc}") from exc


def paper_chunk_summarizer_chain(llm: BaseLLM) -> RunnableSequence:
    """Build a chain that summarizes a chunk of a paper."""
    try:
        return PAPER_CHUNK_SUMMARIZER_PROMPT | llm
    except Exception as exc:
        raise RuntimeError(f"Failed to build paper chunk summarizer chain: {exc}") from exc


def reference_generator_chain(llm: BaseLLM) -> RunnableSequence:
    """Build a reference generation chain for APA references."""
    try:
        return REFERENCE_PROMPT | llm
    except Exception as exc:
        raise RuntimeError(f"Failed to build reference generator chain: {exc}") from exc


def json_repair_chain(llm: BaseLLM) -> RunnableSequence:
    """Build a chain that repairs invalid JSON into valid JSON."""
    try:
        return JSON_REPAIR_PROMPT | llm
    except Exception as exc:
        raise RuntimeError(f"Failed to build JSON repair chain: {exc}") from exc


def gap_idea_chain(llm: BaseLLM) -> RunnableSequence:
    """Build a chain that returns plain-text gap + idea lines."""
    try:
        return GAP_IDEA_PROMPT | llm
    except Exception as exc:
        raise RuntimeError(f"Failed to build gap/idea chain: {exc}") from exc


def gap_list_chain(llm: BaseLLM) -> RunnableSequence:
    """Build a chain that returns JSON gaps + idea + steps."""
    try:
        return GAP_LIST_PROMPT | llm
    except Exception as exc:
        raise RuntimeError(f"Failed to build gap list chain: {exc}") from exc
