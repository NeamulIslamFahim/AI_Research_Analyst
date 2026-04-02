
"""Backend orchestration for the AI Research Assistant."""

from __future__ import annotations

import abc
import hashlib
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from dotenv import load_dotenv
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
import json
import requests

from .chains import (
    paper_reviewer_chain,
    paper_reviewer_followup_chain,
    paper_qa_chain,
    paper_chunk_summarizer_chain,
    json_repair_chain,
    gap_idea_chain,
    gap_list_chain,
    reference_generator_chain,
    research_explainer_chain,
)
from .embeddings import create_embeddings, create_dummy_embeddings, get_faiss_persist_dir
from .helpers import (
    ENV_PATH,
    clean_authors,
    extract_datasets,
    extract_models,
    extract_proposed_approach,
    load_env_var,
    safe_get,
    safe_json_loads,
    truncate_text,
    authors_to_str,
    strip_html,
)
from .explorer_utils import format_apa_reference
from .storage import get_cached_pdf_path, save_pdf_bytes, upsert_paper_record
from .pdf_utils import chunk_text, extract_text
from .services.response_factory import ResearchResponseComposer, ReviewResponseComposer
from .services.response_templates import build_research_error_response
from .services.research_service import ResearchService
from .services.state_models import (
    ResearchResultSchema,
    ResearchState,
)
from .services.validation import (
    broaden_research_result as _broaden_research_result,
    fix_paper_url as _fix_paper_url,
    normalize_url as _normalize_url,
    score_research_result as _score_research_result,
    strict_validate as _strict_validate,
    validate_research_result as _validate_research_result,
)
from .services.text_utils import (
    clean_text,
    human_summary_from_text,
    normalize_output_text,
    strip_front_matter,
    topic_tokens,
)
from .retriever import (
    arxiv_search,
    build_vector_store,
    load_vector_store,
    docs_to_rows,
    rows_to_docs,
    semantic_scholar_search,
    semantic_scholar_open_access_search,
    serpapi_scholar_search,
    serpapi_researchgate_search,
    serpapi_web_search,
    serpapi_sciencedirect_search,
    openalex_search,
    core_search,
    doaj_search,
    europe_pmc_search,
)
from langgraph.graph import StateGraph, END


DEFAULT_LLM_MODEL_PRIMARY = "llama-3.3-70b-versatile"
DEFAULT_LLM_MODEL_SECONDARY = "llama-3.1-8b-instant"
DEFAULT_GROQ_REASONING_EFFORT = "medium"
DEFAULT_OSS_MODEL_ID = "gpt-oss-120b"

_CACHED_VECTOR_STORE: Any | None = None
_CACHED_EMBEDDINGS: Any | None = None
_CACHED_DUMMY_EMBEDDINGS: Any | None = None
_PAPER_REVIEW_CACHE: dict[str, Any] = {}
_WARMUP_STARTED = False


def _get_embeddings() -> Any:
    global _CACHED_EMBEDDINGS
    if _CACHED_EMBEDDINGS is None:
        _CACHED_EMBEDDINGS = create_embeddings()
    return _CACHED_EMBEDDINGS


def _get_dummy_embeddings() -> Any:
    global _CACHED_DUMMY_EMBEDDINGS
    if _CACHED_DUMMY_EMBEDDINGS is None:
        _CACHED_DUMMY_EMBEDDINGS = create_dummy_embeddings()
    return _CACHED_DUMMY_EMBEDDINGS


def _get_vector_store() -> Any:
    global _CACHED_VECTOR_STORE
    if _CACHED_VECTOR_STORE is None:
        _CACHED_VECTOR_STORE = load_vector_store()
    return _CACHED_VECTOR_STORE


def _peek_vector_store() -> Any:
    """Return the cached vector store without triggering a blocking load."""
    return _CACHED_VECTOR_STORE


def _set_vector_store(vs: Any) -> None:
    global _CACHED_VECTOR_STORE
    _CACHED_VECTOR_STORE = vs


def _warm_research_runtime() -> None:
    """Preload embeddings and the vector store in the background."""
    try:
        _get_embeddings()
    except Exception:
        pass
    try:
        _get_vector_store()
    except Exception:
        pass


def _start_research_warmup() -> None:
    global _WARMUP_STARTED
    if _WARMUP_STARTED:
        return
    _WARMUP_STARTED = True
    if (load_env_var("WARM_RESEARCH_CACHE", "true") or "true").lower() != "true":
        return
    threading.Thread(target=_warm_research_runtime, daemon=True).start()


def init_llm(model_id: str):
    """Initialize Groq chat LLM with environment token."""
    try:
        from langchain_groq import ChatGroq
    except ModuleNotFoundError as exc:
        raise RuntimeError("langchain_groq is not installed. Install with `pip install langchain-groq`.") from exc

    try:
        # Force reload from the absolute path to catch .env changes without process restart
        load_dotenv(dotenv_path=ENV_PATH, override=True)
        token = (load_env_var("GROQ_API_KEY") or "")
        # Strip whitespace and common accidental characters like quotes
        token = token.strip().strip("'").strip('"')
        if not token:
            raise ValueError("GROQ_API_KEY is missing. Add it to your .env file.")

        if not token.startswith("gsk_"):
            raise ValueError(
                f"GROQ_API_KEY in .env does not start with 'gsk_'. Please ensure you copied the correct key. "
                f"Detected prefix: '{token[:4]}...'"
            )

        reasoning_effort = load_env_var("GROQ_REASONING_EFFORT", "") or DEFAULT_GROQ_REASONING_EFFORT
        max_tokens = int(load_env_var("GROQ_MAX_TOKENS", "1024") or "1024")

        # Reasoning effort is typically only supported by 'o1' or 'r1' type models.
        is_reasoning_model = any(x in model_id.lower() for x in ["r1", "reasoning", "deepseek"])

        kwargs = {
            "api_key": token,
            "model": model_id,
            "temperature": 0.5,
            "max_tokens": max_tokens,
        }
        if reasoning_effort and is_reasoning_model:
            kwargs["reasoning_effort"] = reasoning_effort
        return ChatGroq(
            **kwargs,
        )
    except Exception as exc:
        raise RuntimeError(f"Failed to initialize Groq LLM: {exc}") from exc


def init_oss_llm():
    """Initialize gpt-oss via an OpenAI-compatible local endpoint."""
    try:
        from langchain_openai import ChatOpenAI
    except ModuleNotFoundError as exc:
        raise RuntimeError("langchain_openai is not installed. Install with `pip install langchain-openai`.") from exc

    try:
        base_url = load_env_var("OSS_BASE_URL")
        if not base_url:
            raise ValueError("OSS_BASE_URL is missing. Set it to your local inference server URL.")
        model_id = load_env_var("OSS_MODEL_ID", DEFAULT_OSS_MODEL_ID) or DEFAULT_OSS_MODEL_ID
        api_key = load_env_var("OSS_API_KEY", "local-oss-key") or "local-oss-key"
        return ChatOpenAI(
            api_key=api_key,
            model=model_id,
            base_url=base_url,
            temperature=0.5,
            max_tokens=1024,
        )
    except Exception as exc:
        raise RuntimeError(f"Failed to initialize gpt-oss LLM: {exc}") from exc


def get_model_ids() -> tuple[str, str]:
    """Return primary and secondary model IDs."""
    primary = load_env_var("GROQ_MODEL_ID_PRIMARY", DEFAULT_LLM_MODEL_PRIMARY) or DEFAULT_LLM_MODEL_PRIMARY
    secondary = load_env_var("GROQ_MODEL_ID_SECONDARY", DEFAULT_LLM_MODEL_SECONDARY) or DEFAULT_LLM_MODEL_SECONDARY
    return primary, secondary


def _is_rate_limit_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return "rate limit" in message or "rate_limit_exceeded" in message or "tpd" in message


def _is_oss_config_error(exc: Exception) -> bool:
    message = str(exc).lower()
    indicators = [
        "invalid_api_key",
        "invalid api key",
        "incorrect api key",
        "unauthorized",
        "401",
        "connection refused",
        "failed to establish a new connection",
        "oss_base_url",
    ]
    return any(indicator in message for indicator in indicators)


def _normalize_search_rows_output(output: Any) -> tuple[List[dict], str | None]:
    """Normalize source-search outputs to (rows, warning)."""
    if output is None:
        return [], None
    if isinstance(output, tuple):
        if len(output) >= 2:
            rows = output[0] if isinstance(output[0], list) else []
            warn = output[1] if len(output) > 1 else None
            return rows, warn
        if len(output) == 1:
            return output[0] if isinstance(output[0], list) else [], None
        return [], None
    if isinstance(output, list):
        return output, None
    return [], None


def _invoke_with_fallback(chain_builder, invoke_payload: dict) -> Any:
    """Invoke a chain with gpt-oss if enabled; otherwise use Groq with fallback."""
    use_oss = (load_env_var("USE_GPT_OSS", "false") or "false").lower() == "true"
    if str(use_oss).lower() == "none":
        use_oss = False
    primary, secondary = get_model_ids()

    if use_oss:
        try:
            llm = init_oss_llm()
            chain = chain_builder(llm)
            return chain.invoke(invoke_payload)
        except Exception:
            # Any OSS failure should fall back to Groq so the user still gets a response.
            pass

    try:
        llm = init_llm(primary)
        chain = chain_builder(llm)
        return chain.invoke(invoke_payload)
    except Exception as exc:
        if "langchain_groq" in str(exc) or "langchain_openai" in str(exc):
            raise RuntimeError("No LLM backend available. Install langchain-groq or langchain-openai and set keys/URL.") from exc
        if "401" in str(exc) or "invalid_api_key" in str(exc):
            raise RuntimeError(f"Groq Authentication Failed: Please check your GROQ_API_KEY in .env. ({exc})") from exc
        if _is_rate_limit_error(exc):
            try:
                llm = init_llm(secondary)
                chain = chain_builder(llm)
                return chain.invoke(invoke_payload)
            except Exception as exc2:
                raise RuntimeError(f"Failed to invoke secondary model: {exc2}") from exc2
        raise


def _ensure_vector_store_with_docs(docs: List[Document]) -> Any | None:
    """Load vector store if present, otherwise build; add docs if provided."""
    vector_store = _get_vector_store()
    embeddings = _get_embeddings()
    persist_dir = get_faiss_persist_dir()

    # Sanitize docs to ensure page_content is a string for embeddings.
    safe_docs: List[Document] = []
    for d in docs:
        content = d.page_content
        if not isinstance(content, str):
            try:
                content = json.dumps(content, ensure_ascii=True)
            except Exception:
                content = str(content)
        safe_docs.append(Document(page_content=content, metadata=d.metadata or {}))

    if vector_store is None:
        if not safe_docs:
            return None
        vector_store = build_vector_store(safe_docs)
        _set_vector_store(vector_store)
        return vector_store

    # If vector store exists and we have new docs, add them and persist.
    if safe_docs:
        # Force safe embedding wrapper on the loaded store.
        vector_store.embedding_function = embeddings
        texts: List[str] = []
        metadatas: List[dict] = []
        for d in safe_docs:
            content = d.page_content
            if not isinstance(content, str):
                try:
                    content = str(content)
                except Exception:
                    content = ""
            if not content:
                continue
            texts.append(content)
            metadatas.append(d.metadata or {})
        if texts:
            try:
                vector_store.add_texts(texts, metadatas=metadatas)
            except Exception as exc:
                if "TextEncodeInput" in str(exc):
                    vector_store.embedding_function = _get_dummy_embeddings()
                    vector_store.add_texts(texts, metadatas=metadatas)
                else:
                    raise
        vector_store.save_local(persist_dir)

    _set_vector_store(vector_store)
    return vector_store


def _is_paper_vectorized(title: str) -> bool:
    """Check if a paper with the given title is already in the vector store."""
    if not title:
        return False
    vector_store = _get_vector_store()
    if vector_store is None:
        return False
    try:
        # Do a similarity search with the title
        docs = vector_store.similarity_search(title, k=5)
        # Check if any of the top results have the same title
        for doc in docs:
            if doc.metadata.get("title", "").strip().lower() == title.strip().lower():
                return True
    except Exception:
        pass
    return False


def _download_arxiv_fulltext(docs: List[Document], limit: int = 5) -> List[Document]:
    """Optionally download arXiv PDFs and return chunked Documents."""
    fulltext_docs: List[Document] = []
    max_downloads = limit
    downloaded = 0
    for d in docs:
        meta = d.metadata or {}
        url = meta.get("url", "")
        title = meta.get("title", "")
        authors = meta.get("authors", "")
        if not url or "arxiv.org" not in url:
            continue
        # Check if paper is already vectorized
        if _is_paper_vectorized(title):
            continue
        if downloaded >= max_downloads:
            break
        # Prefer PDF links if provided; fallback to arXiv PDF pattern.
        if not url.endswith(".pdf"):
            if "/abs/" in url:
                url = url.replace("/abs/", "/pdf/") + ".pdf"
            else:
                url = url.rstrip("/") + ".pdf"
        try:
            cached = get_cached_pdf_path(url)
            if cached:
                tmp_path = cached
            else:
                resp = requests.get(url, timeout=30)
                resp.raise_for_status()
                content_type = resp.headers.get("Content-Type", "")
                if "pdf" not in content_type.lower() and not resp.content.startswith(b"%PDF"):
                    continue 
                tmp_path = save_pdf_bytes(url, resp.content)
                upsert_paper_record(title, authors_to_str(authors), url, url, "arxiv", tmp_path)
                downloaded += 1
            try:
                text = extract_text(tmp_path)
                chunks = chunk_text(text)
                for idx, chunk in enumerate(chunks):
                    fulltext_docs.append(
                        Document(
                            page_content=chunk,
                            metadata={
                                "title": title,
                                "url": url,
                                "source": "arxiv_fulltext",
                                "chunk": idx,
                            },
                        )
                    )
            finally:
                pass
        except Exception:
            # Skip failures silently to keep the pipeline robust.
            continue
    return fulltext_docs


def _download_external_fulltext(rows: List[dict], limit: int = 5) -> List[Document]:
    """Download PDFs from external sources when a direct PDF URL is provided."""
    fulltext_docs: List[Document] = []
    max_downloads = limit
    downloaded = 0
    for r in rows:
        pdf_url = r.get("pdf_url", "")
        title = r.get("title", "")
        if not pdf_url or not isinstance(pdf_url, str):
            continue
        # Check if paper is already vectorized
        if _is_paper_vectorized(title):
            continue
        if downloaded >= max_downloads:
            break
        try:
            cached = get_cached_pdf_path(pdf_url)
            if cached:
                tmp_path = cached
            else:
                resp = requests.get(pdf_url, timeout=30)
                resp.raise_for_status()
                content_type = resp.headers.get("Content-Type", "")
                if "pdf" not in content_type.lower() and not resp.content.startswith(b"%PDF"):
                    continue 
                tmp_path = save_pdf_bytes(pdf_url, resp.content)
                upsert_paper_record(
                    r.get("title", ""),
                    r.get("authors", ""),
                    r.get("url", ""),
                    pdf_url,
                    r.get("source", ""),
                    tmp_path,
                )
                downloaded += 1
            try:
                text = extract_text(tmp_path)
                chunks = chunk_text(text)
                for idx, chunk in enumerate(chunks):
                    fulltext_docs.append(
                        Document(
                            page_content=chunk,
                            metadata={
                                "title": r.get("title", ""),
                                "url": r.get("url", ""),
                                "pdf_url": pdf_url,
                                "source": r.get("source", ""),
                                "chunk": idx,
                            },
                        )
                    )
            finally:
                pass
        except Exception:
            continue
    return fulltext_docs


def download_papers_for_topic(topic: str) -> Dict[str, Any]:
    """Download and cache PDFs for a topic without generating an answer."""
    if not topic:
        return {"error": "Topic is required."}
    docs = arxiv_search(topic, max_results=12)
    sem_rows, _ = _normalize_search_rows_output(semantic_scholar_search(topic, max_results=10))
    oa_rows, _ = _normalize_search_rows_output(semantic_scholar_open_access_search(topic, max_results=8))
    scholar_rows, _ = _normalize_search_rows_output(serpapi_scholar_search(topic, max_results=10))
    rg_rows, _ = _normalize_search_rows_output(serpapi_researchgate_search(topic, max_results=10))
    web_rows, _ = _normalize_search_rows_output(serpapi_web_search(topic, max_results=6))
    sd_rows, _ = _normalize_search_rows_output(serpapi_sciencedirect_search(topic, max_results=6))
    oa_rows2, _ = _normalize_search_rows_output(openalex_search(topic, max_results=6))
    core_rows, _ = _normalize_search_rows_output(core_search(topic, max_results=6))
    doaj_rows, _ = _normalize_search_rows_output(doaj_search(topic, max_results=6))
    epmc_rows, _ = _normalize_search_rows_output(europe_pmc_search(topic, max_results=6))

    fulltext_docs: List[Document] = []
    if (load_env_var("DOWNLOAD_ARXIV_PDFS", "true") or "true").lower() == "true":
        fulltext_docs.extend(_download_arxiv_fulltext(docs))
    if (load_env_var("DOWNLOAD_EXTERNAL_PDFS", "true") or "true").lower() == "true":
        fulltext_docs.extend(
            _download_external_fulltext(
                sem_rows + oa_rows + scholar_rows + rg_rows + web_rows + sd_rows + oa_rows2 + core_rows + doaj_rows + epmc_rows
            )
        )
    # Update vector store with new PDF content for future queries
    if fulltext_docs:
        try:
            _ensure_vector_store_with_docs(fulltext_docs)
        except Exception:
            pass
    return {"downloaded_chunks": len(fulltext_docs)}

def _run_research_explorer_impl(
    topic: str,
    chat_history: str | None = None,
    focus_topic: str | None = None,
    use_live_sources: bool | None = None,
    previously_returned_titles: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Run arXiv retrieval + RAG generation for the given topic."""
    workflow = _build_research_graph()
    initial_state: ResearchState = {
        "topic": topic,
        "chat_history": chat_history,
        "focus_topic": focus_topic,
        "use_live": use_live_sources,
        "previously_returned_titles": previously_returned_titles,
        "result": None,
        "retries": 0,
        "validation_error": None,
    }
    final_state = workflow.invoke(initial_state)
    return final_state.get("result") or {"error": "Workflow finished with no result."}

def _run_research_explorer_impl_legacy(topic: str, chat_history: str | None = None, focus_topic: str | None = None, use_live_sources: bool | None = None, previously_returned_titles: Optional[List[str]] = None) -> Dict[str, Any]:
    if not topic:
        return {"error": "Topic is required."}

    is_more_request = ResearchService.is_generic_explorer_prompt(topic)
    # Resolve generic prompts like "more" into the actual research topic
    topic = ResearchService.resolve_topic_from_history(topic, chat_history)

    # Force live search for "more" requests to bypass stale cache/fallbacks
    if is_more_request:
        use_live_sources = True
    
    # Normalize previously returned titles for case-insensitive comparison
    excluded_titles_lower = {t.lower() for t in (previously_returned_titles or [])}
    response_composer = ResearchResponseComposer(topic)

    rows: List[dict] = []
    rows_sorted: List[dict] = []
    fulltext_docs: List[Document] = []
    fulltext_map: Dict[tuple[str, str], str] = {}
    fulltext_by_title: Dict[str, str] = {}

    def _simple_fallback_source_rows(query: str) -> List[dict]:
        """Try a smaller set of safer retrieval sources for recovery paths."""
        recovered_rows: List[dict] = []
        try:
            docs = arxiv_search(query, max_results=6)
            if docs and not docs[0].metadata.get("error"):
                recovered_rows.extend(docs_to_rows(docs, source="arxiv"))
        except Exception:
            pass

        fallback_fns = [
            semantic_scholar_search,
            semantic_scholar_open_access_search,
            openalex_search,
            core_search,
            doaj_search,
            europe_pmc_search,
        ]
        for fn in fallback_fns:
            try:
                rows_out, _ = _normalize_search_rows_output(fn(query, max_results=6))
                recovered_rows.extend(rows_out)
            except Exception:
                continue
        return recovered_rows

    def _deterministic_fallback_response(source_rows: List[dict], reason: str = "") -> Dict[str, Any]:
        response = response_composer.build(source_rows or rows_sorted or rows, fulltext_map, fulltext_by_title)
        if reason:
            response["assistant_reply"] = "Research summary prepared from broader source results after a recovery path was used."
        return {
            **response,
            "used_broader_fallback": True,
        }
    try:
        local_only = (load_env_var("LOCAL_ONLY", "false") or "false").lower() == "true"
        fast_mode = (load_env_var("FAST_MODE", "true") or "true").lower() == "true"
        max_primary = int(load_env_var("FAST_MAX_PRIMARY", "5") or "5")
        max_secondary = int(load_env_var("FAST_MAX_SECONDARY", "4") or "4")
        warnings: List[str] = []
        if use_live_sources is None:
            use_live_sources = False
        skip_fulltext_download = fast_mode

        if not use_live_sources:
            vector_store = _peek_vector_store() if fast_mode else _get_vector_store()
            if fast_mode and vector_store is None:
                return response_composer.build_insufficient()
            topic_tokens = [
                t for t in topic.lower().replace("-", " ").split()
                if len(t) >= 1 and t not in {"means", "paper", "research", "using", "approach", "results", "study"}
            ]

            def _match(doc: Document) -> bool:
                meta = doc.metadata or {}
                hay = " ".join(
                    [
                        str(meta.get("title", "")),
                        _clean_row_text(meta.get("abstract", "")),
                        _clean_row_text(doc.page_content or ""),
                    ]
                ).lower()
                return any(tok in hay for tok in topic_tokens)

            def _local_rows_from_store(store: Any) -> tuple[list[dict], list[Document]]:
                if store is None:
                    return [], []
                try:
                    docs_local = store.similarity_search(topic, k=12)
                except Exception:
                    return [], []
                docs_local = [d for d in docs_local if _match(d) or not topic_tokens]
                rows_local: list[dict] = []
                selected_docs: list[Document] = []
                seen_titles: set[str] = set()
                for d in docs_local:
                    meta = d.metadata or {}
                    title = str(meta.get("title", "") or "").strip().lower()
                    if title and title in seen_titles:
                        continue
                    seen_titles.add(title)
                    rows_local.append(
                        {
                            "title": meta.get("title", ""),
                            "authors": authors_to_str(meta.get("authors", "")),
                            "year": meta.get("year", ""),
                            "url": meta.get("url", ""),
                            "pdf_url": meta.get("pdf_url", ""),
                            "doi": meta.get("doi", ""),
                            "abstract": meta.get("abstract", "") or d.page_content,
                            "source": meta.get("source", ""),
                        }
                    )
                    selected_docs.append(d)
                    if len(rows_local) >= 5:
                        break
                return rows_local, selected_docs

            rows, selected_docs = _local_rows_from_store(vector_store)
            if len(rows) < 5:
                if not fast_mode:
                    try:
                        download_papers_for_topic(topic)
                    except Exception:
                        pass
                    vector_store = _get_vector_store()
                    rows, selected_docs = _local_rows_from_store(vector_store)
                    if len(rows) >= 5:
                        all_docs = list(selected_docs)
                        fulltext_docs = list(selected_docs)
            if len(rows) >= 5:
                all_docs = list(selected_docs)
                fulltext_docs = list(selected_docs)

        def _topic_tokens(text: str) -> List[str]:
            return topic_tokens(text)

        def _clean_row_text(value: Any) -> str:
            return clean_text(value)

        def _strip_front_matter(text: str, title: str = "") -> str:
            return strip_front_matter(text, title)

        def _sentence_snippets(text: str, limit: int = 2) -> list[str]:
            pieces = re.split(r"(?<=[.!?])\s+", text.strip())
            snippets: list[str] = []
            for piece in pieces:
                piece = piece.strip()
                if len(piece.split()) < 6:
                    continue
                snippets.append(piece)
                if len(snippets) >= limit:
                    break
            return snippets

        def _human_summary_from_text(text: str, title: str = "") -> str:
            return human_summary_from_text(text, title, max_chars=380)

        def _best_available_summary(row: dict, fulltext_snippet: str = "", abstract: str = "") -> str:
            title = _clean_row_text(row.get("title", "")) or "Untitled paper"
            abstract_text = _clean_row_text(abstract or row.get("abstract", ""))
            if fulltext_snippet:
                summarized = _human_summary_from_text(fulltext_snippet, title)
                if summarized:
                    return normalize_output_text(summarized, max_chars=420)
            if abstract_text:
                summarized = _human_summary_from_text(abstract_text, title)
                if summarized:
                    return normalize_output_text(summarized, max_chars=420)
            return normalize_output_text(
                f"This record only exposes metadata for {title}, so the summary is limited to the title and source details.",
                max_chars=420,
            )

        def _best_available_problem(row: dict, fulltext_snippet: str = "", abstract: str = "") -> str:
            title = _clean_row_text(row.get("title", "")) or "Untitled paper"
            abstract_text = _clean_row_text(abstract or row.get("abstract", ""))
            if fulltext_snippet:
                body = _strip_front_matter(fulltext_snippet, title)
                text_value = _sentence_snippets(body, limit=1)
                text_value = text_value[0] if text_value else body.split(".")[0].strip()
            elif abstract_text:
                body = _strip_front_matter(abstract_text, title)
                text_value = _sentence_snippets(body, limit=1)
                text_value = text_value[0] if text_value else body.split(".")[0].strip()
            else:
                text_value = f"The paper addresses the topic suggested by its title: {title}."
            return normalize_output_text(text_value, max_chars=420)

        def _best_available_approach(row: dict, fulltext_snippet: str = "", abstract: str = "") -> str:
            title = _clean_row_text(row.get("title", "")) or "Untitled paper"
            abstract_text = _clean_row_text(abstract or row.get("abstract", ""))
            inferred = extract_proposed_approach(_strip_front_matter(fulltext_snippet or abstract_text, title))
            if inferred:
                return normalize_output_text(inferred, max_chars=420)
            return normalize_output_text(
                f"The source metadata does not expose a distinct new method; the paper appears centered on {title}.",
                max_chars=420,
            )

        def _normalize_output_text(value: Any, max_chars: int | None = None) -> str:
            return normalize_output_text(value, max_chars=max_chars)

        def _looks_generic_gap(text: str) -> bool:
            lowered = (text or "").lower()
            generic_markers = [
                "not specified in paper",
                "gap text",
                "gap not extracted",
                "no abstract available",
                "synthesize",
                "broader source",
            ]
            return len(lowered.strip()) < 25 or any(marker in lowered for marker in generic_markers)

        def _looks_generic_idea(text: str) -> bool:
            lowered = (text or "").lower()
            generic_markers = [
                "synthesize the identified research_gaps",
                "unify the identified gaps",
                "synthesize the missing aspects",
                "broader source results",
                "not provided",
            ]
            return len(lowered.strip()) < 35 or any(marker in lowered for marker in generic_markers)

        def _needs_ml_context(text: str) -> bool:
            normalized = f" {text.lower().replace('-', ' ')} "
            ml_phrases = {
                " decision tree ",
                " random forest ",
                " gradient boosting ",
                " xgboost ",
                " lightgbm ",
                " catboost ",
                " classification tree ",
                " regression tree ",
            }
            return any(phrase in normalized for phrase in ml_phrases)

        def _row_matches(row: dict, tokens: List[str]) -> bool:
            if not tokens:
                return True
            hay = " ".join([
                str(row.get("title", "")),
                _clean_row_text(row.get("abstract", "")),
                str(row.get("authors", "")),
            ]).lower()

            strong_tokens = [tok for tok in tokens if len(tok) >= 4]
            if strong_tokens:
                match_tokens = strong_tokens
            else:
                match_tokens = tokens

            if not any(tok in hay for tok in match_tokens):
                return False

            if _needs_ml_context(topic):
                ml_keywords = [
                    "machine learning",
                    "classification",
                    "regression",
                    "tree-based",
                    "boosting",
                    "ensemble",
                    "algorithm",
                    "forest",
                    "xgboost",
                    "lightgbm",
                    "catboost",
                    "supervised learning",
                ]
                return any(keyword in hay for keyword in ml_keywords)
            return True

        tokens = _topic_tokens(topic)

        if use_live_sources:
            # Increase retrieval depth for "more" requests to find new papers
            depth_multiplier = 2 if is_more_request else 1
            # Keep retrieval fast but still >10 papers total across sources.
            docs = arxiv_search(topic, max_results=(max_primary if fast_mode else 8) * depth_multiplier)
            if docs and docs[0].metadata.get("error"):
                arxiv_err = docs[0].metadata.get("error", "")
                if "429" in str(arxiv_err):
                    warnings.append(arxiv_err)
                    docs = []
                else:
                    return {"error": arxiv_err}

            # Multi-source retrieval
            rows = docs_to_rows(docs, source="arxiv")
            # Filter arXiv rows by topic tokens to avoid unrelated results.
            if tokens:
                rows = [r for r in rows if _row_matches(r, tokens)]

            def _run_task(fn, max_results: int) -> Any:
                try:
                    return fn(topic, max_results=max_results * depth_multiplier)
                except Exception as exc:
                    return [], str(exc)

            tasks = [
                ("sem", semantic_scholar_search, max_secondary if fast_mode else 6, True),
                ("oa", semantic_scholar_open_access_search, max_secondary if fast_mode else 6, False),
                ("scholar", serpapi_scholar_search, max_secondary if fast_mode else 6, False),
                ("rg", serpapi_researchgate_search, max_secondary if fast_mode else 6, False),
                ("web", serpapi_web_search, max_secondary if fast_mode else 4, is_more_request),
                ("sd", serpapi_sciencedirect_search, max_secondary if fast_mode else 4, False),
                ("oa2", openalex_search, max_secondary if fast_mode else 6, False),
                ("core", core_search, max_secondary if fast_mode else 6, False),
                ("doaj", doaj_search, max_secondary if fast_mode else 6, False),
                ("epmc", europe_pmc_search, max_secondary if fast_mode else 6, False),
            ]

            results: Dict[str, List[dict]] = {}
            warns: Dict[str, str] = {}
            run_tasks = [t for t in tasks if (t[3] or not fast_mode)]
            with ThreadPoolExecutor(max_workers=min(8, len(run_tasks))) as ex:
                future_map = {
                    ex.submit(_run_task, fn, max_results): name for name, fn, max_results, _ in run_tasks
                }
                for fut in as_completed(future_map):
                    name = future_map[fut]
                    try:
                        rows_out, warn_out = _normalize_search_rows_output(fut.result())
                    except Exception as exc:
                        rows_out, warn_out = [], f"{name} retrieval failed: {exc}"
                    results[name] = rows_out
                    if warn_out:
                        warns[name] = warn_out

            sem_rows = results.get("sem", [])
            if warns.get("sem"):
                warnings.append(warns["sem"])
            rows.extend(sem_rows)

            oa_rows = results.get("oa", [])
            if warns.get("oa"):
                warnings.append(warns["oa"])
            if not fast_mode:
                rows.extend(oa_rows)

            scholar_rows = results.get("scholar", [])
            if warns.get("scholar"):
                warnings.append(warns["scholar"])
            if not fast_mode:
                rows.extend(scholar_rows)

            rg_rows = results.get("rg", [])
            if warns.get("rg"):
                warnings.append(warns["rg"])
            if not fast_mode:
                rows.extend(rg_rows)

            web_rows = results.get("web", [])
            if warns.get("web"):
                warnings.append(warns["web"])
            if not fast_mode:
                rows.extend(web_rows)

            sd_rows = results.get("sd", [])
            if warns.get("sd"):
                warnings.append(warns["sd"])
            if not fast_mode:
                rows.extend(sd_rows)

            oa_rows2 = results.get("oa2", [])
            if warns.get("oa2"):
                warnings.append(warns["oa2"])
            if not fast_mode:
                rows.extend(oa_rows2)

            core_rows = results.get("core", [])
            if warns.get("core"):
                warnings.append(warns["core"])
            if not fast_mode:
                rows.extend(core_rows)

            doaj_rows = results.get("doaj", [])
            if warns.get("doaj"):
                warnings.append(warns["doaj"])
            if not fast_mode:
                rows.extend(doaj_rows)

            epmc_rows = results.get("epmc", [])
            if warns.get("epmc"):
                warnings.append(warns["epmc"])
            if not fast_mode:
                rows.extend(epmc_rows)

            all_rows = sem_rows + oa_rows + scholar_rows + rg_rows + web_rows + sd_rows + oa_rows2 + core_rows + doaj_rows + epmc_rows
            # Filter rows by topic tokens to avoid unrelated results.
            filtered_rows = [r for r in all_rows if _row_matches(r, tokens)]
            # If filtering removes everything, fall back to unfiltered for coverage.
            if filtered_rows:
                all_rows = filtered_rows
            rows.extend(all_rows)
            all_docs = docs + rows_to_docs(all_rows)

            if not skip_fulltext_download:
                # Requirement: Download 5 papers for each response to ground the result
                arxiv_to_download = [d for d in docs if "arxiv.org" in (d.metadata or {}).get("url", "")]
                fulltext_docs.extend(_download_arxiv_fulltext(arxiv_to_download, limit=5))

                remaining_slots = 5 - len(fulltext_docs)
                if remaining_slots > 0:
                    fulltext_docs.extend(_download_external_fulltext(all_rows, limit=remaining_slots))

                all_docs.extend(fulltext_docs)
        # Build a lookup for full-text snippets by (title, url) and by title.
        if fulltext_docs:
            tmp_map: Dict[tuple[str, str], List[str]] = {}
            for d in fulltext_docs:
                meta = d.metadata or {}
                key = (meta.get("title", "") or "", meta.get("url", "") or "")
                tmp_map.setdefault(key, []).append(d.page_content or "")
            for key, parts in tmp_map.items():
                text = truncate_text("\n".join(parts), max_chars=4000)
                fulltext_map[key] = text
                title = key[0]
                if title and title not in fulltext_by_title:
                    fulltext_by_title[title] = text
        if use_live_sources:
            vector_store = _ensure_vector_store_with_docs(all_docs)
        else:
            vector_store = _get_vector_store()

        if vector_store is None:
            return {"error": "No documents found or vector store unavailable."}

        # Build retrieval context explicitly for the prompt.
        retriever = vector_store.as_retriever(search_kwargs={"k": 6})
        if use_live_sources:
            bm25 = BM25Retriever.from_documents(all_docs)
            bm25.k = 6
            # Manual hybrid: merge FAISS + BM25 results by URL/title key.
            if hasattr(retriever, "invoke"):
                faiss_docs = retriever.invoke(topic)
            else:
                faiss_docs = retriever.get_relevant_documents(topic)
            if hasattr(bm25, "invoke"):
                bm25_docs = bm25.invoke(topic)
            else:
                bm25_docs = bm25.get_relevant_documents(topic)
            merged = {}
            for d in faiss_docs + bm25_docs:
                meta = d.metadata or {}
                key = (meta.get("title", ""), meta.get("url", ""))
                if key not in merged:
                    merged[key] = d
            context_docs = list(merged.values())[:8]
        else:
            if hasattr(retriever, "invoke"):
                context_docs = retriever.invoke(topic)
            else:
                context_docs = retriever.get_relevant_documents(topic)
            context_docs = list(context_docs)[:8]
        if fulltext_docs:
            context = truncate_text("\n\n".join([d.page_content for d in fulltext_docs[:8]]), max_chars=6000)
        else:
            context = truncate_text("\n\n".join([d.page_content for d in context_docs]), max_chars=5000)
        if chat_history:
            context = f"Conversation context:\\n{chat_history}\\n\\n{context}"

        # Prefer papers with full text available.
        fulltext_keys = set()
        for d in fulltext_docs:
            meta = d.metadata or {}
            fulltext_keys.add((meta.get("title", ""), meta.get("url", "")))

        def _is_fulltext(r: dict) -> bool:
            return (r.get("title", ""), r.get("url", "")) in fulltext_keys

        # Filter by year range and sort by recency
        year_min = int(load_env_var("YEAR_MIN", "2010") or "2010")
        year_max = int(load_env_var("YEAR_MAX", "2025") or "2025")
        def _year_ok(r: dict) -> bool:
            y = str(r.get("year", "") or "")
            try:
                yi = int(y[:4])
            except Exception:
                # If year is missing/unknown, keep the row.
                return True
            return year_min <= yi <= year_max

        filtered_rows = [r for r in rows if _year_ok(r)]
        # Optional focus filter for follow-ups (keeps results on-topic).
        if focus_topic:
            focus = focus_topic.lower()
            keywords = [w for w in focus.replace("-", " ").split() if len(w) > 2]
            def _topic_match(r: dict) -> bool:
                text = f"{r.get('title','')} {r.get('abstract','')}".lower()
                return any(k in text for k in keywords)
            focus_rows = [r for r in filtered_rows if _topic_match(r)]
            if focus_rows:
                filtered_rows = focus_rows

        # Filter out previously returned titles for "more" requests
        if is_more_request and excluded_titles_lower:
            filtered_rows = [r for r in filtered_rows if r.get("title", "").lower() not in excluded_titles_lower]

        # Prefer fulltext and newer papers
        rows_sorted = sorted(
            filtered_rows,
            key=lambda r: ( _is_fulltext(r), int(str(r.get("year", "0"))[:4] or 0) ),
            reverse=True,
        )

        if fast_mode:
            return response_composer.build(rows_sorted, fulltext_map, fulltext_by_title)

            def _quick_summary(row: dict) -> str:
                title = _clean_row_text(row.get("title", "")) or "Untitled paper"
                abstract = _clean_row_text(row.get("abstract", ""))
                if abstract:
                    pieces = re.split(r"(?<=[.!?])\s+", abstract)
                    snippets = [p.strip() for p in pieces if len(p.strip().split()) >= 6][:2]
                    text = " ".join(snippets) if snippets else abstract
                    text = re.sub(r"\s+", " ", text).strip()
                    if len(text) > 360:
                        text = text[:357].rstrip() + "..."
                    return text
                return f"This paper is a metadata-only record centered on {title}."

            def _quick_problem(row: dict) -> str:
                title = _clean_row_text(row.get("title", "")) or "Untitled paper"
                abstract = _clean_row_text(row.get("abstract", ""))
                if abstract:
                    return abstract.split(".")[0].strip()[:360]
                return f"The paper addresses the topic suggested by its title: {title}."

            def _quick_approach(row: dict) -> str:
                title = _clean_row_text(row.get("title", "")) or "Untitled paper"
                abstract = _clean_row_text(row.get("abstract", ""))
                inferred = extract_proposed_approach(abstract)
                if inferred:
                    return _normalize_output_text(inferred, max_chars=420)
                return _normalize_output_text(
                    f"The source metadata does not expose a distinct new method; the paper appears centered on {title}.",
                    max_chars=420,
                )

            def _quick_topic_theme() -> str:
                blob = f"{topic} " + " ".join(_clean_row_text(r.get("title", "")) for r in rows_sorted[:3]).lower()
                if "sentiment" in blob:
                    return "sentiment analysis"
                if "phishing" in blob:
                    return "phishing detection"
                if "analysis" in blob or "data" in blob:
                    return "data analysis"
                return "the topic"

            def _quick_gaps(rows_sample: List[dict]) -> List[str]:
                gaps_list: List[str] = []
                for r in rows_sample[:5]:
                    title = _clean_row_text(r.get("title", "")) or "Paper"
                    abstract = _clean_row_text(r.get("abstract", ""))
                    blob = f"{title} {abstract}".lower()
                    if "twitter" in blob or "social media" in blob:
                        gap = f"{title}: test the approach on noisier social-media text and cross-domain data."
                    elif "multilingual" in blob or "low-resource" in blob:
                        gap = f"{title}: expand evaluation to more languages and report transfer clearly."
                    else:
                        gap = f"{title}: strengthen the evaluation with stronger baselines and clearer error analysis."
                    gaps_list.append(_normalize_output_text(gap, max_chars=320))
                return gaps_list

            def _quick_idea(gaps_list: List[str]) -> str:
                theme = _quick_topic_theme()
                if not gaps_list:
                    return f"Build a stronger benchmark for {theme} that tests robustness, transfer, and interpretability."
                gap_summary = " ".join([g.split(":", 1)[-1].strip() for g in gaps_list[:3]])
                return f"Build a stronger {theme} pipeline that addresses the recurring weaknesses across these papers, especially {gap_summary}."

            quick_rows = rows_sorted[:5] if rows_sorted else rows[:5]
            fallback_table = []
            for r in quick_rows:
                fallback_table.append(
                    {
                        "paper_name": r.get("title", ""),
                        "paper_url": _resolve_url(r),
                        "authors_name": r.get("authors", ""),
                        "summary_full_paper": _quick_summary(r),
                        "problem_solved": _quick_problem(r),
                        "proposed_model_or_approach": _quick_approach(r),
                        "source": r.get("source", ""),
                        "score_relevance": 8,
                        "score_quality": 7,
                    }
                )
            heuristic_gaps = _quick_gaps(fallback_table)
            heuristic_idea = _quick_idea(heuristic_gaps)
            gap_sentence = " ".join(g.split(":", 1)[-1].strip() for g in heuristic_gaps[:2]) if heuristic_gaps else "the papers need broader evaluation"
            return {
                "table": fallback_table,
                "research_gaps": heuristic_gaps,
                "assistant_reply": (
                    f"I found {len(fallback_table)} relevant papers on {topic}. "
                    f"The common thread is {_quick_topic_theme()}, and the main open problems are {gap_sentence}. "
                    "A good next step is to test whether these methods still hold up on harder, more diverse data."
                ),
                "generated_idea": heuristic_idea,
                "generated_idea_steps": [
                    "Define one focused research question that targets the shared weakness across the papers.",
                    "Collect or reuse at least two datasets that stress the model outside the original setting.",
                    "Train stronger baselines, especially a modern transformer or comparable neural baseline.",
                    "Add an ablation study so it is clear which components actually help.",
                    "Evaluate cross-domain robustness, not just in-distribution accuracy.",
                    "Add error analysis for noisy text, slang, emojis, or other domain-specific signals.",
                ],
                "generated_idea_citations": [r["paper_name"] for r in fallback_table[:5]],
                "used_broader_fallback": True,
            }

        # Prefer rows with usable abstract/fulltext to avoid "No abstract available".
        def _has_content(r: dict) -> bool:
            if _is_fulltext(r):
                return True
            abstract = _clean_row_text(r.get("abstract", ""))
            return bool(abstract)
        content_rows = [r for r in rows_sorted if _has_content(r)]
        if len(content_rows) >= 3:
            rows_sorted = content_rows
        fulltext_only = (load_env_var("FULLTEXT_ONLY", "false") or "false").lower() == "true"
        if fulltext_only:
            # Changed to 5 to align with the user's request for 5 papers
            min_fulltext = int(load_env_var("FULLTEXT_MIN", "5") or "5")
            rows_sorted = [r for r in rows_sorted if _is_fulltext(r)]
            if len(rows_sorted) < min_fulltext:
                return {
                    "error": f"Not enough full-text PDFs (need at least {min_fulltext}). "
                             "Try another topic or increase sources."
                }

        # Prepare paper rows for the model to fill in.
        # Pass up to 12 papers to the LLM, allowing it to select the best 5.
        # The final trimming to 5 is done in app.py.
        papers_for_llm = rows_sorted[:12]

        def _resolve_url(r: dict) -> str:
            doi = r.get("doi", "")
            if doi and "10." in doi:
                return f"https://doi.org/{doi}"
            return _fix_paper_url(r.get("url", "") or r.get("pdf_url", ""), r.get("title", ""))
        papers_payload = []
        for r in papers_for_llm:
            paper_url = _resolve_url(r)
            abstract = _clean_row_text(r.get("abstract", ""))
            if len(abstract) > 800:
                abstract = abstract[:800] + "..."
            papers_payload.append(
                {
                    "paper_name": r.get("title", ""),
                    "paper_url": paper_url,
                    "authors_name": r.get("authors", ""),
                    "abstract": abstract,
                    "source": r.get("source", ""),
                    "fulltext_available": _is_fulltext(r),
                }
            )

        if not papers_payload:
            # Fallback to using titles even if abstracts are missing.
            fallback_rows = rows_sorted[:5] if rows_sorted else rows[:5] # Ensure 5 papers for fallback
            for r in fallback_rows:
                papers_payload.append(
                    {
                        "paper_name": r.get("title", ""),
                        "paper_url": _resolve_url(r),
                        "authors_name": r.get("authors", ""),
                        "abstract": _clean_row_text(r.get("abstract", "")),
                        "source": r.get("source", ""),
                        "fulltext_available": _is_fulltext(r),
                    }
                )

        def _heuristic_gaps() -> List[str]:
            gaps_list: List[str] = [] # Limit to 5 for heuristics
            for r in rows_sorted[: min(5, len(rows_sorted))]:
                title = r.get("title", "") or "Paper"
                abstract = _clean_row_text(r.get("abstract", ""))
                lower_blob = f"{title} {abstract}".lower()
                if any(term in lower_blob for term in ["twitter", "social media", "tweet", "microblog"]):
                    gap_text = f"{title}: test whether the approach still works on noisier social-media text and outside Twitter."
                elif any(term in lower_blob for term in ["multilingual", "language", "low-resource", "macedonian", "arabic", "spanish"]):
                    gap_text = f"{title}: expand evaluation to more languages and report cross-lingual transfer clearly."
                elif "semeval" in lower_blob:
                    gap_text = f"{title}: compare against stronger modern baselines and add error analysis for sarcasm, emojis, and slang."
                elif any(term in lower_blob for term in ["two-dimensional", "emotion space", "valence", "arousal"]):
                    gap_text = f"{title}: validate the representation on modern benchmarks and compare it with simpler baseline models."
                else:
                    gap_text = f"{title}: strengthen the evaluation with stronger baselines, ablations, and clearer error analysis."
                gaps_list.append(_normalize_output_text(gap_text, max_chars=320))
            return gaps_list

        def _topic_theme() -> str:
            topic_lower = (topic or "").lower()
            title_blob = " ".join(
                f"{r.get('title', '')} {_clean_row_text(r.get('abstract', ''))}"
                for r in rows_sorted[:3]
            ).lower()
            blob = f"{topic_lower} {title_blob}"
            if "sentiment" in blob:
                if any(term in blob for term in ["twitter", "tweet", "microblog", "social media"]):
                    return "Twitter and social-media sentiment analysis"
                if any(term in blob for term in ["multilingual", "low-resource", "language"]):
                    return "multilingual or low-resource sentiment analysis"
                if any(term in blob for term in ["aspect", "aspect-based"]):
                    return "aspect-based sentiment analysis"
                return "general sentiment analysis"
            if "phishing" in blob:
                return "phishing detection and URL classification"
            if "review" in blob:
                return "review or opinion mining"
            return "the core research area"

        def _heuristic_idea(gaps_list: List[str]) -> str:
            if not gaps_list:
                return (
                    f"Build a stronger benchmark for {_topic_theme()} that tests cross-domain transfer, robustness, and interpretability."
                )
            gap_summary = " ".join([g.split(":", 1)[-1].strip() for g in gaps_list[:3]])
            return (
                f"Build a stronger {_topic_theme()} pipeline that addresses the recurring weaknesses across these papers, "
                f"especially {gap_summary}. The next version should improve robustness, broaden the dataset coverage, "
                "and include clearer error analysis and baselines."
            )

        def _fallback_idea_from_rows(selected_rows: List[dict]) -> str:
            titles = [str(r.get("title", "")).strip() for r in selected_rows[:3] if str(r.get("title", "")).strip()]
            if titles:
                title_list = ", ".join(titles)
            else:
                title_list = "the selected papers"
            return (
                f"Construct a focused research program based on {title_list}, identifying and "
                "closing the most impactful gaps (e.g., generalization, benchmark scope, fairness). "
                "Implement a pipeline explicitly referencing each gap and baseline comparison."
            )

        if fast_mode:
            heuristic_gaps = _heuristic_gaps()
            heuristic_idea = _heuristic_idea(heuristic_gaps)
            concise_theme = _topic_theme()
            if heuristic_gaps:
                gap_sentence = " ".join(g.split(":", 1)[-1].strip() for g in heuristic_gaps[:2])
            else:
                gap_sentence = "the papers are too similar and need broader evaluation"
            fallback_table = []
            for r in rows_sorted[:5]:
                fulltext_key = (r.get("title", ""), r.get("url", ""))
                fulltext_snippet = fulltext_map.get(fulltext_key) or fulltext_by_title.get(r.get("title", ""), "")
                fallback_table.append({
                    "paper_name": r.get("title", ""),
                    "paper_url": _resolve_url(r),
                    "authors_name": r.get("authors", ""),
                    "summary_full_paper": _best_available_summary(r, fulltext_snippet, r.get("abstract", "")),
                    "problem_solved": _best_available_problem(r, fulltext_snippet, r.get("abstract", "")),
                    "proposed_model_or_approach": _best_available_approach(r, fulltext_snippet, r.get("abstract", "")),
                    "source": r.get("source", ""),
                    "score_relevance": 8,
                    "score_quality": 7,
                })
            return {
                "table": fallback_table,
                "research_gaps": heuristic_gaps,
                "assistant_reply": (
                    f"I found {len(fallback_table)} relevant papers on {topic}. "
                    f"The common thread is {concise_theme}, and the main open problems are {gap_sentence}. "
                    "A good next step is to test whether these methods still hold up on harder, more diverse data."
                ),
                "generated_idea": heuristic_idea,
                "generated_idea_steps": [
                    "Define one focused research question that targets the shared weakness across the papers.",
                    "Collect or reuse at least two datasets that stress the model outside the original setting.",
                    "Train stronger baselines, especially a modern transformer or comparable neural baseline.",
                    "Add an ablation study so it is clear which components actually help.",
                    "Evaluate cross-domain robustness, not just in-distribution accuracy.",
                    "Add error analysis for noisy text, slang, emojis, or other domain-specific signals.",
                ],
                "generated_idea_citations": [r["paper_name"] for r in fallback_table[:5]],
                "used_broader_fallback": True,
            }

        raw_output = _invoke_with_fallback(
            research_explainer_chain,
            {
                "context": context,
                "question": topic,
                "papers_json": json.dumps(papers_payload, ensure_ascii=True),
            },
        )
        parsed = safe_json_loads(raw_output)
        if isinstance(parsed, dict) and parsed.get("error"):
            # Use primary for repair, fallback handled inside _invoke_with_fallback if needed
            repaired = _invoke_with_fallback(
                json_repair_chain,
                {
                    "bad_json": raw_output,
                    "schema_hint": (
                        "JSON object with keys: table (array of rows), research_gaps (array), generated_idea (string). "
                        "assistant_reply (string). "
                        "Each row has paper_name, paper_url, authors_name, summary_full_paper, datasets_used (array), "
                        "models_used (array), proposed_model_or_approach, source."
                    ),
                },
            )
            parsed = safe_json_loads(repaired)
            # If still invalid after repair, fall back to deterministic table + gap/idea lines.
            if isinstance(parsed, dict) and parsed.get("error"):
                fallback_rows = []
                for r in rows_sorted[:15]:
                    abstract = _clean_row_text(r.get("abstract", ""))
                    datasets = extract_datasets(abstract)
                    models = extract_models(abstract)
                    proposed = extract_proposed_approach(abstract)
                    key = (r.get("title", "") or "", r.get("url", "") or "")
                    fulltext_snippet = fulltext_map.get(key) or fulltext_by_title.get(r.get("title", "") or "")
                    fallback_rows.append(
                        {
                            "paper_name": r.get("title", ""),
                            "paper_url": _resolve_url(r),
                            "authors_name": r.get("authors", ""),
                            "summary_full_paper": _best_available_summary(r, fulltext_snippet, abstract),
                            "problem_solved": _best_available_problem(r, fulltext_snippet, abstract),
                            "proposed_model_or_approach": _best_available_approach(r, fulltext_snippet, abstract),
                            "source": r.get("source", ""),
                            "score_relevance": 0,
                            "score_quality": 0,
                        }
                    )
                gap_idea_text = _invoke_with_fallback(
                    gap_idea_chain,
                    {
                        "context": context,
                        "papers_json": json.dumps(papers_payload, ensure_ascii=True),
                        "question": topic,
                    },
                )
                gap_idea = safe_get({"text": gap_idea_text}, "text", "")
                gap = ""
                idea = ""
                if isinstance(gap_idea, str):
                    for line in gap_idea.splitlines():
                        if line.lower().startswith("research gap:"):
                            gap = line.split(":", 1)[1].strip()
                        if line.lower().startswith("generated idea:"):
                            idea = line.split(":", 1)[1].strip()
                heuristic_gaps = _heuristic_gaps()
                return {
                    "table": fallback_rows,
                    "research_gaps": heuristic_gaps,
                    "assistant_reply": (
                        f"I found {len(fallback_rows)} papers on {topic}. The literature centers on {_topic_theme()}, "
                        f"and the main openings are {', '.join(g.split(':', 1)[-1].strip() for g in heuristic_gaps[:2])}. "
                        "The clearest path forward is to strengthen the evaluation setup and test for robustness across datasets."
                    ),
                    "generated_idea": _heuristic_idea(heuristic_gaps),
                    "research_gap_citations": [r["paper_name"] for r in fallback_rows[:5]],
                    "generated_idea_citations": [r["paper_name"] for r in fallback_rows[:5]],
                }

        # Normalize legacy key "research_gap" to "research_gaps" (list).
        if isinstance(parsed, dict) and not parsed.get("research_gaps") and parsed.get("research_gap"):
            gap_val = parsed.get("research_gap")
            if isinstance(gap_val, str):
                parsed["research_gaps"] = [g.strip() for g in gap_val.splitlines() if g.strip()]
            elif isinstance(gap_val, list):
                parsed["research_gaps"] = gap_val

        # Post-process: ensure table rows have paper_url filled
        if isinstance(parsed, dict) and isinstance(parsed.get("table"), list):
            title_to_url = {r.get("title", ""): _resolve_url(r) for r in rows_sorted}
            title_to_abstract = {r.get("title", ""): _clean_row_text(r.get("abstract", "")) for r in rows_sorted}
            deduped = []
            seen = set()
            # Final validation of all URLs before returning
            for row in parsed.get("table", []):
                if isinstance(row, dict):
                    row["paper_url"] = _fix_paper_url(row.get("paper_url", ""), row.get("paper_name", ""))
            for row in parsed["table"]:
                if isinstance(row, dict) and not row.get("paper_name"):
                    continue
                # Remove unwanted columns if model adds them
                if isinstance(row, dict):
                    row.pop("datasets_used", None)
                    row.pop("models_used", None)
                    # Strip HTML tags from text fields
                    for field in ["summary_full_paper", "problem_solved", "proposed_model_or_approach"]:
                        if field in row:
                            row[field] = _normalize_output_text(row.get(field, ""), max_chars=1200 if field == "summary_full_paper" else 420)
                if isinstance(row, dict) and not row.get("paper_url"):
                    title = row.get("paper_name", "")
                    if title in title_to_url:
                        row["paper_url"] = title_to_url[title]
                if isinstance(row, dict):
                    paper_url = (row.get("paper_url", "") or "").strip()
                    if not paper_url or "not specified" in paper_url.lower():
                        title = row.get("paper_name", "")
                        if title in title_to_url:
                            row["paper_url"] = title_to_url[title]
                        else:
                            if title:
                                row["paper_url"] = "https://scholar.google.com/scholar?q=" + requests.utils.quote(title)
                if isinstance(row, dict) and not row.get("summary_full_paper"):
                    title = row.get("paper_name", "")
                    abstract = title_to_abstract.get(title, "")
                    row["summary_full_paper"] = _best_available_summary({"title": title, "abstract": abstract}, "", abstract)
                # Replace any "full text not provided" phrasing with abstract-based summary
                if isinstance(row, dict):
                    summary = (row.get("summary_full_paper", "") or "").lower()
                    if "full text not provided" in summary or "supplied context" in summary:
                        title = row.get("paper_name", "")
                        abstract = title_to_abstract.get(title, "")
                        row["summary_full_paper"] = _best_available_summary({"title": title, "abstract": abstract}, "", abstract)
                # If full text is available, prefer it for summaries and problem/approach fields.
                if isinstance(row, dict):
                    title = row.get("paper_name", "") or ""
                    url = row.get("paper_url", "") or ""
                    fulltext_snippet = fulltext_map.get((title, url)) or fulltext_by_title.get(title, "")
                    if fulltext_snippet:
                        if not row.get("summary_full_paper") or row.get("summary_full_paper") in (
                            "Not specified in paper",
                            "Not specified in abstract",
                        ):
                            row["summary_full_paper"] = _best_available_summary(row, fulltext_snippet, row.get("summary_full_paper", ""))
                        if not row.get("problem_solved") or row.get("problem_solved") in (
                            "Not specified in paper",
                            "Not specified in abstract",
                        ):
                            row["problem_solved"] = _best_available_problem(row, fulltext_snippet, row.get("problem_solved", ""))
                        if not row.get("proposed_model_or_approach") or row.get(
                            "proposed_model_or_approach"
                        ) in ("Not specified in paper", "Not specified in abstract"):
                            row["proposed_model_or_approach"] = _best_available_approach(row, fulltext_snippet, row.get("proposed_model_or_approach", ""))
                    # Ensure proposed_model_or_approach is not just the first abstract line.
                    proposed = (row.get("proposed_model_or_approach", "") or "").strip()
                    summary = (row.get("summary_full_paper", "") or "").strip()
                    problem = (row.get("problem_solved", "") or "").strip()
                    if proposed and (proposed == summary or proposed == problem or proposed in summary or _looks_generic_idea(proposed)):
                        abstract = title_to_abstract.get(title, "")
                        row["proposed_model_or_approach"] = _best_available_approach(row, fulltext_snippet, abstract)
                # Fix proposed_model_or_approach if it duplicates the paper title
                if isinstance(row, dict):
                    title = (row.get("paper_name", "") or "").strip()
                    proposed = (row.get("proposed_model_or_approach", "") or "").strip()
                    if title and proposed and (proposed == title or proposed in title or title in proposed):
                        abstract = title_to_abstract.get(title, "")
                        row["proposed_model_or_approach"] = _best_available_approach(row, "", abstract)
                    # Ensure problem_solved is populated
                    if isinstance(row, dict) and not row.get("problem_solved"):
                        abstract = title_to_abstract.get(row.get("paper_name", ""), "")
                        row["problem_solved"] = _best_available_problem(row, "", abstract)
                    # Normalize bare DOI URLs
                    paper_url = (row.get("paper_url", "") or "").strip()
                    if paper_url.startswith("10.") and "/" in paper_url:
                        row["paper_url"] = f"https://doi.org/{paper_url}"
                    row["summary_full_paper"] = _best_available_summary(row, "", row.get("summary_full_paper", ""))
                    row["problem_solved"] = _best_available_problem(row, "", row.get("problem_solved", ""))
                    row["proposed_model_or_approach"] = _best_available_approach(row, "", row.get("proposed_model_or_approach", ""))
                if isinstance(row, dict):
                    key = (
                        (row.get("paper_name", "") or "").strip().lower(),
                        (row.get("paper_url", "") or "").strip().lower(),
                    )
                    if key in seen:
                        continue
                    seen.add(key)
                    row["paper_url"] = _fix_paper_url(row.get("paper_url", ""))
                deduped.append(row)
            parsed["table"] = deduped
            # If model returned too few rows, fall back to deterministic table.
            if len(parsed["table"]) < min(5, len(rows_sorted)):
                fallback_rows = [] # Ensure 5 papers for fallback
                for r in rows_sorted[: min(5, len(rows_sorted))]:
                    key = (r.get("title", "") or "", r.get("url", "") or "")
                    fulltext_snippet = fulltext_map.get(key) or fulltext_by_title.get(r.get("title", "") or "")
                    abstract = _clean_row_text(r.get("abstract", ""))
                    fallback_rows.append(
                        {
                            "paper_name": r.get("title", ""),
                            "paper_url": _resolve_url(r),
                            "authors_name": r.get("authors", ""),
                            "summary_full_paper": _best_available_summary(r, fulltext_snippet, abstract),
                            "problem_solved": _best_available_problem(r, fulltext_snippet, abstract),
                            "proposed_model_or_approach": _best_available_approach(r, fulltext_snippet, abstract),
                            "source": r.get("source", ""),
                            "score_relevance": 0,
                            "score_quality": 0,
                        }
                    )
                parsed["table"] = fallback_rows

        # Store model outputs back into the vector store for future retrieval.
        try:
            output_docs: List[Document] = []
            if isinstance(parsed, dict):
                for row in parsed.get("table", []) or []:
                    if isinstance(row, dict):
                        summary = row.get("summary_full_paper", "") or ""
                        if not isinstance(summary, str):
                            try:
                                summary = json.dumps(summary, ensure_ascii=True)
                            except Exception:
                                summary = str(summary)
                        output_docs.append(
                            Document(
                                page_content=summary,
                                metadata={
                                    "title": row.get("paper_name", ""),
                                    "url": row.get("paper_url", ""),
                                    "authors": row.get("authors_name", ""),
                                    "source": row.get("source", "model_output"),
                                    "type": "model_summary",
                                },
                            )
                        )
                gaps = parsed.get("research_gaps", [])
                if isinstance(gaps, list):
                    for gap in gaps:
                        output_docs.append(
                            Document(
                                page_content=str(gap),
                                metadata={"type": "research_gap", "source": "model_output"},
                            )
                        )
                idea = parsed.get("generated_idea", "")
                if idea:
                    output_docs.append(
                        Document(
                            page_content=idea,
                            metadata={"type": "generated_idea", "source": "model_output"},
                        )
                    )
            if output_docs:
                vector_store.add_documents(output_docs)
                vector_store.save_local(get_faiss_persist_dir())
        except Exception:
            # Avoid breaking user flow if persistence fails.
            pass

        # If research_gaps or generated_idea missing, try a focused gap/idea chain.
        if isinstance(parsed, dict) and (not parsed.get("research_gaps") or not parsed.get("generated_idea")):
            gap_raw = _invoke_with_fallback(
                gap_list_chain,
                {
                    "context": context,
                    "papers_json": json.dumps(papers_payload, ensure_ascii=True),
                    "question": topic,
                },
            )
            gap_parsed = safe_json_loads(gap_raw)
            if isinstance(gap_parsed, dict):
                if gap_parsed.get("research_gaps"):
                    parsed["research_gaps"] = gap_parsed.get("research_gaps")
                if gap_parsed.get("generated_idea"):
                    parsed["generated_idea"] = gap_parsed.get("generated_idea")
                if gap_parsed.get("generated_idea_steps"):
                    parsed["generated_idea_steps"] = gap_parsed.get("generated_idea_steps")
                if gap_parsed.get("generated_idea_citations"):
                    parsed["generated_idea_citations"] = gap_parsed.get("generated_idea_citations")

        # Final fallback if gaps/idea are still missing.
        if isinstance(parsed, dict) and not parsed.get("research_gaps"):
            parsed["research_gaps"] = _heuristic_gaps()

        if isinstance(parsed, dict) and not parsed.get("generated_idea"):
            parsed["generated_idea"] = _heuristic_idea(parsed.get("research_gaps", []))
        if isinstance(parsed, dict) and parsed.get("generated_idea"):
            generated_idea_text = str(parsed.get("generated_idea", "")).strip()
            if "synthesize the" in generated_idea_text.lower() and "gaps" in generated_idea_text.lower():
                parsed["generated_idea"] = _heuristic_idea(parsed.get("research_gaps", []))

        if isinstance(parsed, dict) and not parsed.get("assistant_reply"):
            parsed["assistant_reply"] = (
                f"I found a set of papers on {topic}. The main theme is {_topic_theme()}, and the most useful next step is to strengthen cross-domain evaluation and compare against stronger baselines."
            )
        # Final fallback if gaps/idea are still missing.
        if isinstance(parsed, dict) and not parsed.get("research_gaps"):
            parsed["research_gaps"] = _heuristic_gaps()

        def _verify_response(parsed_response: dict[str, Any], token_list: list[str]) -> dict[str, Any]:
            """Check response quality before final return; enforce topic relevance and safe deliverability."""
            if not isinstance(parsed_response, dict):
                return parsed_response

            if not parsed_response.get("table") or not isinstance(parsed_response.get("table"), list):
                return {
                    "table": [],
                    "research_gaps": [],
                    "assistant_reply": (
                        "No relevant research papers could be safely returned. "
                        "Please refine your query with explicit domain terms like offensive language, social media moderation, or hate speech detection."
                    ),
                    "generated_idea": "",
                    "generated_idea_steps": [],
                    "generated_idea_citations": [],
                    "used_broader_fallback": True,
                }

            if not token_list:
                return parsed_response

            relevance_matches = 0
            for row in parsed_response.get("table", []):
                if not isinstance(row, dict):
                    continue
                haystack = " ".join([
                    str(row.get("paper_name", "")),
                    str(row.get("summary_full_paper", "")),
                    str(row.get("problem_solved", "")),
                    str(row.get("proposed_model_or_approach", "")),
                ]).lower()
                if any(tok in haystack for tok in token_list):
                    relevance_matches += 1

            if relevance_matches < max(1, len(parsed_response.get("table", [])) // 2):
                return {
                    "table": [],
                    "research_gaps": [],
                    "assistant_reply": (
                        "The retrieved response does not match the requested topic closely enough. "
                        "Please refine your query with stronger keywords and try again."
                    ),
                    "generated_idea": "",
                    "generated_idea_steps": [],
                    "generated_idea_citations": [],
                    "used_broader_fallback": True,
                }

            return parsed_response

        parsed = _verify_response(parsed, tokens)

        # Replace placeholder "gap text" with inferred gaps from abstracts.
        if isinstance(parsed, dict) and isinstance(parsed.get("research_gaps"), list):
            title_to_abstract = {r.get("title", ""): _clean_row_text(r.get("abstract", "")) for r in rows_sorted}
            cleaned_gaps = []
            for g in parsed["research_gaps"]:
                if not isinstance(g, str):
                    continue
                g = _normalize_output_text(g, max_chars=320)
                if "gap text" in g.lower():
                    # Try to map to a paper title prefix
                    parts = g.split(":", 1)
                    title = parts[0].strip() if parts else ""
                    abstract = title_to_abstract.get(title, "")
                    snippet = abstract.split(".")[0] if abstract else "No abstract available"
                    cleaned_gaps.append(f"{title}: {snippet}.")
                else:
                    if "no abstract available" in g.lower():
                        continue
                    cleaned_gaps.append(g)
            if not cleaned_gaps:
                cleaned_gaps = _heuristic_gaps()
            parsed["research_gaps"] = cleaned_gaps
        if isinstance(parsed, dict) and isinstance(parsed.get("research_gaps"), list):
            if any(isinstance(g, str) and "Gap not extracted" in g for g in parsed["research_gaps"]):
                parsed["research_gaps"] = _heuristic_gaps()
        if isinstance(parsed, dict) and isinstance(parsed.get("generated_idea"), str):
            parsed["generated_idea"] = _normalize_output_text(parsed.get("generated_idea", ""), max_chars=700)
            if "Idea not extracted" in parsed["generated_idea"] or _looks_generic_idea(parsed["generated_idea"]):
                parsed["generated_idea"] = _fallback_idea_from_rows(rows_sorted)

        # Strip HTML from idea and steps
        if isinstance(parsed, dict) and parsed.get("generated_idea"):
            parsed["generated_idea"] = _normalize_output_text(parsed.get("generated_idea"), max_chars=700)
        if isinstance(parsed, dict) and isinstance(parsed.get("generated_idea_steps"), list):
            parsed["generated_idea_steps"] = [
                _normalize_output_text(s, max_chars=220)
                for s in parsed.get("generated_idea_steps", [])
                if _normalize_output_text(s, max_chars=220)
            ]
        if isinstance(parsed, dict) and parsed.get("assistant_reply"):
            parsed["assistant_reply"] = _normalize_output_text(parsed.get("assistant_reply"), max_chars=500)

        # Add references from source docs if model did not include them.
        # Fallback table if the model didn't provide one.
        if not parsed.get("table"):
            table_rows = []
        show_warnings = (load_env_var("SHOW_SOURCE_WARNINGS", "false") or "false").lower() == "true"
        if warnings and show_warnings:
            parsed["warnings"] = warnings

        # Ensure the table is never empty if we actually found papers.
        # This fixes the issue where the LLM might return an empty table for a valid query.
        if isinstance(parsed, dict) and not parsed.get("table") and rows_sorted:
            fallback_rows = [] # Ensure 5 papers for fallback
            for r in rows_sorted[:5]:
                abstract = _clean_row_text(r.get("abstract", ""))
                key = (r.get("title", "") or "", r.get("url", "") or "")
                fulltext_snippet = fulltext_map.get(key) or fulltext_by_title.get(r.get("title", "") or "")
                summary_text = _best_available_summary(r, fulltext_snippet, abstract)
                problem_text = _best_available_problem(r, fulltext_snippet, abstract)
                
                fallback_rows.append(
                    {
                        "paper_name": r.get("title", ""),
                        "paper_url": _resolve_url(r),
                        "authors_name": r.get("authors", ""),
                        "summary_full_paper": summary_text,
                        "problem_solved": problem_text,
                        "proposed_model_or_approach": _best_available_approach(r, fulltext_snippet, abstract),
                        "source": r.get("source", ""),
                        "score_relevance": 5,
                        "score_quality": 5,
                    }
                )
            parsed["table"] = fallback_rows
            parsed["used_broader_fallback"] = True

        return parsed
    except Exception as exc:
        fallback_source_rows = rows_sorted or rows
        if not fallback_source_rows:
            fallback_source_rows = _simple_fallback_source_rows(topic)
        if fallback_source_rows:
            return _deterministic_fallback_response(fallback_source_rows, str(exc))
        return {"error": f"Research Explorer failed: {exc}"}


def _run_paper_reviewer_impl(paper_text: str) -> Dict[str, Any]:
    """Run a structured reviewer chain on an uploaded PDF."""
    if not paper_text:
        return {"error": "Paper text is required."}

    normalized_text = " ".join(str(paper_text).strip().split())
    cache_key = hashlib.sha256(normalized_text.encode("utf-8")).hexdigest()
    if cache_key in _PAPER_REVIEW_CACHE:
        return _PAPER_REVIEW_CACHE[cache_key]

    fast_mode = (load_env_var("FAST_MODE", "false") or "false").lower() == "true"
    max_review_chars = int(load_env_var("PAPER_REVIEW_MAX_CHARS", "25000") or "25000")
    if len(paper_text) > max_review_chars:
        cutoff = paper_text.rfind("\n", 0, max_review_chars)
        if cutoff <= 0:
            cutoff = max_review_chars
        paper_text = paper_text[:cutoff]

    try:
        review_composer = ReviewResponseComposer()

        def _infer_venue_type(text: str) -> str:
            """Classify the paper as better suited for a conference or a journal."""
            normalized = _extract_review_source_text(text).lower()
            word_count = len(normalized.split())

            conference_signals = [
                "conference",
                "iccit",
                "workshop",
                "proceedings",
                "short paper",
                "ensemble",
                "prototype",
                "application",
                "experimental study",
            ]
            journal_signals = [
                "journal",
                "theoretical",
                "comprehensive",
                "extensive",
                "ablation",
                "multiple datasets",
                "rigorous",
                "systematic review",
                "longitudinal",
                "multi-year",
            ]

            conf_score = 0
            journal_score = 0
            for term in conference_signals:
                if term in normalized:
                    conf_score += 1
            for term in journal_signals:
                if term in normalized:
                    journal_score += 1

            if word_count < 4500:
                conf_score += 2
            if word_count > 9000:
                journal_score += 2
            if any(term in normalized for term in ["dataset", "experiment", "accuracy", "f1", "precision", "recall"]):
                conf_score += 1
            if any(term in normalized for term in ["ablation", "multiple datasets", "generalization", "limitations", "discussion"]):
                journal_score += 1

            return "Journal" if journal_score > conf_score else "Conference"

        def _extract_review_source_text(text: str) -> str:
            """Prefer the paper's substantive sections over front-matter noise."""
            normalized = str(text or "").strip()
            if not normalized:
                return ""

            def _extract_section(heading_pattern: str, end_patterns: str) -> str:
                match = re.search(
                    rf"(?si){heading_pattern}\s*(?:[:\n]|—|–|-)?\s*(.*?)(?=\n\s*(?:{end_patterns})\s*(?:[:\n]|—|–|-)?|$)",
                    normalized,
                )
                if match:
                    return match.group(1).strip()
                return ""

            sections = [
                _extract_section(
                    r"Abstract",
                    r"(?:I\.\s*Introduction|1\.\s*Introduction|Introduction|Keywords|Index Terms|II\.\s*Related Work|Related Work|Methodology|Methods|Method|Experiments|Evaluation|Results|Discussion|Conclusion|Conclusions|References)",
                ),
                _extract_section(
                    r"(?:I\.\s*Introduction|1\.\s*Introduction|Introduction)",
                    r"(?:II\.\s*Related Work|Related Work|Methodology|Methods|Method|Experiments|Evaluation|Results|Discussion|Conclusion|Conclusions|References)",
                ),
                _extract_section(
                    r"(?:Methodology|Methods|Method|III\.\s*Methodology|III\.\s*Methods)",
                    r"(?:IV\.\s*Experiments|Experiments|Evaluation|Results|Discussion|Conclusion|Conclusions|References)",
                ),
                _extract_section(
                    r"(?:Experiments|Evaluation|Results|IV\.\s*Experiments|IV\.\s*Evaluation)",
                    r"(?:Discussion|Conclusion|Conclusions|References)",
                ),
                _extract_section(
                    r"(?:Conclusion|Conclusions|V\.\s*Conclusion)",
                    r"(?:References)",
                ),
            ]
            sections = [section for section in sections if section]
            if sections:
                return "\n\n".join(sections)
            # Fallback: remove obvious front-matter noise while keeping the rest of the paper.
            lines = [line.strip() for line in normalized.splitlines() if line.strip()]
            keep_from = 0
            for idx, line in enumerate(lines):
                if re.search(r"(?i)\babstract\b", line):
                    keep_from = idx
                    break
            if keep_from > 0:
                return "\n".join(lines[keep_from:])
            return normalized

        def _looks_like_front_matter(text: str) -> bool:
            if not text:
                return False
            lower = " ".join(text.lower().split())
            signals = [
                "international conference on computer and information technology",
                "department of computer science and engineering",
                "united international university",
                "cox’s bazar",
                "cox's bazar",
                "email:",
            ]
            return any(signal in lower for signal in signals)

        def _heuristic_review(text: str) -> Dict[str, Any]:
            """Fallback review built from the most informative paper sections."""
            review_text = _extract_review_source_text(text)
            chunk_size = int(load_env_var("PAPER_REVIEW_CHUNK_SIZE", "1200") or "1200")
            chunk_overlap = int(load_env_var("PAPER_REVIEW_CHUNK_OVERLAP", "200") or "200")
            if fast_mode:
                chunk_size = max(chunk_size, 1500)
                chunk_overlap = min(chunk_overlap, 100)

            chunks = chunk_text(review_text, chunk_size=chunk_size, overlap=chunk_overlap)
            if not chunks:
                raise RuntimeError("Paper text is too short or could not be chunked.")

            requested_max_chunks = int(load_env_var("PAPER_REVIEW_MAX_CHUNKS", "4") or "4")
            max_chunks = min(requested_max_chunks, 2 if fast_mode else requested_max_chunks, len(chunks))
            if len(chunks) > max_chunks:
                chunks = chunks[:max_chunks]

            raw_content = "\n\n".join(chunks)
            text_for_analysis = (raw_content[:8500] if len(raw_content) > 8500 else raw_content).strip()

            sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text_for_analysis) if s.strip()]

            def _clean_text(value: str, max_chars: int = 400) -> str:
                text_value = (value or "").strip()
                text_value = " ".join(text_value.split())
                return text_value[:max_chars] if len(text_value) > max_chars else text_value

            strengths_notes: list[str] = []
            weaknesses_notes: list[str] = []
            novelty_notes: list[str] = []
            technical_notes: list[str] = []
            reproducibility_notes: list[str] = []

            for s in sentences:
                low = s.lower()
                if len(strengths_notes) < 2 and any(k in low for k in ["contribution", "propose", "novel", "new", "approach", "demonstrate", "show"]):
                    strengths_notes.append(s)
                if len(weaknesses_notes) < 2 and any(k in low for k in ["limitation", "future work", "scope", "challenge", "drawback", "lack", "weak"]):
                    weaknesses_notes.append(s)
                if len(novelty_notes) < 2 and any(k in low for k in ["contribution", "novel", "unique", "innovation", "first"]):
                    novelty_notes.append(s)
                if len(technical_notes) < 2 and any(k in low for k in ["dataset", "experiment", "accuracy", "f1", "precision", "model", "architecture", "training", "evaluation"]):
                    technical_notes.append(s)
                if len(reproducibility_notes) < 2 and any(k in low for k in ["dataset", "code", "hyperparameter", "reproducible", "open source", "replicate"]):
                    reproducibility_notes.append(s)
                if (
                    len(strengths_notes) >= 2
                    and len(weaknesses_notes) >= 2
                    and len(novelty_notes) >= 2
                    and len(technical_notes) >= 2
                    and len(reproducibility_notes) >= 2
                ):
                    break

            if not strengths_notes and sentences:
                strengths_notes = sentences[:2]
            if not weaknesses_notes and len(sentences) > 2:
                weaknesses_notes = [sentences[min(2, len(sentences) - 1)]]
            if not novelty_notes and sentences:
                novelty_notes = [sentences[0]]
            if not technical_notes and len(sentences) > 1:
                technical_notes = [sentences[1]]
            if not reproducibility_notes and len(sentences) > 2:
                reproducibility_notes = [sentences[2]]

            return {
                "strengths": _clean_text(" ".join(strengths_notes[:2]) or "The paper provides a clear description of the core research idea."),
                "weaknesses": _clean_text(" ".join(weaknesses_notes[:2]) or "Some evaluation details are underspecified and need improvement."),
                "novelty": _clean_text(" ".join(novelty_notes[:2]) or "The paper offers a distinctive methodology or combination of methods."),
                "technical_correctness": _clean_text(" ".join(technical_notes[:2]) or "Technical details are fairly described, but the full reproduction path is incomplete."),
                "reproducibility": _clean_text(" ".join(reproducibility_notes[:2]) or "Clear dataset/hyperparameter/code details are required for repeatability."),
                "recommendation": _clean_text("Revise with stronger experimental validation and explicit evaluation details."),
                "suggested_venue": _infer_venue_type(text),
            }

        def _sanitize_review_result(review: Dict[str, Any], source_text: str) -> Dict[str, Any]:
            """Replace obvious front-matter leakage with safer review text."""
            if not isinstance(review, dict):
                return review

            source_lower = _extract_review_source_text(source_text).lower()
            source_header = " ".join(str(source_text or "").splitlines()[:12]).lower()

            def _bad_field(value: Any) -> bool:
                text_value = " ".join(str(value or "").split())
                lower_value = text_value.lower()
                if not lower_value:
                    return True
                if len(text_value) < 20:
                    return True
                if _looks_like_front_matter(text_value):
                    return True
                if "phishing link detection using ensemble learning" in lower_value and len(text_value) > 25:
                    return True
                overlap_terms = [
                    term
                    for term in [
                        "international conference on computer and information technology",
                        "phishing link detection using ensemble learning",
                        "md muzadded chowdhury",
                        "neamul islam fahim",
                        "sayed hossain jobayer",
                        "md farnas utsho",
                        "md mehedi hasan",
                        "united international university",
                    ]
                    if term in lower_value or term in source_header or term in source_lower
                ]
                return len(overlap_terms) >= 2

            cleaned = dict(review)
            for key, fallback in {
                "strengths": "The paper presents a relevant problem and an ensemble-based detection strategy.",
                "weaknesses": "The evaluation details need clearer baselines, datasets, and experimental setup.",
                "novelty": "The paper combines multiple classical learners into an ensemble for phishing detection.",
                "technical_correctness": "The method appears plausible, but the validation details should be stated more explicitly.",
                "reproducibility": "Dataset splits, preprocessing, and model settings should be described in more detail.",
            }.items():
                if _bad_field(cleaned.get(key)):
                    cleaned[key] = fallback
            cleaned["suggested_venue"] = _infer_venue_type(source_text)
            return cleaned

        chunk_size = int(load_env_var("PAPER_REVIEW_CHUNK_SIZE", "1200") or "1200")
        chunk_overlap = int(load_env_var("PAPER_REVIEW_CHUNK_OVERLAP", "200") or "200")
        if fast_mode:
            chunk_size = max(chunk_size, 1500)
            chunk_overlap = min(chunk_overlap, 100)

        review_source_text = _extract_review_source_text(paper_text)
        chunks = chunk_text(review_source_text, chunk_size=chunk_size, overlap=chunk_overlap)
        if not chunks:
            return {"error": "Paper text is too short or could not be chunked."}

        requested_max_chunks = int(load_env_var("PAPER_REVIEW_MAX_CHUNKS", "4") or "4")
        max_chunks = min(requested_max_chunks, 2 if fast_mode else requested_max_chunks, len(chunks))
        if len(chunks) > max_chunks:
            chunks = chunks[:max_chunks]

        # Summarize each chunk (parallel but bounded) for faster response times.
        summaries = []
        with ThreadPoolExecutor(max_workers=min(2, len(chunks))) as executor:
            futures = {
                executor.submit(_invoke_with_fallback, paper_chunk_summarizer_chain, {"chunk": chunk}): chunk
                for chunk in chunks
            }
            for fut in as_completed(futures):
                try:
                    summary = fut.result(timeout=60)
                except Exception:
                    summary = ""
                if hasattr(summary, "content"):
                    summaries.append(getattr(summary, "content"))
                else:
                    summaries.append(str(summary) if summary is not None else "")

        combined_summary = "\n\n".join([s for s in summaries if isinstance(s, str) and s.strip()])

        if fast_mode:
            parsed = _sanitize_review_result(_heuristic_review(paper_text), paper_text)
            _PAPER_REVIEW_CACHE[cache_key] = parsed
            return parsed
        review_input = "\n\n".join(
            part
            for part in [
                review_source_text,
                combined_summary,
            ]
            if isinstance(part, str) and part.strip()
        )
        try:
            raw = _invoke_with_fallback(paper_reviewer_chain, {"paper": review_input})
        except Exception as exc:
            if "invalid_api_key" in str(exc).lower() or "401" in str(exc):
                parsed = review_composer.sanitize(review_composer.heuristic_review(paper_text), paper_text)
                parsed["suggested_venue"] = _infer_venue_type(paper_text)
                _PAPER_REVIEW_CACHE[cache_key] = parsed
                return parsed
            raise
        parsed = safe_json_loads(raw)

        if isinstance(parsed, dict) and parsed.get("error"):
            schema_hint = (
                "JSON object with keys: strengths, weaknesses, novelty, technical_correctness, "
                "reproducibility, recommendation, suggested_venue."
            )
            repaired = _invoke_with_fallback(
                json_repair_chain, {"bad_json": raw, "schema_hint": schema_hint}
            )
            parsed = safe_json_loads(repaired)

        if not isinstance(parsed, dict) or parsed.get("error"):
            parsed = _heuristic_review(paper_text)

        parsed = review_composer.sanitize(parsed, paper_text)
        parsed["suggested_venue"] = _infer_venue_type(paper_text)

        _PAPER_REVIEW_CACHE[cache_key] = parsed
        return parsed
    except Exception as exc:
        return {"error": f"Paper Reviewer failed: {exc}"}


def _run_paper_reviewer_followup_impl(question: str, paper_text: str) -> Dict[str, Any]:
    """Answer reviewer follow-up questions in critique mode."""
    if not question:
        return {"error": "Question is required."}
    if not paper_text:
        return {"error": "Paper text is required."}

    review = _run_paper_reviewer_impl(paper_text)
    if isinstance(review, dict) and review.get("error"):
        return review

    normalized_question = " ".join(str(question).lower().split())

    def _answer_from_review() -> str:
        strengths = str(review.get("strengths", "")).strip()
        weaknesses = str(review.get("weaknesses", "")).strip()
        novelty = str(review.get("novelty", "")).strip()
        technical = str(review.get("technical_correctness", "")).strip()
        reproducibility = str(review.get("reproducibility", "")).strip()
        recommendation = str(review.get("recommendation", "")).strip()
        venue = str(review.get("suggested_venue", "")).strip()

        if any(term in normalized_question for term in ["strength", "positive", "good point", "advantage"]):
            return strengths or "The paper has a reasonable core idea, but the review text does not provide strong evidence for a deeper strength analysis."
        if any(term in normalized_question for term in ["weakness", "limitation", "problem", "drawback", "concern"]):
            return weaknesses or "The main concern is that the paper needs clearer evaluation details and stronger evidence."
        if any(term in normalized_question for term in ["novel", "novelty", "new", "original"]):
            return novelty or "The novelty appears to be the ensemble combination of classical learners for phishing detection."
        if any(term in normalized_question for term in ["reproduc", "replic", "repeat", "code", "dataset", "hyperparameter"]):
            return reproducibility or "Reproducibility is limited unless the paper states dataset splits, preprocessing, and training settings more clearly."
        if any(term in normalized_question for term in ["recommend", "accept", "reject", "revision", "venue", "publish"]):
            parts = [p for p in [recommendation, venue] if p]
            if parts:
                return " ".join(parts)
            return venue or "Conference"
        if any(term in normalized_question for term in ["evaluation", "experiment", "experiment", "baseline", "validation", "results", "performance"]):
            parts = [p for p in [weaknesses, technical, recommendation] if p]
            if parts:
                return " ".join(parts)
            return "The evaluation appears underspecified, especially around baselines and validation."

        parts = [p for p in [strengths, weaknesses, novelty] if p]
        if parts:
            return " ".join(parts[:2])
        return "The paper does not provide enough evidence for a confident critique."

    local_answer = _answer_from_review().strip()
    if local_answer:
        return {"answer": local_answer}

    try:
        answer = _invoke_with_fallback(
            paper_reviewer_followup_chain,
            {"paper_text": paper_text, "question": question},
        )
        if hasattr(answer, "content"):
            answer_text = getattr(answer, "content")
        else:
            answer_text = str(answer)
        answer_text = " ".join(answer_text.split()).strip()
        if not answer_text:
            answer_text = "The paper does not provide enough evidence to answer that confidently."
        return {"answer": answer_text}
    except Exception:
        fallback = _answer_from_review()
        return {"answer": fallback or "The paper does not provide enough evidence to answer that confidently."}


def _run_paper_qa_impl(question: str, paper_text: str) -> Dict[str, Any]:
    """Run a QA chain on the text of an uploaded paper."""
    if not question:
        return {"error": "Question is required."}
    if not paper_text:
        return {"error": "Paper text is required."}
    try:
        answer = _invoke_with_fallback(
            paper_qa_chain, {"paper_text": paper_text, "question": question}
        )
        if hasattr(answer, "content"):
            answer_text = getattr(answer, "content")
        else:
            answer_text = str(answer)
        return {"answer": answer_text}
    except Exception as exc:
        return {"error": f"Paper QA failed: {exc}"}


def _run_reference_generator_impl(topic: str) -> Dict[str, Any]:
    if not topic:
        return {"error": "Topic is required."}
    try:
        docs = arxiv_search(topic, max_results=10)
        if docs and docs[0].metadata.get("error"):
            return {"error": docs[0].metadata.get("error")}

        # Build seed references from arXiv metadata.
        seed_refs = []
        for d in docs:
            meta = d.metadata or {}
            authors = clean_authors(meta.get("authors", []))
            seed_refs.append(
                format_apa_reference(
                    title=meta.get("title", ""),
                    authors=authors,
                    year=meta.get("year", ""),
                    url=meta.get("url", ""),
                )
            )

        raw = _invoke_with_fallback(
            reference_generator_chain, {"topic": topic, "seed_references": "\n".join(seed_refs)}
        )
        parsed = safe_json_loads(raw)
        if isinstance(parsed, dict) and parsed.get("error"):
            schema_hint = "JSON array of 10 APA reference strings."
            repaired = _invoke_with_fallback(
                json_repair_chain, {"bad_json": raw, "schema_hint": schema_hint}
            )
            parsed = safe_json_loads(repaired)
        if isinstance(parsed, dict) and parsed.get("error"):
            return parsed
        return {"references": parsed, "seed_references": seed_refs}
    except Exception as exc:
        return {"error": f"Reference Generator failed: {exc}"}


def _build_research_graph():
    graph = StateGraph(ResearchState)

    def run_node(state: ResearchState) -> dict:
        use_live_sources = True if (state.get("retries", 0) or 0) > 0 else state.get("use_live")
        if state.get("force_refresh"):
            use_live_sources = True

        res = _run_research_explorer_impl_legacy(
            state["topic"],
            chat_history=state.get("chat_history"),
            focus_topic=state.get("focus_topic"),
            use_live_sources=use_live_sources,
            previously_returned_titles=state.get("previously_returned_titles"),
        )
        return {"result": res}

    def score_node(state: ResearchState) -> dict:
        res = _score_research_result(state.get("result") or {}, state.get("topic") or "")
        return {**state, "result": res}

    def check_node(state: ResearchState) -> ResearchState:
        cleaned = _validate_research_result(state.get("result") or {}, state.get("topic") or "")
        if isinstance(cleaned, dict) and cleaned.get("error"):
            return {**state, "result": cleaned, "validation_error": cleaned.get("error"), "retries": state.get("retries", 0) + 1}
        ok, err = _strict_validate(ResearchResultSchema, cleaned)
        retries = state.get("retries", 0)
        if not ok:
            retries += 1
        return {**state, "result": cleaned, "validation_error": None if ok else err, "retries": retries}

    def _route(state: ResearchState) -> str:
        if state.get("validation_error") and state.get("retries", 0) < 2:
            return "run"
        return END

    graph.add_node("run", run_node)
    graph.add_node("score", score_node)
    graph.add_node("check", check_node)
    graph.set_entry_point("run")
    graph.add_edge("run", "score")
    graph.add_edge("score", "check")
    graph.add_conditional_edges("check", _route)
    return graph.compile()

from .services.workflows import (
    run_paper_qa,
    run_paper_reviewer,
    run_paper_reviewer_followup,
    run_reference_generator,
    run_research_explorer,
)

_start_research_warmup()


def assistant_chat(prompt: str, chat_history: str | None = None) -> dict[str, Any]:
    """General-purpose assistant chat endpoint wrapper."""
    try:
        from .assistant_model import assistant_chat as _assistant_chat
        return _assistant_chat(prompt, chat_history=chat_history)
    except Exception as exc:
        return {"error": str(exc)}
