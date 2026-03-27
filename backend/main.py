
"""Backend orchestration for the AI Research Assistant."""

from __future__ import annotations

import os
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple, TypedDict
from pydantic import BaseModel, Field, ValidationError

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langgraph.graph import StateGraph, END
import json
import requests

from .chains import (
    paper_reviewer_chain,
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
    clean_authors,
    extract_datasets,
    extract_models,
    extract_proposed_approach,
    format_apa_reference,
    load_env_var,
    safe_get,
    safe_json_loads,
    truncate_text,
    authors_to_str,
    strip_html,
)
from .storage import get_cached_pdf_path, save_pdf_bytes, upsert_paper_record
from .pdf_utils import chunk_text, extract_text
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


DEFAULT_LLM_MODEL_PRIMARY = "llama-3.3-70b-versatile"
DEFAULT_LLM_MODEL_SECONDARY = "llama-3.1-8b-instant"
DEFAULT_GROQ_REASONING_EFFORT = "medium"
DEFAULT_OSS_MODEL_ID = "gpt-oss-120b"

_CACHED_VECTOR_STORE: Any | None = None
_CACHED_EMBEDDINGS: Any | None = None
_CACHED_DUMMY_EMBEDDINGS: Any | None = None


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


def _set_vector_store(vs: Any) -> None:
    global _CACHED_VECTOR_STORE
    _CACHED_VECTOR_STORE = vs


def init_llm(model_id: str) -> ChatGroq:
    """Initialize Groq chat LLM with environment token."""
    try:
        load_dotenv()
        token = load_env_var("GROQ_API_KEY")
        if not token:
            raise ValueError("GROQ_API_KEY is missing. Add it to your .env file.")

        reasoning_effort = load_env_var("GROQ_REASONING_EFFORT", DEFAULT_GROQ_REASONING_EFFORT)
        max_tokens = int(load_env_var("GROQ_MAX_TOKENS", "1024") or "1024")

        kwargs = {
            "api_key": token,
            "model": model_id,
            "temperature": 0.2,
            "max_tokens": max_tokens,
        }
        if reasoning_effort:
            kwargs["reasoning_effort"] = reasoning_effort
        return ChatGroq(
            **kwargs,
        )
    except Exception as exc:
        raise RuntimeError(f"Failed to initialize Groq LLM: {exc}") from exc


def init_oss_llm() -> ChatOpenAI:
    """Initialize gpt-oss via an OpenAI-compatible local endpoint."""
    try:
        load_dotenv()
        base_url = load_env_var("OSS_BASE_URL")
        if not base_url:
            raise ValueError("OSS_BASE_URL is missing. Set it to your local inference server URL.")
        model_id = load_env_var("OSS_MODEL_ID", DEFAULT_OSS_MODEL_ID) or DEFAULT_OSS_MODEL_ID
        api_key = load_env_var("OSS_API_KEY", "local-oss-key") or "local-oss-key"
        return ChatOpenAI(
            api_key=api_key,
            model=model_id,
            base_url=base_url,
            temperature=0.2,
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


def _invoke_with_fallback(chain_builder, invoke_payload: dict) -> Any:
    """Invoke a chain with gpt-oss if enabled; otherwise use Groq with fallback."""
    use_oss = (load_env_var("USE_GPT_OSS", "false") or "false").lower() == "true"
    if use_oss:
        llm = init_oss_llm()
        chain = chain_builder(llm)
        return chain.invoke(invoke_payload)

    primary, secondary = get_model_ids()
    try:
        llm = init_llm(primary)
        chain = chain_builder(llm)
        return chain.invoke(invoke_payload)
    except Exception as exc:
        if _is_rate_limit_error(exc):
            llm = init_llm(secondary)
            chain = chain_builder(llm)
            return chain.invoke(invoke_payload)
        raise


def _ensure_vector_store_with_docs(docs: List[Document]) -> Any:
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


def _download_arxiv_fulltext(docs: List[Document]) -> List[Document]:
    """Optionally download arXiv PDFs and return chunked Documents."""
    fulltext_docs: List[Document] = []
    max_downloads = int(load_env_var("MAX_PDF_DOWNLOADS", "15") or "15")
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


def _download_external_fulltext(rows: List[dict]) -> List[Document]:
    """Download PDFs from external sources when a direct PDF URL is provided."""
    fulltext_docs: List[Document] = []
    max_downloads = int(load_env_var("MAX_PDF_DOWNLOADS", "15") or "15")
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
    sem_rows, _ = semantic_scholar_search(topic, max_results=10)
    oa_rows, _ = semantic_scholar_open_access_search(topic, max_results=8)
    scholar_rows, _ = serpapi_scholar_search(topic, max_results=10)
    rg_rows, _ = serpapi_researchgate_search(topic, max_results=10)
    web_rows, _ = serpapi_web_search(topic, max_results=6)
    sd_rows, _ = serpapi_sciencedirect_search(topic, max_results=6)
    oa_rows2, _ = openalex_search(topic, max_results=6)
    core_rows, _ = core_search(topic, max_results=6)
    doaj_rows, _ = doaj_search(topic, max_results=6)
    epmc_rows, _ = europe_pmc_search(topic, max_results=6)

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
) -> Dict[str, Any]:
    """Run arXiv retrieval + RAG generation for the given topic."""
    if not topic:
        return {"error": "Topic is required."}
    try:
        load_dotenv()
        local_only = (load_env_var("LOCAL_ONLY", "false") or "false").lower() == "true"
        fast_mode = (load_env_var("FAST_MODE", "false") or "false").lower() == "true"
        max_primary = int(load_env_var("FAST_MAX_PRIMARY", "5") or "5")
        max_secondary = int(load_env_var("FAST_MAX_SECONDARY", "4") or "4")
        warnings: List[str] = []
        if use_live_sources is None:
            use_live_sources = not local_only

        if not use_live_sources:
            vector_store = _get_vector_store()
            if vector_store is None:
                use_live_sources = True
            else:
                docs = vector_store.similarity_search(topic, k=8)
                if not docs:
                    use_live_sources = True
                else:
                    # Basic relevance check: ensure at least one keyword matches title/abstract.
                    topic_tokens = [t for t in topic.lower().replace("-", " ").split() if len(t) > 2]
                    def _match(doc: Document) -> bool:
                        meta = doc.metadata or {}
                        hay = " ".join(
                            [
                                str(meta.get("title", "")),
                                str(meta.get("abstract", "")),
                                str(doc.page_content or ""),
                            ]
                        ).lower()
                        return any(tok in hay for tok in topic_tokens)
                    if topic_tokens and not any(_match(d) for d in docs):
                        use_live_sources = True
                    else:
                        rows = []
                        for d in [d for d in docs if _match(d) or not topic_tokens]:
                            meta = d.metadata or {}
                            rows.append(
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
                        all_docs = [d for d in docs if _match(d) or not topic_tokens]
                        fulltext_docs = []
                        if not rows:
                            use_live_sources = True

        def _topic_tokens(text: str) -> List[str]:
            stop = {"the","and","for","with","from","that","this","into","in","on","of","to","a","an","is","are"}
            tokens = [t for t in text.lower().replace("-", " ").split() if len(t) > 2 and t not in stop]
            return list(dict.fromkeys(tokens))

        def _row_matches(row: dict, tokens: List[str]) -> bool:
            if not tokens:
                return True
            hay = " ".join([
                str(row.get("title","")),
                str(row.get("abstract","")),
                str(row.get("authors","")),
            ]).lower()
            return any(tok in hay for tok in tokens)

        tokens = _topic_tokens(topic)

        if use_live_sources:
            # Keep retrieval fast but still >10 papers total across sources.
            docs = arxiv_search(topic, max_results=(max_primary if fast_mode else 8))
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

            def _run_task(fn, max_results: int) -> Tuple[List[dict], str]:
                try:
                    return fn(topic, max_results=max_results)
                except Exception as exc:
                    return [], str(exc)

            tasks = [
                ("sem", semantic_scholar_search, max_secondary if fast_mode else 6, True),
                ("oa", semantic_scholar_open_access_search, max_secondary if fast_mode else 6, False),
                ("scholar", serpapi_scholar_search, max_secondary if fast_mode else 6, False),
                ("rg", serpapi_researchgate_search, max_secondary if fast_mode else 6, False),
                ("web", serpapi_web_search, max_secondary if fast_mode else 4, False),
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
                    rows_out, warn_out = fut.result()
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
            if tokens and not rows:
                return {"error": "No relevant papers found for the topic. Please refine your query."}
            all_docs = docs + rows_to_docs(all_rows)
            fulltext_docs: List[Document] = []
            if not fast_mode and (load_env_var("DOWNLOAD_ARXIV_PDFS", "true") or "true").lower() == "true":
                fulltext_docs.extend(_download_arxiv_fulltext(docs))
            if not fast_mode and (load_env_var("DOWNLOAD_EXTERNAL_PDFS", "true") or "true").lower() == "true":
                fulltext_docs.extend(
                    _download_external_fulltext(
                        all_rows
                    )
                )

            all_docs.extend(fulltext_docs)
        # Build a lookup for full-text snippets by (title, url) and by title.
        fulltext_map: Dict[tuple[str, str], str] = {}
        fulltext_by_title: Dict[str, str] = {}
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
        # Prefer fulltext and newer papers
        rows_sorted = sorted(
            filtered_rows,
            key=lambda r: ( _is_fulltext(r), int(str(r.get("year", "0"))[:4] or 0) ),
            reverse=True,
        )
        # Prefer rows with usable abstract/fulltext to avoid "No abstract available".
        def _has_content(r: dict) -> bool:
            if _is_fulltext(r):
                return True
            abstract = (r.get("abstract", "") or "").strip()
            return bool(abstract)
        content_rows = [r for r in rows_sorted if _has_content(r)]
        if len(content_rows) >= 3:
            rows_sorted = content_rows
        fulltext_only = (load_env_var("FULLTEXT_ONLY", "false") or "false").lower() == "true"
        if fulltext_only:
            min_fulltext = int(load_env_var("FULLTEXT_MIN", "10") or "10")
            rows_sorted = [r for r in rows_sorted if _is_fulltext(r)]
            if len(rows_sorted) < min_fulltext:
                return {
                    "error": f"Not enough full-text PDFs (need at least {min_fulltext}). "
                             "Try another topic or increase sources."
                }

        # Prepare paper rows for the model to fill in.
        # Trim to keep prompt reasonable but ensure >10 papers.
        def _resolve_url(r: dict) -> str:
            doi = r.get("doi", "")
            url = r.get("url", "") or ""
            if url:
                return url
            pdf_url = r.get("pdf_url", "") or ""
            if pdf_url:
                return pdf_url
            if doi:
                doi_str = str(doi).strip()
                if _looks_like_arxiv_id(doi_str):
                    doi_str = ""
                if "doi.org/" in doi_str:
                    doi_str = doi_str.split("doi.org/", 1)[1]
                doi_str = doi_str.replace("https://", "").replace("http://", "").strip()
                doi_str = doi_str.replace("doi:", "").strip()
                doi_str = doi_str.replace(" ", "")
                if doi_str:
                    return f"https://doi.org/{doi_str}"
            title = r.get("title", "") or ""
            if title:
                return "https://scholar.google.com/scholar?q=" + requests.utils.quote(title)
            return ""

        papers_payload = []
        for r in rows_sorted:
            paper_url = _resolve_url(r)
            abstract = (r.get("abstract", "") or "").strip()
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
        if len(papers_payload) > 12:
            papers_payload = papers_payload[:12]
        if not papers_payload:
            # Fallback to using titles even if abstracts are missing.
            fallback_rows = rows_sorted[:12] if rows_sorted else rows[:12]
            for r in fallback_rows:
                papers_payload.append(
                    {
                        "paper_name": r.get("title", ""),
                        "paper_url": _resolve_url(r),
                        "authors_name": r.get("authors", ""),
                        "abstract": (r.get("abstract", "") or "").strip(),
                        "source": r.get("source", ""),
                        "fulltext_available": _is_fulltext(r),
                    }
                )

        def _heuristic_gaps() -> List[str]:
            gaps_list: List[str] = []
            for r in rows_sorted[: min(12, len(rows_sorted))]:
                title = r.get("title", "") or "Paper"
                abstract = (r.get("abstract", "") or "").strip()
                snippet = abstract.split(".")[0] if abstract else "No abstract available"
                gaps_list.append(f"{title}: {snippet}.")
            return gaps_list

        def _heuristic_idea(gaps_list: List[str]) -> str:
            if not gaps_list:
                return (
                    "Synthesize the missing aspects across papers into a unified approach and evaluate on shared benchmarks."
                )
            return (
                "Unify the identified gaps into a single research direction by designing a method that addresses the missing "
                "assumptions, data coverage, and evaluation gaps across the selected papers."
            )

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
                    abstract = r.get("abstract", "") or ""
                    datasets = extract_datasets(abstract)
                    models = extract_models(abstract)
                    proposed = extract_proposed_approach(abstract)
                key = (r.get("title", "") or "", r.get("url", "") or "")
                fulltext_snippet = fulltext_map.get(key) or fulltext_by_title.get(r.get("title", "") or "")
                summary_text = truncate_text(fulltext_snippet, max_chars=1200) if fulltext_snippet else abstract
                problem_text = ""
                if fulltext_snippet:
                    problem_text = fulltext_snippet.split(".")[0].strip()
                if not problem_text and abstract:
                    problem_text = abstract.split(".")[0].strip()
                proposed_full = extract_proposed_approach(fulltext_snippet) if fulltext_snippet else ""
                fallback_rows.append(
                    {
                        "paper_name": r.get("title", ""),
                        "paper_url": _resolve_url(r),
                        "authors_name": r.get("authors", ""),
                        "summary_full_paper": summary_text or "Not specified in paper",
                        "problem_solved": problem_text or "Not specified in paper",
                        "proposed_model_or_approach": proposed_full or proposed or "Not specified in paper",
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
                        "Research summary: The retrieved papers define the problem scope, dominant methods, and current "
                        "limitations. Below is the structured evidence table, followed by paper-level gaps and a unified "
                        "research direction."
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
            title_to_abstract = {r.get("title", ""): (r.get("abstract", "") or "") for r in rows_sorted}
            deduped = []
            seen = set()
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
                            row[field] = strip_html(row.get(field, ""))
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
                    row["summary_full_paper"] = abstract or "Not specified in paper"
                # Replace any "full text not provided" phrasing with abstract-based summary
                if isinstance(row, dict):
                    summary = (row.get("summary_full_paper", "") or "").lower()
                    if "full text not provided" in summary or "supplied context" in summary:
                        title = row.get("paper_name", "")
                        abstract = title_to_abstract.get(title, "")
                        row["summary_full_paper"] = abstract or "Not specified in paper"
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
                            row["summary_full_paper"] = truncate_text(fulltext_snippet, max_chars=1200)
                        if not row.get("problem_solved") or row.get("problem_solved") in (
                            "Not specified in paper",
                            "Not specified in abstract",
                        ):
                            row["problem_solved"] = fulltext_snippet.split(".")[0].strip() or row.get(
                                "problem_solved", ""
                            )
                        if not row.get("proposed_model_or_approach") or row.get(
                            "proposed_model_or_approach"
                        ) in ("Not specified in paper", "Not specified in abstract"):
                            inferred = extract_proposed_approach(fulltext_snippet)
                            row["proposed_model_or_approach"] = inferred or row.get("proposed_model_or_approach", "")
                    # Ensure proposed_model_or_approach is not just the first abstract line.
                    proposed = (row.get("proposed_model_or_approach", "") or "").strip()
                    summary = (row.get("summary_full_paper", "") or "").strip()
                    problem = (row.get("problem_solved", "") or "").strip()
                    if proposed and (proposed == summary or proposed == problem or proposed in summary):
                        abstract = title_to_abstract.get(title, "")
                        inferred = extract_proposed_approach(fulltext_snippet or abstract)
                        row["proposed_model_or_approach"] = inferred or "Not specified in paper"
                # Fix proposed_model_or_approach if it duplicates the paper title
                if isinstance(row, dict):
                    title = (row.get("paper_name", "") or "").strip()
                    proposed = (row.get("proposed_model_or_approach", "") or "").strip()
                    if title and proposed and (proposed == title or proposed in title or title in proposed):
                        abstract = title_to_abstract.get(title, "")
                        inferred = extract_proposed_approach(abstract)
                        row["proposed_model_or_approach"] = inferred or "Not specified in paper"
                    # Ensure problem_solved is populated
                    if isinstance(row, dict) and not row.get("problem_solved"):
                        abstract = title_to_abstract.get(row.get("paper_name", ""), "")
                        snippet = abstract.split(".")[0] if abstract else ""
                        row["problem_solved"] = snippet or "Not specified in paper"
                    # Normalize bare DOI URLs
                    paper_url = (row.get("paper_url", "") or "").strip()
                    if paper_url.startswith("10.") and "/" in paper_url:
                        row["paper_url"] = f"https://doi.org/{paper_url}"
                if isinstance(row, dict):
                    key = (
                        (row.get("paper_name", "") or "").strip().lower(),
                        (row.get("paper_url", "") or "").strip().lower(),
                    )
                    if key in seen:
                        continue
                    seen.add(key)
                    row["paper_url"] = _fix_paper_url(row.get("paper_url", ""))
                    # If DOI looks invalid (no slash), fall back to source search link.
                    paper_url = row.get("paper_url", "") or ""
                    title = row.get("paper_name", "") or ""
                    if "doi.org/" in paper_url:
                        suffix = paper_url.split("doi.org/", 1)[1]
                        if "/" not in suffix and title:
                            row["paper_url"] = "https://scholar.google.com/scholar?q=" + requests.utils.quote(title)
                    # Ensure scores exist
                    if row.get("score_relevance") is None:
                        row["score_relevance"] = 0
                    if row.get("score_quality") is None:
                        row["score_quality"] = 0
                deduped.append(row)
            parsed["table"] = deduped
            # If model returned too few rows, fall back to deterministic table.
            if len(parsed["table"]) < min(5, len(rows_sorted)):
                fallback_rows = []
                for r in rows_sorted[: min(12, len(rows_sorted))]:
                    key = (r.get("title", "") or "", r.get("url", "") or "")
                    fulltext_snippet = fulltext_map.get(key) or fulltext_by_title.get(r.get("title", "") or "")
                    abstract = (r.get("abstract", "") or "").strip()
                    summary_text = truncate_text(fulltext_snippet, max_chars=1200) if fulltext_snippet else abstract
                    problem_text = ""
                    if fulltext_snippet:
                        problem_text = fulltext_snippet.split(".")[0].strip()
                    if not problem_text and abstract:
                        problem_text = abstract.split(".")[0].strip()
                    proposed_full = extract_proposed_approach(fulltext_snippet) if fulltext_snippet else ""
                    fallback_rows.append(
                        {
                            "paper_name": r.get("title", ""),
                            "paper_url": _resolve_url(r),
                            "authors_name": r.get("authors", ""),
                            "summary_full_paper": summary_text or "Not specified in paper",
                            "problem_solved": problem_text or "Not specified in paper",
                            "proposed_model_or_approach": proposed_full or extract_proposed_approach(abstract) or "Not specified in paper",
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
            parsed["generated_idea"] = _heuristic_idea(parsed.get("research_gaps") or [])
        if isinstance(parsed, dict) and not parsed.get("assistant_reply"):
            parsed["assistant_reply"] = (
                "Research summary: The evidence below synthesizes the retrieved papers, then details paper-level gaps and "
                "a consolidated research direction."
            )
        if isinstance(parsed, dict) and not parsed.get("generated_idea_steps"):
            parsed["generated_idea_steps"] = [
                "Define a unified problem statement covering all gaps.",
                "Collect or curate datasets that address the missing aspects.",
                "Implement a baseline and a new method targeting the gaps.",
                "Evaluate on standard metrics plus gap-specific diagnostics.",
                "Run ablations to isolate each component's impact.",
                "Release code, data splits, and evaluation scripts for reproducibility.",
            ]
        # Final fallback if gaps/idea are still missing.
        if isinstance(parsed, dict) and not parsed.get("research_gaps"):
            parsed["research_gaps"] = _heuristic_gaps()

        # Replace placeholder "gap text" with inferred gaps from abstracts.
        if isinstance(parsed, dict) and isinstance(parsed.get("research_gaps"), list):
            title_to_abstract = {r.get("title", ""): (r.get("abstract", "") or "") for r in rows_sorted}
            cleaned_gaps = []
            for g in parsed["research_gaps"]:
                if not isinstance(g, str):
                    continue
                g = strip_html(g)
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
            parsed["research_gaps"] = cleaned_gaps
        if isinstance(parsed, dict) and isinstance(parsed.get("research_gaps"), list):
            if any(isinstance(g, str) and "Gap not extracted" in g for g in parsed["research_gaps"]):
                parsed["research_gaps"] = _heuristic_gaps()
        if isinstance(parsed, dict) and isinstance(parsed.get("generated_idea"), str):
            if "Idea not extracted" in parsed["generated_idea"]:
                parsed["generated_idea"] = _heuristic_idea(parsed.get("research_gaps") or [])

        if isinstance(parsed, dict) and not parsed.get("generated_idea"):
            parsed["generated_idea"] = (
                "Combine the identified gaps into a unified approach that improves coverage and evaluation rigor."
            )
        if isinstance(parsed, dict) and (not parsed.get("generated_idea_steps") or not parsed.get("generated_idea_steps")):
            parsed["generated_idea_steps"] = [
                "Define a unified problem statement covering all gaps.",
                "Collect or curate datasets that address the missing aspects.",
                "Implement a baseline and a new method targeting the gaps.",
                "Evaluate on standard metrics plus gap-specific diagnostics.",
                "Run ablations to isolate each component's impact.",
                "Release code, data splits, and evaluation scripts for reproducibility.",
            ]
        # Strip HTML from idea and steps
        if isinstance(parsed, dict) and parsed.get("generated_idea"):
            parsed["generated_idea"] = strip_html(parsed.get("generated_idea"))
        if isinstance(parsed, dict) and isinstance(parsed.get("generated_idea_steps"), list):
            parsed["generated_idea_steps"] = [strip_html(s) for s in parsed.get("generated_idea_steps", [])]

        # Add references from source docs if model did not include them.
        # Fallback table if the model didn't provide one.
        if not parsed.get("table"):
            table_rows = []
            for r in rows:
                doi = r.get("doi", "")
                doi_url = f"https://doi.org/{doi}" if doi else ""
                if _looks_like_arxiv_id(str(doi or "")):
                    doi_url = ""
                paper_url = doi_url or r.get("url", "")
                table_rows.append(
                    {
                        "paper_name": r.get("title", ""),
                        "paper_url": _fix_paper_url(paper_url),
                        "authors_name": r.get("authors", ""),
                        "summary_full_paper": r.get("abstract", ""),
                        "datasets_used": [],
                        "models_used": [],
                        "proposed_model_or_approach": "",
                        "source": r.get("source", ""),
                    }
                )
            parsed["table"] = table_rows
        show_warnings = (load_env_var("SHOW_SOURCE_WARNINGS", "false") or "false").lower() == "true"
        if warnings and show_warnings:
            parsed["warnings"] = warnings

        return parsed
    except Exception as exc:
        return {"error": f"Research Explorer failed: {exc}"}


def _run_paper_reviewer_impl(paper_text: str) -> Dict[str, Any]:
    """Run a structured reviewer chain on an uploaded PDF."""
    if not paper_text:
        return {"error": "Paper text is required."}
    try:
        chunks = chunk_text(paper_text)
        
        # Summarize each chunk
        summaries = []
        for chunk in chunks:
            summary = _invoke_with_fallback(paper_chunk_summarizer_chain, {"chunk": chunk})
            if hasattr(summary, "content"):
                summaries.append(getattr(summary, "content"))
            else:
                summaries.append(str(summary))
            
        combined_summary = "\n\n".join([s for s in summaries if isinstance(s, str)])
        
        raw = _invoke_with_fallback(paper_reviewer_chain, {"paper": combined_summary})
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
        return parsed
    except Exception as exc:
        return {"error": f"Paper Reviewer failed: {exc}"}


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
    """Generate 10 APA references for a topic using real arXiv metadata."""
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
from langchain_community.retrievers import BM25Retriever


class ResearchState(TypedDict):
    topic: str
    chat_history: Optional[str]
    focus_topic: Optional[str]
    use_live: Optional[bool]
    result: Optional[Dict[str, Any]]
    retries: int
    validation_error: Optional[str]


class ReviewState(TypedDict):
    paper_text: str
    result: Optional[Dict[str, Any]]
    retries: int
    validation_error: Optional[str]


class QAState(TypedDict):
    question: str
    paper_text: str
    result: Optional[Dict[str, Any]]
    retries: int
    validation_error: Optional[str]


class ReferenceState(TypedDict):
    topic: str
    result: Optional[Dict[str, Any]]
    retries: int
    validation_error: Optional[str]


class ResearchRowSchema(BaseModel):
    paper_name: str
    paper_url: str
    authors_name: str
    summary_full_paper: str
    problem_solved: str
    proposed_model_or_approach: str
    source: str
    score_relevance: int = Field(ge=0, le=10)
    score_quality: int = Field(ge=0, le=10)


class ResearchResultSchema(BaseModel):
    table: List[ResearchRowSchema]
    research_gaps: List[str]
    assistant_reply: str
    generated_idea: str
    generated_idea_steps: List[str]
    generated_idea_citations: List[str] = []


class ReviewResultSchema(BaseModel):
    strengths: str
    weaknesses: str
    novelty: str
    technical_correctness: str
    reproducibility: str
    recommendation: str
    suggested_venue: str


class QAResultSchema(BaseModel):
    answer: str


class ReferenceResultSchema(BaseModel):
    references: List[str]


def _validate_research_result(result: Dict[str, Any], topic: str = "") -> Dict[str, Any]:
    if not isinstance(result, dict):
        return {"error": "Invalid research result."}
    if result.get("error"):
        return result
    # Enforce topic relevance: all rows must match topic tokens.
    tokens = [t for t in (topic or "").lower().replace("-", " ").split() if len(t) > 2]
    if tokens and isinstance(result.get("table"), list):
        filtered = []
        for row in result.get("table", []):
            if not isinstance(row, dict):
                continue
            hay = f"{row.get('paper_name','')} {row.get('summary_full_paper','')}".lower()
            if any(tok in hay for tok in tokens):
                filtered.append(row)
        result["table"] = filtered
        if not filtered:
            return {"error": "No relevant papers found for the topic."}
    result.setdefault("table", [])
    result.setdefault("research_gaps", [])
    result.setdefault("assistant_reply", "Research summary prepared.")
    result.setdefault("generated_idea", "Not provided.")
    result.setdefault("generated_idea_steps", [])
    return result


def _score_research_result(result: Dict[str, Any], topic: str) -> Dict[str, Any]:
    if not isinstance(result, dict):
        return result
    table = result.get("table")
    if not isinstance(table, list):
        return result
    topic_words = [w for w in (topic or "").lower().split() if len(w) > 2]
    for row in table:
        if not isinstance(row, dict):
            continue
        text = f"{row.get('paper_name','')} {row.get('summary_full_paper','')}".lower()
        overlap = sum(1 for w in topic_words if w in text)
        max_overlap = max(len(topic_words), 1)
        relevance = int(min(10, round((overlap / max_overlap) * 10)))
        quality = 0
        if row.get("paper_url"):
            quality += 3
        if row.get("authors_name"):
            quality += 2
        if row.get("summary_full_paper"):
            quality += 3
        if row.get("problem_solved"):
            quality += 1
        if row.get("proposed_model_or_approach"):
            quality += 1
        quality = min(10, quality)
        row["score_relevance"] = relevance
        row["score_quality"] = quality
    return result


def _strict_validate(schema, result: Dict[str, Any]) -> tuple[bool, Optional[str]]:
    try:
        schema.model_validate(result)
        return True, None
    except ValidationError as exc:
        return False, str(exc)


def _normalize_url(url: str) -> str:
    if not url:
        return ""
    trimmed = str(url).strip()
    trimmed = trimmed.replace("https://doi.org/https://doi.org/", "https://doi.org/")
    if trimmed.startswith("http://") or trimmed.startswith("https://"):
        return trimmed
    if trimmed.startswith("doi.org/"):
        return f"https://{trimmed}"
    if trimmed.startswith("doi:"):
        doi = trimmed.replace("doi:", "").strip()
        return f"https://doi.org/{doi}"
    if trimmed.startswith("10."):
        return f"https://doi.org/{trimmed}"
    if trimmed.startswith("arxiv.org/"):
        return f"https://{trimmed}"
    return f"https://{trimmed}"


def _fix_paper_url(url: str) -> str:
    """Fix common invalid paper URLs (e.g., arXiv IDs wrongly formatted as DOIs)."""
    if not url:
        return ""
    trimmed = _normalize_url(url)
    trimmed = trimmed.replace("https://arxiv.org/abs/https://arxiv.org/abs/", "https://arxiv.org/abs/")
    trimmed = trimmed.replace("http://arxiv.org/abs/http://arxiv.org/abs/", "http://arxiv.org/abs/")
    if "doi.org/" in trimmed:
        suffix = trimmed.split("doi.org/", 1)[1]
        if _looks_like_arxiv_id(suffix):
            return f"https://arxiv.org/abs/{suffix}"
        # If DOI is actually a legacy arXiv id (contains slash with category), map to arXiv.
        if "/" in suffix and any(suffix.startswith(prefix) for prefix in ["hep-", "astro-", "cs.", "math.", "physics.", "stat."]):
            return f"https://arxiv.org/abs/{suffix}"
    # If it's a bare arXiv ID, map to arXiv abs page.
    if _looks_like_arxiv_id(trimmed):
        return f"https://arxiv.org/abs/{trimmed}"
    return trimmed


def _looks_like_arxiv_id(value: str) -> bool:
    v = (value or "").strip()
    if not v:
        return False
    if v.startswith("http"):
        v = v.split("/")[-1]
    # arXiv IDs like 2207.08146 or 1809.08274v1
    if v.count(".") == 1 and v.replace("v", "").replace(".", "").isdigit():
        return True
    # arXiv legacy IDs like hep-ex/0412026 or astro-ph/0611001
    if "/" in v and any(v.startswith(prefix) for prefix in ["hep-", "astro-", "cs.", "math.", "physics.", "stat."]):
        return True
    return False


def _validate_review_result(result: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(result, dict):
        return {"error": "Invalid review result."}
    if result.get("error"):
        return result
    for key in [
        "strengths",
        "weaknesses",
        "novelty",
        "technical_correctness",
        "reproducibility",
        "recommendation",
        "suggested_venue",
    ]:
        result.setdefault(key, "Not provided.")
    return result


def _validate_reference_result(result: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(result, dict):
        return {"error": "Invalid references result."}
    if result.get("error"):
        return result
    refs = result.get("references", [])
    if not isinstance(refs, list):
        result["references"] = []
    return result


def _validate_qa_result(result: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(result, dict):
        return {"error": "Invalid QA result."}
    if result.get("error"):
        return result
    result.setdefault("answer", "No answer found.")
    return result


def _build_research_graph():
    graph = StateGraph(ResearchState)

    def run_node(state: ResearchState) -> ResearchState:
        res = _run_research_explorer_impl(
            state["topic"],
            chat_history=state.get("chat_history"),
            focus_topic=state.get("focus_topic"),
            use_live_sources=True if (state.get("retries", 0) or 0) > 0 else state.get("use_live"),
        )
        return {**state, "result": res}

    def score_node(state: ResearchState) -> ResearchState:
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


def _build_review_graph():
    graph = StateGraph(ReviewState)

    def run_node(state: ReviewState) -> ReviewState:
        res = _run_paper_reviewer_impl(state["paper_text"])
        return {**state, "result": res}

    def check_node(state: ReviewState) -> ReviewState:
        cleaned = _validate_review_result(state.get("result") or {})
        ok, err = _strict_validate(ReviewResultSchema, cleaned)
        retries = state.get("retries", 0)
        if not ok:
            retries += 1
        return {**state, "result": cleaned, "validation_error": None if ok else err, "retries": retries}

    def _route(state: ReviewState) -> str:
        if state.get("validation_error") and state.get("retries", 0) < 2:
            return "run"
        return END

    graph.add_node("run", run_node)
    graph.add_node("check", check_node)
    graph.set_entry_point("run")
    graph.add_edge("run", "check")
    graph.add_conditional_edges("check", _route)
    return graph.compile()


def _build_qa_graph():
    graph = StateGraph(QAState)

    def run_node(state: QAState) -> QAState:
        res = _run_paper_qa_impl(state["question"], state["paper_text"])
        return {**state, "result": res}

    def check_node(state: QAState) -> QAState:
        cleaned = _validate_qa_result(state.get("result") or {})
        ok, err = _strict_validate(QAResultSchema, cleaned)
        retries = state.get("retries", 0)
        if not ok:
            retries += 1
        return {**state, "result": cleaned, "validation_error": None if ok else err, "retries": retries}

    def _route(state: QAState) -> str:
        if state.get("validation_error") and state.get("retries", 0) < 2:
            return "run"
        return END

    graph.add_node("run", run_node)
    graph.add_node("check", check_node)
    graph.set_entry_point("run")
    graph.add_edge("run", "check")
    graph.add_conditional_edges("check", _route)
    return graph.compile()


def _build_reference_graph():
    graph = StateGraph(ReferenceState)

    def run_node(state: ReferenceState) -> ReferenceState:
        res = _run_reference_generator_impl(state["topic"])
        return {**state, "result": res}

    def check_node(state: ReferenceState) -> ReferenceState:
        cleaned = _validate_reference_result(state.get("result") or {})
        ok, err = _strict_validate(ReferenceResultSchema, cleaned)
        retries = state.get("retries", 0)
        if not ok:
            retries += 1
        return {**state, "result": cleaned, "validation_error": None if ok else err, "retries": retries}

    def _route(state: ReferenceState) -> str:
        if state.get("validation_error") and state.get("retries", 0) < 2:
            return "run"
        return END

    graph.add_node("run", run_node)
    graph.add_node("check", check_node)
    graph.set_entry_point("run")
    graph.add_edge("run", "check")
    graph.add_conditional_edges("check", _route)
    return graph.compile()


_RESEARCH_GRAPH = _build_research_graph()
_REVIEW_GRAPH = _build_review_graph()
_QA_GRAPH = _build_qa_graph()
_REFERENCE_GRAPH = _build_reference_graph()


def run_research_explorer(
    topic: str,
    chat_history: str | None = None,
    focus_topic: str | None = None,
    use_live: bool | None = None,
) -> Dict[str, Any]:
    state: ResearchState = {
        "topic": topic,
        "chat_history": chat_history,
        "focus_topic": focus_topic,
        "use_live": use_live,
        "result": None,
        "retries": 0,
        "validation_error": None,
    }
    out = _RESEARCH_GRAPH.invoke(state)
    return out.get("result") or {}


def run_paper_reviewer(paper_text: str) -> Dict[str, Any]:
    state: ReviewState = {"paper_text": paper_text, "result": None, "retries": 0, "validation_error": None}
    out = _REVIEW_GRAPH.invoke(state)
    return out.get("result") or {}


def run_paper_qa(question: str, paper_text: str) -> Dict[str, Any]:
    state: QAState = {
        "question": question,
        "paper_text": paper_text,
        "result": None,
        "retries": 0,
        "validation_error": None,
    }
    out = _QA_GRAPH.invoke(state)
    return out.get("result") or {}


def run_reference_generator(topic: str) -> Dict[str, Any]:
    state: ReferenceState = {"topic": topic, "result": None, "retries": 0, "validation_error": None}
    out = _REFERENCE_GRAPH.invoke(state)
    return out.get("result") or {}
