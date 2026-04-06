"""Local trained assistant model backed by cached PDFs and vector data."""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
from pathlib import Path
from typing import Any

from rank_bm25 import BM25Okapi

from .helpers import append_chat_log_entry, ensure_directory, load_env_var
from .pdf_utils import chunk_text, extract_text
from .retriever import load_vector_store
from .storage import list_paper_records


DEFAULT_ASSISTANT_MODEL = "trained-local-corpus"
_MODEL_CACHE: dict[str, Any] | None = None
_VECTOR_STORE_CACHE: Any | None = None
_TRAINING_LOCK = threading.Lock()
_TRAINING_THREAD: threading.Thread | None = None
_TRAINING_REQUESTS = 0
_ASSISTANT_WARMUP_STARTED = False


def _artifact_dir() -> str:
    path = load_env_var("ASSISTANT_MODEL_DIR", "data/trained_assistant") or "data/trained_assistant"
    ensure_directory(path)
    return path


def _artifact_path(name: str) -> str:
    return os.path.join(_artifact_dir(), name)


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", (text or "").lower())


def _query_tokens(text: str) -> list[str]:
    stopwords = {
        "the", "and", "for", "with", "from", "that", "this", "into", "have", "has",
        "had", "are", "was", "were", "what", "when", "where", "which", "who", "whom",
        "will", "would", "could", "should", "can", "about", "main", "than", "then",
        "them", "they", "their", "there", "your", "user", "prompt", "please", "give",
        "tell", "show", "explain", "describe", "summarize", "summary", "research",
    }
    tokens = [token for token in _tokenize(text) if len(token) > 2 and token not in stopwords]
    return list(dict.fromkeys(tokens))


def _doc_id(text: str, metadata: dict[str, Any]) -> str:
    payload = json.dumps(
        {
            "text": _normalize_whitespace(text)[:2000],
            "title": metadata.get("title", ""),
            "source": metadata.get("source", ""),
            "url": metadata.get("url", ""),
            "chunk": metadata.get("chunk", ""),
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _vectorstore_docs() -> list[dict[str, Any]]:
    docs: list[dict[str, Any]] = []
    try:
        store = load_vector_store()
    except Exception:
        store = None
    if store is None:
        return docs
    raw_docs = getattr(getattr(store, "docstore", None), "_dict", {})
    for value in raw_docs.values():
        text = _normalize_whitespace(getattr(value, "page_content", ""))
        metadata = dict(getattr(value, "metadata", {}) or {})
        if not text:
            continue
        docs.append(
            {
                "text": text,
                "metadata": {
                    "title": metadata.get("title", ""),
                    "authors": metadata.get("authors", ""),
                    "source": metadata.get("source", "vectorstore"),
                    "url": metadata.get("url", ""),
                    "pdf_url": metadata.get("pdf_url", ""),
                    "chunk": metadata.get("chunk", ""),
                },
                }
            )
    return docs


def _chat_history_docs() -> list[dict[str, Any]]:
    """Load saved chat workspaces as lightweight retrievable documents."""
    include_logs = (load_env_var("ASSISTANT_INCLUDE_CHAT_LOGS", "true") or "true").lower() == "true"
    if not include_logs:
        return []

    chat_dir = Path(load_env_var("CHAT_LOG_DIR", "data/chat_logs") or "data/chat_logs")
    if not chat_dir.exists():
        return []

    docs: list[dict[str, Any]] = []
    for path in sorted(chat_dir.glob("*.json")):
        try:
            with open(path, "r", encoding="utf-8") as handle:
                record = json.load(handle)
        except Exception:
            continue

        messages = record.get("messages", [])
        if not isinstance(messages, list):
            continue

        lines: list[str] = []
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            role = "User" if msg.get("role") == "user" else "Assistant"
            content = msg.get("effective_query") or msg.get("display_text") or msg.get("content") or ""
            if isinstance(content, dict):
                content = content.get("answer") or content.get("assistant_reply") or ""
            content_text = _normalize_whitespace(str(content))
            if content_text:
                lines.append(f"{role}: {content_text}")

        if not lines:
            continue

        docs.append(
            {
                "text": "\n".join(lines),
                "metadata": {
                    "title": record.get("title", "") or path.stem,
                    "authors": "",
                    "source": "chat_history",
                    "url": str(path),
                    "pdf_url": "",
                    "chunk": 0,
                    "mode": record.get("mode", ""),
                },
            }
        )

    return docs


def _pdf_docs() -> list[dict[str, Any]]:
    docs: list[dict[str, Any]] = []
    for row in list_paper_records():
        file_path = row.get("file_path") or ""
        if not file_path or not os.path.exists(file_path):
            continue
        try:
            text = extract_text(file_path)
        except Exception:
            continue
        for idx, chunk in enumerate(chunk_text(text)):
            clean = _normalize_whitespace(chunk)
            if not clean:
                continue
            docs.append(
                {
                    "text": clean,
                    "metadata": {
                        "title": row.get("title", "") or Path(file_path).stem,
                        "authors": row.get("authors", ""),
                        "source": row.get("source", "pdf"),
                        "url": row.get("url", ""),
                        "pdf_url": row.get("pdf_url", ""),
                        "chunk": idx,
                    },
                }
            )
    return docs


def _build_corpus() -> list[dict[str, Any]]:
    deduped: dict[str, dict[str, Any]] = {}
    for item in _chat_history_docs() + _pdf_docs() + _vectorstore_docs():
        text = item.get("text", "")
        metadata = item.get("metadata", {}) or {}
        if not text:
            continue
        deduped[_doc_id(text, metadata)] = {"text": text, "metadata": metadata}
    return list(deduped.values())


def _corpus_availability() -> dict[str, int]:
    pdf_docs = _pdf_docs()
    vector_docs = _vectorstore_docs()
    return {
        "pdf_chunk_count": len(pdf_docs),
        "vector_chunk_count": len(vector_docs),
        "total_chunk_count": len(pdf_docs) + len(vector_docs),
    }


def _save_json(path: str, payload: dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=True, indent=2)


def _load_json(path: str) -> dict[str, Any] | None:
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_runtime(payload: dict[str, Any]) -> dict[str, Any]:
    chunks = payload.get("chunks", [])
    tokenized = [_tokenize(chunk.get("text", "")) for chunk in chunks]
    bm25 = BM25Okapi(tokenized) if tokenized else None
    return {"meta": payload.get("meta", {}), "chunks": chunks, "tokenized": tokenized, "bm25": bm25}


def train_assistant_model(force: bool = False) -> dict[str, Any]:
    """Train a lightweight local retrieval model from PDFs and vectorstore data."""
    global _MODEL_CACHE

    if not force and _MODEL_CACHE is not None:
        meta = dict(_MODEL_CACHE.get("meta", {}) or {})
        meta["status"] = "ready"
        return meta

    corpus = _build_corpus()
    if not corpus:
        stats = _corpus_availability()
        return {
            "status": "not_trained",
            "model": DEFAULT_ASSISTANT_MODEL,
            "message": "No cached PDF or vectorstore content found for training.",
            **stats,
        }

    source_counts: dict[str, int] = {}
    doc_titles: set[str] = set()
    chunks: list[dict[str, Any]] = []
    for item in corpus:
        metadata = dict(item.get("metadata", {}) or {})
        source = metadata.get("source", "unknown") or "unknown"
        title = metadata.get("title", "") or "Untitled"
        source_counts[source] = source_counts.get(source, 0) + 1
        doc_titles.add(title)
        chunks.append({"text": item.get("text", ""), "metadata": metadata})

    payload = {
        "meta": {
            "status": "ready",
            "model": DEFAULT_ASSISTANT_MODEL,
            "trained_at": int(time.time()),
            "chunk_count": len(chunks),
            "document_count": len(doc_titles),
            "source_counts": source_counts,
        },
        "chunks": chunks,
    }
    _save_json(_artifact_path("assistant_index.json"), payload)
    _save_json(
        _artifact_path("assistant_config.json"),
        {"default_model": DEFAULT_ASSISTANT_MODEL, "trained_at": payload["meta"]["trained_at"]},
    )
    _MODEL_CACHE = _load_runtime(payload)
    return payload["meta"]


def schedule_assistant_retrain() -> None:
    """Queue a background retrain, coalescing repeated requests."""
    global _TRAINING_THREAD, _TRAINING_REQUESTS

    with _TRAINING_LOCK:
        _TRAINING_REQUESTS += 1
        if _TRAINING_THREAD is not None and _TRAINING_THREAD.is_alive():
            return

        thread = threading.Thread(target=_assistant_retrain_worker, daemon=True)
        _TRAINING_THREAD = thread
        thread.start()


def _assistant_retrain_worker() -> None:
    """Process queued retrain requests until the queue is drained."""
    global _TRAINING_THREAD

    last_processed = 0
    while True:
        with _TRAINING_LOCK:
            request_count = _TRAINING_REQUESTS
        if request_count <= last_processed:
            break
        last_processed = request_count
        try:
            train_assistant_model(force=True)
        except Exception:
            # Retraining is a best-effort background task.
            pass

    with _TRAINING_LOCK:
        _TRAINING_THREAD = None


def get_assistant_status() -> dict[str, Any]:
    """Return current training status for the local assistant model."""
    global _MODEL_CACHE

    if _MODEL_CACHE is None:
        payload = _load_json(_artifact_path("assistant_index.json"))
        if payload:
            _MODEL_CACHE = _load_runtime(payload)
    if _MODEL_CACHE is None:
        return {
            "status": "not_trained",
            "model": DEFAULT_ASSISTANT_MODEL,
            "message": "Assistant model has not been trained yet.",
            **_corpus_availability(),
        }
    meta = dict(_MODEL_CACHE.get("meta", {}) or {})
    meta["status"] = meta.get("status", "ready")
    return meta


def _ensure_model_loaded() -> dict[str, Any]:
    global _MODEL_CACHE
    if _MODEL_CACHE is None:
        payload = _load_json(_artifact_path("assistant_index.json"))
        if payload:
            _MODEL_CACHE = _load_runtime(payload)
    if _MODEL_CACHE is None:
        meta = train_assistant_model(force=True)
        if meta.get("status") != "ready":
            raise RuntimeError(meta.get("message", "Assistant model is not available."))
    if _MODEL_CACHE is None:
        raise RuntimeError("Assistant model is not available.")
    return _MODEL_CACHE


def _get_vector_store_cached() -> Any:
    global _VECTOR_STORE_CACHE
    if _VECTOR_STORE_CACHE is None:
        try:
            _VECTOR_STORE_CACHE = load_vector_store()
        except Exception:
            _VECTOR_STORE_CACHE = None
    return _VECTOR_STORE_CACHE


def _warm_assistant_runtime() -> None:
    """Preload the assistant cache in the background."""
    try:
        train_assistant_model(force=False)
    except Exception:
        pass
    try:
        _get_vector_store_cached()
    except Exception:
        pass


def _start_assistant_warmup() -> None:
    global _ASSISTANT_WARMUP_STARTED
    if _ASSISTANT_WARMUP_STARTED:
        return
    _ASSISTANT_WARMUP_STARTED = True
    if (load_env_var("WARM_ASSISTANT_CACHE", "true") or "true").lower() != "true":
        return
    threading.Thread(target=_warm_assistant_runtime, daemon=True).start()


def _bm25_hits(model: dict[str, Any], prompt: str, limit: int) -> list[dict[str, Any]]:
    bm25 = model.get("bm25")
    chunks = model.get("chunks", [])
    if bm25 is None or not chunks:
        return []
    query_tokens = _tokenize(prompt)
    if not query_tokens:
        return []
    scores = bm25.get_scores(query_tokens)
    ranked = sorted(enumerate(scores), key=lambda item: item[1], reverse=True)[:limit]
    hits: list[dict[str, Any]] = []
    for idx, score in ranked:
        chunk = chunks[idx]
        hits.append(
            {
                "id": _doc_id(chunk.get("text", ""), chunk.get("metadata", {}) or {}),
                "score": float(score),
                "text": chunk.get("text", ""),
                "metadata": chunk.get("metadata", {}) or {},
                "retriever": "bm25",
            }
        )
    return hits


def _vector_hits(prompt: str, limit: int) -> list[dict[str, Any]]:
    hits: list[dict[str, Any]] = []
    store = _get_vector_store_cached()
    if store is None:
        return hits
    try:
        docs = store.similarity_search(prompt, k=limit)
    except Exception:
        return hits
    score = float(limit)
    for doc in docs:
        metadata = dict(getattr(doc, "metadata", {}) or {})
        text = _normalize_whitespace(getattr(doc, "page_content", ""))
        if not text:
            continue
        hits.append(
            {
                "id": _doc_id(text, metadata),
                "score": score,
                "text": text,
                "metadata": metadata,
                "retriever": "vector",
            }
        )
        score -= 1.0
    return hits


def _hybrid_retrieve(prompt: str, limit: int = 6) -> list[dict[str, Any]]:
    model = _ensure_model_loaded()
    query_terms = _query_tokens(prompt)
    combined: dict[str, dict[str, Any]] = {}
    for hit in _bm25_hits(model, prompt, limit * 2):
        existing = combined.get(hit["id"])
        if existing is None or hit["score"] > existing["score"]:
            combined[hit["id"]] = hit
    for hit in _vector_hits(prompt, limit * 2):
        existing = combined.get(hit["id"])
        if existing is None:
            combined[hit["id"]] = hit
        else:
            existing["score"] = float(existing.get("score", 0.0)) + float(hit["score"])
            existing["retriever"] = "hybrid"
    reranked: list[dict[str, Any]] = []
    for item in combined.values():
        metadata = item.get("metadata", {}) or {}
        title = _normalize_whitespace(metadata.get("title", ""))
        text = _normalize_whitespace(item.get("text", ""))
        title_tokens = set(_query_tokens(title))
        text_tokens = set(_query_tokens(text[:2000]))
        overlap = len(set(query_terms).intersection(title_tokens | text_tokens))
        title_overlap = len(set(query_terms).intersection(title_tokens))

        # Strongly favor chunks that match the user's real terms, especially in titles.
        adjusted_score = float(item.get("score", 0.0))
        adjusted_score += overlap * 6.0
        adjusted_score += title_overlap * 10.0

        if query_terms and overlap == 0:
            adjusted_score -= 50.0

        reranked.append(
            {
                **item,
                "score": adjusted_score,
                "overlap": overlap,
                "title_overlap": title_overlap,
            }
        )

    ranked = sorted(
        [
            item
            for item in reranked
            if not query_terms or item.get("overlap", 0) > 0 or item.get("title_overlap", 0) > 0
        ],
        key=lambda item: item.get("score", 0.0),
        reverse=True,
    )

    if not ranked:
        ranked = sorted(reranked, key=lambda item: item.get("score", 0.0), reverse=True)

    diversified: list[dict[str, Any]] = []
    per_title_counts: dict[str, int] = {}
    max_per_title = 2

    for item in ranked:
        metadata = item.get("metadata", {}) or {}
        title = _normalize_whitespace(metadata.get("title", "") or "Untitled")
        count = per_title_counts.get(title, 0)
        if count >= max_per_title:
            continue
        per_title_counts[title] = count + 1
        diversified.append(item)
        if len(diversified) >= limit:
            break

    if len(diversified) < limit:
        for item in ranked:
            if item in diversified:
                continue
            diversified.append(item)
            if len(diversified) >= limit:
                break

    return diversified[:limit]


def _diverse_hits_for_answer(hits: list[dict[str, Any]], limit: int = 4) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    seen_titles: set[str] = set()

    for hit in hits:
        metadata = hit.get("metadata", {}) or {}
        title = _normalize_whitespace(metadata.get("title", "") or "Untitled")
        if title in seen_titles:
            continue
        seen_titles.add(title)
        selected.append(hit)
        if len(selected) >= limit:
            return selected

    for hit in hits:
        if hit in selected:
            continue
        selected.append(hit)
        if len(selected) >= limit:
            break

    return selected


def _best_sentences_from_hit(hit: dict[str, Any], prompt_tokens: set[str], limit: int = 2) -> list[str]:
    text = _normalize_whitespace(hit.get("text", ""))
    title = (hit.get("metadata", {}) or {}).get("title", "Untitled")
    if not text:
        return []

    scored: list[tuple[int, str]] = []
    seen: set[str] = set()
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        clean = _normalize_whitespace(sentence)
        if len(clean) < 40:
            continue
        alpha_count = sum(1 for ch in clean if ch.isalpha())
        word_count = len(clean.split())
        if alpha_count < 25 or word_count < 8:
            continue
        if clean.lower() == title.lower():
            continue
        if clean.count("|") >= 1 and word_count < 14:
            continue
        if clean.isupper():
            continue
        overlap = len(prompt_tokens.intersection(_query_tokens(clean)))
        if prompt_tokens and overlap <= 0:
            continue
        key = clean[:220]
        if key in seen:
            continue
        seen.add(key)
        scored.append((overlap, f"{clean} (Source: {title})"))

    scored.sort(key=lambda item: item[0], reverse=True)
    return [sentence for _, sentence in scored[:limit]]


def _extractive_grounded_answer(prompt: str, hits: list[dict[str, Any]]) -> str:
    if not hits:
        return "I could not find relevant trained corpus content for that prompt."

    prompt_tokens = set(_query_tokens(prompt))
    passages: list[str] = []
    used: set[str] = set()

    for hit in _diverse_hits_for_answer(hits, limit=4):
        for sentence in _best_sentences_from_hit(hit, prompt_tokens, limit=2):
            key = sentence[:220]
            if key in used:
                continue
            used.add(key)
            passages.append(sentence)
            if len(passages) >= 5:
                break
        if len(passages) >= 5:
            break

    if not passages:
        lead = hits[0]
        title = (lead.get("metadata", {}) or {}).get("title", "the local corpus")
        snippet = lead.get("text", "")[:900].strip()
        return (
            f"Based on the trained assistant knowledge base, the closest relevant material comes from {title}.\n\n"
            f"{snippet}"
        )

    return (
        "Based on the trained assistant knowledge base, here is the most relevant grounded answer:\n\n"
        + "\n\n".join(passages)
    )


def _fallback_answer(prompt: str, hits: list[dict[str, Any]]) -> str:
    return _extractive_grounded_answer(prompt, hits)


def _has_relevant_local_answer(hits: list[dict[str, Any]], prompt: str) -> bool:
    if not hits:
        return False

    query_terms = set(_query_tokens(prompt))
    if not query_terms:
        return bool(hits)

    strong_hits = 0
    for hit in hits[:4]:
        overlap = int(hit.get("overlap", 0) or 0)
        title_overlap = int(hit.get("title_overlap", 0) or 0)
        text = _normalize_whitespace(hit.get("text", ""))
        text_tokens = set(_query_tokens(text[:1200]))
        extra_overlap = len(query_terms.intersection(text_tokens))
        if title_overlap >= 1 or overlap >= 2 or extra_overlap >= 2:
            strong_hits += 1

    return strong_hits >= 1


def _generic_follow_up(prompt: str) -> bool:
    normalized = _normalize_whitespace(prompt).lower()
    generic_prompts = {
        "more",
        "more papers",
        "show more",
        "give more",
        "more results",
        "next",
        "another",
        "others",
        "more please",
    }
    return normalized in generic_prompts


def _previous_source_titles(chat_history: str | None) -> set[str]:
    if not chat_history:
        return set()
    titles: set[str] = set()
    for line in str(chat_history).splitlines():
        normalized = line.strip()
        lower = normalized.lower()
        if lower.startswith("sources:") or lower.startswith("papers:"):
            raw = normalized.split(":", 1)[1] if ":" in normalized else ""
            for piece in raw.split("|"):
                clean = _normalize_whitespace(piece).lower()
                if clean:
                    titles.add(clean)
    return titles


def _source_query_variants(prompt: str) -> list[str]:
    base = _normalize_whitespace(prompt)
    if not base:
        return []

    variants = [base]
    tokens = _query_tokens(base)
    if tokens:
        variants.append(tokens[0])
        variants.append(f"{tokens[0]} networking")
        variants.append(f"{tokens[0]} ai")
    if len(tokens) > 1:
        variants.append(" ".join(tokens[:2]))
    variants.extend(
        [
            "networking",
            "network optimization",
            "artificial intelligence",
            "ai networking",
            "ai applications",
        ]
    )
    seen: set[str] = set()
    deduped: list[str] = []
    for query in variants:
        normalized = _normalize_whitespace(query).lower()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(query)
    return deduped


def _topic_from_history(chat_history: str | None) -> str | None:
    if not chat_history:
        return None
    for line in reversed(str(chat_history).splitlines()):
        normalized = line.strip()
        if normalized.startswith("User:"):
            candidate = _normalize_whitespace(normalized.split(":", 1)[1])
            if candidate and not _generic_follow_up(candidate):
                return candidate
    return None


def _background_incremental_learning(topic: str) -> None:
    try:
        from .main import download_papers_for_topic

        download_papers_for_topic(topic)
        train_assistant_model(force=True)
    except Exception:
        # Keep user-facing answer fast and resilient even if background learning fails.
        pass


def _start_incremental_learning(topic: str) -> None:
    threading.Thread(
        target=_background_incremental_learning,
        args=(topic,),
        daemon=True,
    ).start()


def _record_chat_interaction(
    prompt: str,
    response: str,
    chat_history: str | None,
    answer_source: str,
    extra: dict[str, Any] | None = None,
) -> None:
    try:
        entry: dict[str, Any] = {
            "timestamp": int(time.time()),
            "prompt": prompt,
            "response": response,
            "chat_history": chat_history or "",
            "answer_source": answer_source,
        }
        if extra:
            entry["extra"] = extra
        append_chat_log_entry(entry)
    except Exception:
        pass


def get_assistant_sources(prompt: str, chat_history: str | None = None, limit: int = 5) -> list[dict[str, Any]]:
    """Return paper-like sources from the trained local corpus for fallback use."""
    if not prompt or not prompt.strip():
        return []

    clean_prompt = prompt.strip()
    effective_topic = clean_prompt
    if _generic_follow_up(clean_prompt):
        effective_topic = _topic_from_history(chat_history) or clean_prompt

    excluded_titles = _previous_source_titles(chat_history)
    sources: list[dict[str, Any]] = []
    seen_source_keys: set[tuple[str, str]] = set()
    for query in _source_query_variants(effective_topic):
        if len(sources) >= limit:
            break
        try:
            hits = _hybrid_retrieve(query, limit=max(limit * 2, 8))
        except Exception:
            continue
        for hit in _diverse_hits_for_answer(hits, limit=max(limit * 2, 8)):
            metadata = hit.get("metadata", {}) or {}
            title = _normalize_whitespace(metadata.get("title", "") or "Untitled")
            if title.lower() in excluded_titles:
                continue
            url = metadata.get("url", "") or metadata.get("pdf_url", "")
            if not url or str(url).strip().lower().endswith(".json"):
                continue
            source_key = (title, str(url))
            if source_key in seen_source_keys:
                continue
            seen_source_keys.add(source_key)
            sources.append(
                {
                    "title": title,
                    "source": metadata.get("source", "unknown"),
                    "url": url,
                    "pdf_url": metadata.get("pdf_url", "") or url,
                    "authors": metadata.get("authors", ""),
                    "snippet": hit.get("text", "")[:280].strip(),
                    "chunk": metadata.get("chunk", ""),
                }
            )
            if len(sources) >= limit:
                break
    return sources


def assistant_chat(prompt: str, chat_history: str | None = None) -> dict[str, Any]:
    """Answer the user's prompt using the trained local corpus as the default model."""
    if not prompt or not prompt.strip():
        raise RuntimeError("Prompt is required.")

    clean_prompt = prompt.strip()
    effective_topic = clean_prompt
    if _generic_follow_up(clean_prompt):
        effective_topic = _topic_from_history(chat_history) or clean_prompt
    excluded_titles = _previous_source_titles(chat_history)

    try:
        hits = _hybrid_retrieve(effective_topic)
    except Exception as exc:
        status = get_assistant_status()
        error_answer = (
            "The assistant knowledge base is not trained yet. "
            "Please add papers or vector data first, then train the assistant."
        )
        response_payload = {
            "model": status.get("model", DEFAULT_ASSISTANT_MODEL),
            "assistant_only": (load_env_var("ASSISTANT_MODEL_ONLY", "false") or "false").lower() == "true",
            "trained_at": status.get("trained_at"),
            "answer": error_answer,
            "sources": [],
            "status": status.get("status", "not_trained"),
            "message": str(exc),
        }
        _record_chat_interaction(
            prompt=clean_prompt,
            response=error_answer,
            chat_history=chat_history,
            answer_source="not_trained",
            extra={"exception": str(exc)},
        )
        return response_payload
    wants_more_papers = _generic_follow_up(clean_prompt)
    local_relevant = (not wants_more_papers) and _has_relevant_local_answer(hits, effective_topic)
    answer = ""

    assistant_only = (load_env_var("ASSISTANT_MODEL_ONLY", "false") or "false").lower() == "true"

    if local_relevant:
        # Keep the general assistant fast and local. Deeper paper work belongs in Research Explorer.
        answer = _fallback_answer(clean_prompt, hits)

        sources = get_assistant_sources(clean_prompt, chat_history=chat_history, limit=5)

        status = get_assistant_status()
        response_payload = {
            "model": status.get("model", DEFAULT_ASSISTANT_MODEL),
            "assistant_only": assistant_only,
            "trained_at": status.get("trained_at"),
            "answer": answer,
            "sources": sources,
            "answer_source": "vectordb",
            "incremental_learning_started": False,
        }
        _record_chat_interaction(
            prompt=clean_prompt,
            response=answer,
            chat_history=chat_history,
            answer_source="vectordb",
            extra={"sources": sources},
        )
        return response_payload

    status = get_assistant_status()
    response_payload = {
        "model": status.get("model", DEFAULT_ASSISTANT_MODEL),
        "assistant_only": assistant_only,
        "trained_at": status.get("trained_at"),
        "answer": answer,
        "sources": [],
        "answer_source": "vectordb_fallback",
        "incremental_learning_started": False,
    }
    _record_chat_interaction(
        prompt=clean_prompt,
        response=answer,
        chat_history=chat_history,
        answer_source="vectordb_fallback",
    )
    return response_payload


_start_assistant_warmup()
