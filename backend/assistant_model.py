"""Local trained assistant model backed by cached PDFs and vector data."""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any

from rank_bm25 import BM25Okapi

from .chains import assistant_answer_chain
from .helpers import ensure_directory, load_env_var
from .pdf_utils import chunk_text, extract_text
from .retriever import load_vector_store
from .storage import list_paper_records


DEFAULT_ASSISTANT_MODEL = "trained-local-corpus"
_MODEL_CACHE: dict[str, Any] | None = None


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
    for item in _pdf_docs() + _vectorstore_docs():
        text = item.get("text", "")
        metadata = item.get("metadata", {}) or {}
        if not text:
            continue
        deduped[_doc_id(text, metadata)] = {"text": text, "metadata": metadata}
    return list(deduped.values())


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
        raise RuntimeError("No cached PDF or vectorstore content found for training.")

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


def get_assistant_status() -> dict[str, Any]:
    """Return current training status for the local assistant model."""
    global _MODEL_CACHE

    if _MODEL_CACHE is None:
        payload = _load_json(_artifact_path("assistant_index.json"))
        if payload:
            _MODEL_CACHE = _load_runtime(payload)
    if _MODEL_CACHE is None:
        return {"status": "not_trained", "model": DEFAULT_ASSISTANT_MODEL}
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
        train_assistant_model(force=True)
    if _MODEL_CACHE is None:
        raise RuntimeError("Assistant model is not available.")
    return _MODEL_CACHE


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
    try:
        store = load_vector_store()
    except Exception:
        store = None
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
    ranked = sorted(combined.values(), key=lambda item: item.get("score", 0.0), reverse=True)
    return ranked[:limit]


def _build_context(hits: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for idx, hit in enumerate(hits, start=1):
        meta = hit.get("metadata", {}) or {}
        parts.append(
            (
                f"[Source {idx}]\n"
                f"Title: {meta.get('title', 'Untitled')}\n"
                f"Source: {meta.get('source', 'unknown')}\n"
                f"URL: {meta.get('url', '') or meta.get('pdf_url', '')}\n"
                f"Chunk: {meta.get('chunk', '')}\n"
                f"Text: {hit.get('text', '')}"
            )
        )
    return "\n\n".join(parts)


def _fallback_answer(prompt: str, hits: list[dict[str, Any]]) -> str:
    if not hits:
        return "I could not find relevant trained corpus content for that prompt."
    lead = hits[0]
    title = (lead.get("metadata", {}) or {}).get("title", "the local corpus")
    snippet = lead.get("text", "")[:900].strip()
    return (
        f"I found relevant material in {title}, but a generative model was not available. "
        f"Closest grounded passage for your prompt '{prompt}':\n\n{snippet}"
    )


def assistant_chat(prompt: str, chat_history: str | None = None) -> dict[str, Any]:
    """Answer the user's prompt using the trained local corpus as the default model."""
    if not prompt or not prompt.strip():
        raise RuntimeError("Prompt is required.")

    hits = _hybrid_retrieve(prompt.strip())
    context = _build_context(hits)
    answer = ""

    assistant_only = (load_env_var("ASSISTANT_MODEL_ONLY", "false") or "false").lower() == "true"

    try:
        if assistant_only:
            raise RuntimeError("Assistant model only mode is enabled.")

        from .main import _invoke_with_fallback

        raw = _invoke_with_fallback(
            assistant_answer_chain,
            {
                "prompt": prompt.strip(),
                "chat_history": (chat_history or "").strip() or "No prior chat history.",
                "context": context or "No relevant context found.",
            },
        )
        answer = getattr(raw, "content", str(raw)).strip()
    except Exception:
        answer = _fallback_answer(prompt.strip(), hits)

    sources = []
    for hit in hits:
        metadata = hit.get("metadata", {}) or {}
        sources.append(
            {
                "title": metadata.get("title", "Untitled"),
                "source": metadata.get("source", "unknown"),
                "url": metadata.get("url", "") or metadata.get("pdf_url", ""),
                "chunk": metadata.get("chunk", ""),
                "snippet": hit.get("text", "")[:280].strip(),
            }
        )

    status = get_assistant_status()
    return {
        "model": status.get("model", DEFAULT_ASSISTANT_MODEL),
        "assistant_only": assistant_only,
        "trained_at": status.get("trained_at"),
        "answer": answer,
        "sources": sources,
    }
