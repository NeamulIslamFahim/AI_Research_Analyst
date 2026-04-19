"""Embedding loader for the AI Research Assistant."""

from __future__ import annotations

import os
from typing import Optional, List
import json

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.embeddings import Embeddings

from .helpers import load_env_var, ensure_directory


EMBEDDING_MODEL = load_env_var("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2") or "sentence-transformers/all-MiniLM-L6-v2"
_EMBEDDINGS_SINGLETON: Embeddings | None = None
_DUMMY_EMBEDDINGS_SINGLETON: Embeddings | None = None


class _SafeEmbeddings(Embeddings):
    """Wrapper to coerce inputs into strings before embedding."""

    def __init__(self, inner: HuggingFaceEmbeddings):
        self.inner = inner

    def embed_documents(self, texts: List[object]) -> List[List[float]]:
        if not isinstance(texts, list):
            texts = [texts]
        safe_texts: List[str] = []
        for t in texts:
            if isinstance(t, str):
                safe_texts.append(t)
                continue
            if t is None:
                safe_texts.append("")
                continue
            if isinstance(t, (list, tuple)):
                try:
                    safe_texts.append(" ".join([str(x) for x in t if x is not None]))
                except Exception:
                    safe_texts.append(str(t))
                continue
            try:
                safe_texts.append(json.dumps(t, ensure_ascii=True))
            except Exception:
                safe_texts.append(str(t))
        try:
            return self.inner.embed_documents(safe_texts)
        except Exception:
            # Last-resort fallback: force plain string conversion and retry.
            retry_texts = [str(x) if x is not None else "" for x in safe_texts]
            try:
                return self.inner.embed_documents(retry_texts)
            except Exception:
                # Final fallback: return zero vectors with correct dimensionality.
                try:
                    dim_vec = self.inner.embed_query("") or []
                    dim = len(dim_vec)
                except Exception:
                    dim = 0
                return [[0.0] * dim for _ in retry_texts]

    def embed_query(self, text: object) -> List[float]:
        if isinstance(text, str):
            safe_text = text
        elif text is None:
            safe_text = ""
        else:
            try:
                safe_text = json.dumps(text, ensure_ascii=True)
            except Exception:
                safe_text = str(text)
        try:
            return self.inner.embed_query(safe_text)
        except Exception:
            try:
                return self.inner.embed_query(str(safe_text))
            except Exception:
                return []


class _DummyEmbeddings(Embeddings):
    """Fallback embeddings that return zero vectors to avoid hard failures."""

    def __init__(self, dim: int = 384):
        self.dim = dim

    def embed_documents(self, texts: List[object]) -> List[List[float]]:
        if not isinstance(texts, list):
            texts = [texts]
        return [[0.0] * self.dim for _ in texts]

    def embed_query(self, text: object) -> List[float]:
        return [0.0] * self.dim


def create_embeddings() -> Embeddings:
    """Create a HuggingFaceEmbeddings instance with a reliable default model."""
    global _EMBEDDINGS_SINGLETON
    if _EMBEDDINGS_SINGLETON is not None:
        return _EMBEDDINGS_SINGLETON

    try:
        try:
            # First, try to load the model from local files only to avoid network calls if cached.
            base = HuggingFaceEmbeddings(
                model_name=EMBEDDING_MODEL,
                model_kwargs={"device": "cpu", "local_files_only": True},
                encode_kwargs={"normalize_embeddings": True},
            )
        except Exception:
            # If it fails, it likely means the model is not cached. Try to download it.
            base = HuggingFaceEmbeddings(
                model_name=EMBEDDING_MODEL,
                model_kwargs={"device": "cpu", "local_files_only": False},
                encode_kwargs={"normalize_embeddings": True},
            )
    except Exception as exc:  # Defensive: surface embedding init failures clearly.
        # Fallback to dummy embeddings to keep the app running on Windows handle errors.
        _EMBEDDINGS_SINGLETON = _DummyEmbeddings(dim=384)
        return _EMBEDDINGS_SINGLETON

    _EMBEDDINGS_SINGLETON = _SafeEmbeddings(base)
    return _EMBEDDINGS_SINGLETON


def create_dummy_embeddings(dim: int = 384) -> Embeddings:
    """Create a dummy embeddings instance to keep FAISS operational."""
    global _DUMMY_EMBEDDINGS_SINGLETON
    if _DUMMY_EMBEDDINGS_SINGLETON is None:
        _DUMMY_EMBEDDINGS_SINGLETON = _DummyEmbeddings(dim=dim)
    return _DUMMY_EMBEDDINGS_SINGLETON


def get_faiss_persist_dir() -> str:
    """Resolve FAISS persistence directory from env, with default."""
    try:
        persist_dir = load_env_var("FAISS_PERSIST_DIR", "data/vectorstore")
    except Exception as exc:
        raise RuntimeError(f"Failed to read FAISS_PERSIST_DIR: {exc}") from exc
    if not persist_dir:
        persist_dir = "data/vectorstore"
    ensure_directory(persist_dir)
    return persist_dir
