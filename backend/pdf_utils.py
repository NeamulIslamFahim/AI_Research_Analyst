"""PDF loading and text chunking utilities."""

from __future__ import annotations

import logging
from typing import List

from pypdf import PdfReader


def extract_text(pdf_file: str) -> str:
    """Extract text from a PDF file path, page by page."""
    try:
        logging.getLogger("pypdf").setLevel(logging.ERROR)
        reader = PdfReader(pdf_file)
        pages: list[str] = []
        for page in reader.pages:
            page_text = (page.extract_text() or "").strip()
            if page_text:
                pages.append(page_text)
        return "\n\n".join(pages)
    except Exception as exc:
        raise RuntimeError(f"Failed to extract PDF text: {exc}") from exc


def _best_breakpoint(text: str, start: int, end: int) -> int:
    for marker in ("\n\n", "\n", " "):
        idx = text.rfind(marker, start, end)
        if idx > start:
            return idx + len(marker)
    return end


def chunk_text(text: str, chunk_size: int = 1200, overlap: int = 200) -> List[str]:
    """Chunk text into overlapping segments for retrieval."""
    try:
        normalized = str(text or "").strip()
        if not normalized:
            return []
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than 0")

        safe_overlap = max(0, min(overlap, chunk_size - 1))
        chunks: list[str] = []
        start = 0
        text_length = len(normalized)

        while start < text_length:
            end = min(text_length, start + chunk_size)
            if end < text_length:
                candidate = _best_breakpoint(normalized, start, end)
                if candidate > start + max(chunk_size // 3, 1):
                    end = candidate

            chunk = normalized[start:end].strip()
            if chunk:
                chunks.append(chunk)
            if end >= text_length:
                break

            next_start = max(end - safe_overlap, start + 1)
            if next_start <= start:
                next_start = end
            start = next_start

        return chunks
    except Exception as exc:
        raise RuntimeError(f"Failed to chunk text: {exc}") from exc
