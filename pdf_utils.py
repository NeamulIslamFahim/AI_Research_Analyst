"""PDF loading and text chunking utilities."""

from __future__ import annotations

from typing import List

import logging
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def extract_text(pdf_file: str) -> str:
    """Extract text from a PDF file path, page by page.

    Args:
        pdf_file: Path to the PDF file on disk.

    Returns:
        Concatenated text from all pages.
    """
    try:
        # Silence noisy PDF warnings from pypdf
        logging.getLogger("pypdf").setLevel(logging.ERROR)
        loader = PyPDFLoader(pdf_file)
        pages = loader.load()
        return "\n\n".join([p.page_content for p in pages])
    except Exception as exc:
        raise RuntimeError(f"Failed to extract PDF text: {exc}") from exc


def chunk_text(text: str, chunk_size: int = 1200, overlap: int = 200) -> List[str]:
    """Chunk text into overlapping segments for retrieval."""
    try:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            separators=["\n\n", "\n", " ", ""],
        )
        return splitter.split_text(text)
    except Exception as exc:
        raise RuntimeError(f"Failed to chunk text: {exc}") from exc
