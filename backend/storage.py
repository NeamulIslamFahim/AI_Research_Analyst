"""Local paper storage (SQLite + PDF cache)."""

from __future__ import annotations

import hashlib
import os
import sqlite3
from datetime import datetime
from typing import Optional

from .helpers import ensure_directory, load_env_var


def _get_paths() -> tuple[str, str]:
    base_dir = load_env_var("PAPER_DB_DIR", "paper_db") or "paper_db"
    pdf_dir = os.path.join(base_dir, "pdfs")
    ensure_directory(base_dir)
    ensure_directory(pdf_dir)
    db_path = os.path.join(base_dir, "papers.sqlite")
    return db_path, pdf_dir


def init_db() -> None:
    db_path, _ = _get_paths()
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS papers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT,
                authors TEXT,
                url TEXT,
                pdf_url TEXT UNIQUE,
                source TEXT,
                file_path TEXT,
                added_at TEXT
            )
            """
        )
        conn.commit()
    finally:
        conn.close()


def get_cached_pdf_path(pdf_url: str) -> Optional[str]:
    if not pdf_url:
        return None
    init_db()
    db_path, _ = _get_paths()
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.execute("SELECT file_path FROM papers WHERE pdf_url = ?", (pdf_url,))
        row = cur.fetchone()
        if row and row[0] and os.path.exists(row[0]):
            return row[0]
        return None
    finally:
        conn.close()


def _hash_url(url: str) -> str:
    return hashlib.sha256(url.encode("utf-8")).hexdigest()


def save_pdf_bytes(pdf_url: str, content: bytes) -> str:
    init_db()
    _, pdf_dir = _get_paths()
    filename = _hash_url(pdf_url) + ".pdf"
    file_path = os.path.join(pdf_dir, filename)
    if not os.path.exists(file_path):
        with open(file_path, "wb") as f:
            f.write(content)
    return file_path


def upsert_paper_record(
    title: str,
    authors: str,
    url: str,
    pdf_url: str,
    source: str,
    file_path: Optional[str] = None,
) -> None:
    init_db()
    db_path, _ = _get_paths()
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            """
            INSERT INTO papers (title, authors, url, pdf_url, source, file_path, added_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(pdf_url) DO UPDATE SET
                title=excluded.title,
                authors=excluded.authors,
                url=excluded.url,
                source=excluded.source,
                file_path=COALESCE(excluded.file_path, papers.file_path)
            """,
            (
                title,
                authors,
                url,
                pdf_url,
                source,
                file_path,
                datetime.utcnow().isoformat(),
            ),
        )
        conn.commit()
    finally:
        conn.close()


def list_paper_records() -> list[dict]:
    """Return cached paper metadata rows from local storage."""
    init_db()
    db_path, _ = _get_paths()
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """
            SELECT title, authors, url, pdf_url, source, file_path, added_at
            FROM papers
            ORDER BY added_at DESC, id DESC
            """
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()

