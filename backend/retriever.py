"""Retrievers and vector store utilities."""

from __future__ import annotations

import os
from typing import List

from langchain_community.retrievers import ArxivRetriever
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from .embeddings import create_embeddings, create_dummy_embeddings, get_faiss_persist_dir
from .helpers import ensure_directory, load_env_var, authors_to_str, clean_authors

import time
import requests


def arxiv_search(query: str, max_results: int = 5) -> List[Document]:
    """Retrieve arXiv papers as LangChain Documents.

    Metadata fields include title, authors, abstract, year, and pdf_url.
    """
    if not query:
        return []

    for attempt in range(2):
        try:
            retriever = ArxivRetriever(
                load_max_docs=max_results,
                get_full_documents=False,
            )
            # LangChain retriever API moved from get_relevant_documents -> invoke in newer versions.
            if hasattr(retriever, "get_relevant_documents"):
                docs = retriever.get_relevant_documents(query)
            else:
                docs = retriever.invoke(query)
            break
        except Exception as exc:
            if "429" in str(exc) and attempt == 0:
                time.sleep(2)
                continue
            # Return an empty list but surface error in metadata for transparency.
            return [
                Document(
                    page_content="",
                    metadata={"error": f"arXiv retrieval failed: {exc}"},
                )
            ]

    # Normalize metadata for downstream use.
    normalized: List[Document] = []
    for d in docs:
        meta = dict(d.metadata or {})
        title = meta.get("Title") or meta.get("title") or ""
        authors = meta.get("Authors") or meta.get("authors") or []
        if isinstance(authors, str):
            authors = [a.strip() for a in authors.split(",") if a.strip()]
        authors = clean_authors([a for a in authors if isinstance(a, str)])
        summary = meta.get("Summary") or meta.get("summary") or d.page_content
        entry_id = meta.get("Entry ID") or meta.get("entry_id") or meta.get("id") or ""
        pdf_url = meta.get("pdf_url") or meta.get("pdf") or meta.get("PDF_URL") or ""
        if not pdf_url and isinstance(entry_id, str) and "arxiv.org/abs/" in entry_id:
            pdf_url = entry_id.replace("/abs/", "/pdf/") + ".pdf"
        url = entry_id or pdf_url
        year = meta.get("Published") or meta.get("published") or ""
        normalized.append(
            Document(
                page_content=summary or d.page_content,
                metadata={
                    "title": title,
                    "authors": authors,
                    "abstract": summary,
                    "year": str(year)[:4] if year else "",
                    "url": url,
                    "pdf_url": pdf_url,
                },
            )
        )

    return normalized


def docs_to_rows(docs: List[Document], source: str = "arxiv") -> List[dict]:
    """Convert LangChain Documents to flat rows for UI tables."""
    rows: List[dict] = []
    for d in docs:
        meta = d.metadata or {}
        url = meta.get("url", "")
        doi = meta.get("doi", "")
        if doi:
            url = f"https://doi.org/{doi}"
        rows.append(
            {
                "title": meta.get("title", ""),
                "authors": authors_to_str(meta.get("authors", [])),
                "year": meta.get("year", ""),
                "url": url,
                "pdf_url": meta.get("pdf_url", ""),
                "doi": meta.get("doi", ""),
                "abstract": meta.get("abstract", "") or d.page_content,
                "source": source,
            }
        )
    return rows


def rows_to_docs(rows: List[dict]) -> List[Document]:
    """Convert flat rows into Documents for vector store ingestion."""
    docs: List[Document] = []
    for r in rows:
        content = r.get("abstract") or r.get("title") or ""
        if not isinstance(content, str):
            # Coerce lists/dicts/etc. into a safe string for embeddings.
            try:
                content = str(content)
            except Exception:
                content = ""
        docs.append(
            Document(
                page_content=content,
                metadata={
                    "title": r.get("title", ""),
                    "authors": r.get("authors", ""),
                    "year": r.get("year", ""),
                    "url": r.get("url", ""),
                    "pdf_url": r.get("pdf_url", ""),
                    "doi": r.get("doi", ""),
                    "source": r.get("source", ""),
                },
            )
        )
    return docs


def semantic_scholar_search(query: str, max_results: int = 5) -> tuple[List[dict], str | None]:
    """Search Semantic Scholar API for papers (optional API key)."""
    if not query:
        return [], None
    api_key = load_env_var("SEMANTIC_SCHOLAR_API_KEY")
    headers = {"x-api-key": api_key} if api_key else {}
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {
        "query": query,
        "limit": max_results,
        "fields": "title,authors,year,abstract,url,openAccessPdf,doi,externalIds",
    }
    try:
        # Simple backoff to reduce 429s.
        for attempt in range(2):
            resp = requests.get(url, params=params, headers=headers, timeout=20)
            if resp.status_code == 429 and attempt == 0:
                time.sleep(2)
                continue
            if resp.status_code == 429:
                return [], (
                    "Semantic Scholar rate limit hit (429). "
                    "Add SEMANTIC_SCHOLAR_API_KEY to .env or retry later."
                )
            resp.raise_for_status()
            data = resp.json()
            break
        rows = []
        for p in data.get("data", []):
            authors = [a.get("name", "") for a in p.get("authors", []) if a.get("name")]
            pdf_url = ""
            open_pdf = p.get("openAccessPdf") or {}
            if isinstance(open_pdf, dict):
                pdf_url = open_pdf.get("url", "") or ""
            doi = p.get("doi", "") or ""
            if not doi:
                ext_ids = p.get("externalIds") or {}
                if isinstance(ext_ids, dict):
                    doi = ext_ids.get("DOI", "") or ""
            rows.append(
                {
                    "title": p.get("title", ""),
                    "authors": ", ".join(authors),
                    "year": p.get("year", ""),
                    "url": p.get("url", ""),
                    "pdf_url": pdf_url,
                    "doi": doi,
                    "abstract": p.get("abstract", ""),
                    "source": "semantic_scholar",
                }
            )
        return rows, None
    except Exception as exc:
        return [], f"Semantic Scholar search failed: {exc}"


def semantic_scholar_open_access_search(query: str, max_results: int = 5) -> tuple[List[dict], str | None]:
    """Search Semantic Scholar and return only open-access PDFs."""
    if not query:
        return [], None
    rows, warn = semantic_scholar_search(query, max_results=max_results * 2)
    if warn:
        return [], warn
    open_rows = [r for r in rows if r.get("pdf_url")]
    return open_rows[:max_results], None


def serpapi_scholar_search(query: str, max_results: int = 5) -> tuple[List[dict], str | None]:
    """Search Google Scholar via SerpAPI (no direct scraping)."""
    if not query:
        return [], None
    api_key = load_env_var("SERPAPI_API_KEY")
    if not api_key:
        return [], "SERPAPI_API_KEY not set for Google Scholar search."
    params = {"engine": "google_scholar", "q": query, "api_key": api_key, "num": max_results}
    try:
        resp = requests.get("https://serpapi.com/search.json", params=params, timeout=20)
        if resp.status_code == 401:
            return [], "Google Scholar search failed: SerpAPI key unauthorized."
        resp.raise_for_status()
        data = resp.json()
        rows = []
        for r in data.get("organic_results", [])[:max_results]:
            pdf_url = ""
            resources = r.get("resources", []) or []
            for res in resources:
                if (res.get("file_format") or "").lower() == "pdf" and res.get("link"):
                    pdf_url = res.get("link")
                    break
            if not pdf_url:
                link = r.get("link", "")
                if isinstance(link, str) and link.lower().endswith(".pdf"):
                    pdf_url = link
            rows.append(
                {
                    "title": r.get("title", ""),
                    "authors": (r.get("publication_info", {}) or {}).get("summary", ""),
                    "year": "",
                    "url": r.get("link", ""),
                    "pdf_url": pdf_url,
                    "doi": "",
                    "abstract": r.get("snippet", ""),
                    "source": "google_scholar",
                }
            )
        return rows, None
    except Exception as exc:
        return [], f"Google Scholar search failed: {exc}"


def serpapi_researchgate_search(query: str, max_results: int = 5) -> tuple[List[dict], str | None]:
    """Search ResearchGate results via SerpAPI Google search."""
    if not query:
        return [], None
    api_key = load_env_var("SERPAPI_API_KEY")
    if not api_key:
        return [], "SERPAPI_API_KEY not set for ResearchGate search."
    q = f"site:researchgate.net {query}"
    params = {"engine": "google", "q": q, "api_key": api_key, "num": max_results}
    try:
        resp = requests.get("https://serpapi.com/search.json", params=params, timeout=20)
        if resp.status_code == 401:
            return [], "ResearchGate search failed: SerpAPI key unauthorized."
        resp.raise_for_status()
        data = resp.json()
        rows = []
        for r in data.get("organic_results", [])[:max_results]:
            pdf_url = ""
            link = r.get("link", "")
            if isinstance(link, str) and link.lower().endswith(".pdf"):
                pdf_url = link
            rows.append(
                {
                    "title": r.get("title", ""),
                    "authors": "",
                    "year": "",
                    "url": r.get("link", ""),
                    "pdf_url": pdf_url,
                    "doi": "",
                    "abstract": r.get("snippet", ""),
                    "source": "researchgate",
                }
            )
        return rows, None
    except Exception as exc:
        return [], f"ResearchGate search failed: {exc}"


def serpapi_web_search(query: str, max_results: int = 5) -> tuple[List[dict], str | None]:
    """General web search via SerpAPI (used for augmentation)."""
    if not query:
        return [], None
    api_key = load_env_var("SERPAPI_API_KEY")
    if not api_key:
        return [], "SERPAPI_API_KEY not set for web search."
    params = {"engine": "google", "q": query, "api_key": api_key, "num": max_results}
    try:
        resp = requests.get("https://serpapi.com/search.json", params=params, timeout=20)
        if resp.status_code == 401:
            return [], "Web search failed: SerpAPI key unauthorized."
        resp.raise_for_status()
        data = resp.json()
        rows = []
        for r in data.get("organic_results", [])[:max_results]:
            rows.append(
                {
                    "title": r.get("title", ""),
                    "authors": "",
                    "year": "",
                    "url": r.get("link", ""),
                    "pdf_url": "",
                    "doi": "",
                    "abstract": r.get("snippet", ""),
                    "source": "web",
                }
            )
        return rows, None
    except Exception as exc:
        return [], f"Web search failed: {exc}"


def serpapi_sciencedirect_search(query: str, max_results: int = 5) -> tuple[List[dict], str | None]:
    """Search ScienceDirect for direct PDF links via SerpAPI (no scraping)."""
    if not query:
        return [], None
    api_key = load_env_var("SERPAPI_API_KEY")
    if not api_key:
        return [], "SERPAPI_API_KEY not set for ScienceDirect search."
    q = f"site:sciencedirect.com {query} filetype:pdf"
    params = {"engine": "google", "q": q, "api_key": api_key, "num": max_results}
    try:
        resp = requests.get("https://serpapi.com/search.json", params=params, timeout=20)
        if resp.status_code == 401:
            return [], "ScienceDirect search failed: SerpAPI key unauthorized."
        resp.raise_for_status()
        data = resp.json()
        rows = []
        for r in data.get("organic_results", [])[:max_results]:
            link = r.get("link", "")
            pdf_url = link if isinstance(link, str) and link.lower().endswith(".pdf") else ""
            rows.append(
                {
                    "title": r.get("title", ""),
                    "authors": "",
                    "year": "",
                    "url": r.get("link", ""),
                    "pdf_url": pdf_url,
                    "doi": "",
                    "abstract": r.get("snippet", ""),
                    "source": "sciencedirect",
                }
            )
        return rows, None
    except Exception as exc:
        return [], f"ScienceDirect search failed: {exc}"


def openalex_search(query: str, max_results: int = 5) -> tuple[List[dict], str | None]:
    """Search OpenAlex for open-access papers and PDFs when available."""
    if not query:
        return [], None

    def _openalex_abstract_from_index(index: dict | None) -> str:
        if not isinstance(index, dict) or not index:
            return ""
        positions: dict[int, str] = {}
        for word, idxs in index.items():
            if not isinstance(word, str) or not isinstance(idxs, list):
                continue
            for pos in idxs:
                if isinstance(pos, int):
                    positions[pos] = word
        if not positions:
            return ""
        return " ".join(positions[i] for i in sorted(positions)).strip()

    try:
        params = {
            "search": query,
            "per-page": max_results,
            "filter": "is_oa:true",
            "select": "title,authorships,publication_year,primary_location,doi,open_access,abstract_inverted_index",
        }
        resp = requests.get("https://api.openalex.org/works", params=params, timeout=20)
        resp.raise_for_status()
        data = resp.json()
        rows = []
        for w in data.get("results", []):
            authors = []
            for a in w.get("authorships", [])[:6]:
                name = (a.get("author", {}) or {}).get("display_name", "")
                if name:
                    authors.append(name)
            doi = w.get("doi", "") or ""
            pdf_url = ""
            loc = w.get("primary_location") or {}
            if isinstance(loc, dict):
                pdf_url = (loc.get("pdf_url") or "") if isinstance(loc.get("pdf_url"), str) else ""
            abstract = _openalex_abstract_from_index(w.get("abstract_inverted_index"))
            rows.append(
                {
                    "title": w.get("title", ""),
                    "authors": ", ".join(authors),
                    "year": w.get("publication_year", ""),
                    "url": (loc.get("landing_page_url") or "") if isinstance(loc, dict) else "",
                    "pdf_url": pdf_url,
                    "doi": doi,
                    "abstract": abstract,
                    "source": "openalex",
                }
            )
        return rows, None
    except Exception as exc:
        return [], f"OpenAlex search failed: {exc}"


def core_search(query: str, max_results: int = 5) -> tuple[List[dict], str | None]:
    """Search CORE for open-access papers (API key optional)."""
    if not query:
        return [], None
    api_key = load_env_var("CORE_API_KEY")
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    params = {"q": query, "limit": max_results}
    try:
        resp = requests.get("https://api.core.ac.uk/v3/search/works", params=params, headers=headers, timeout=20)
        if resp.status_code == 401:
            return [], "CORE search failed: API key unauthorized."
        resp.raise_for_status()
        data = resp.json()
        rows = []
        for item in data.get("results", []):
            authors = item.get("authors", [])
            if isinstance(authors, list):
                authors_str = ", ".join([a.get("name", "") for a in authors if a.get("name")])
            else:
                authors_str = ""
            rows.append(
                {
                    "title": item.get("title", ""),
                    "authors": authors_str,
                    "year": item.get("yearPublished", ""),
                    "url": item.get("downloadUrl", "") or item.get("sourceFulltextUrls", [""])[0],
                    "pdf_url": item.get("downloadUrl", ""),
                    "doi": item.get("doi", ""),
                    "abstract": item.get("abstract", ""),
                    "source": "core",
                }
            )
        return rows, None
    except Exception as exc:
        return [], f"CORE search failed: {exc}"


def doaj_search(query: str, max_results: int = 5) -> tuple[List[dict], str | None]:
    """Search DOAJ for open-access articles."""
    if not query:
        return [], None
    try:
        params = {"page": 1, "pageSize": max_results, "query": query}
        resp = requests.get("https://doaj.org/api/v2/search/articles", params=params, timeout=20)
        resp.raise_for_status()
        data = resp.json()
        rows = []
        for item in data.get("results", []):
            bib = (item.get("bibjson") or {})
            links = bib.get("link", []) or []
            pdf_url = ""
            for l in links:
                if l.get("type") == "fulltext" and l.get("url"):
                    pdf_url = l.get("url")
                    break
            rows.append(
                {
                    "title": bib.get("title", ""),
                    "authors": ", ".join([a.get("name", "") for a in bib.get("author", []) if a.get("name")]),
                    "year": bib.get("year", ""),
                    "url": (links[0].get("url") if links else ""),
                    "pdf_url": pdf_url,
                    "doi": bib.get("doi", ""),
                    "abstract": bib.get("abstract", ""),
                    "source": "doaj",
                }
            )
        return rows, None
    except Exception as exc:
        return [], f"DOAJ search failed: {exc}"


def europe_pmc_search(query: str, max_results: int = 5) -> tuple[List[dict], str | None]:
    """Search Europe PMC for open-access articles with PDF links when available."""
    if not query:
        return [], None
    try:
        params = {"query": query, "format": "json", "pageSize": max_results, "resultType": "core"}
        resp = requests.get("https://www.ebi.ac.uk/europepmc/webservices/rest/search", params=params, timeout=20)
        resp.raise_for_status()
        data = resp.json()
        rows = []
        for item in data.get("resultList", {}).get("result", []):
            pdf_url = ""
            if item.get("hasPDF") == "Y" and item.get("pmcid"):
                pdf_url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/{item['pmcid']}/pdf"
            rows.append(
                {
                    "title": item.get("title", ""),
                    "authors": item.get("authorString", ""),
                    "year": item.get("pubYear", ""),
                    "url": item.get("fullTextUrl", "") or item.get("doi", ""),
                    "pdf_url": pdf_url,
                    "doi": item.get("doi", ""),
                    "abstract": item.get("abstractText", ""),
                    "source": "europe_pmc",
                }
            )
        return rows, None
    except Exception as exc:
        return [], f"Europe PMC search failed: {exc}"


def build_vector_store(docs: List[Document]) -> FAISS:
    """Build and persist a FAISS vector store from documents."""
    try:
        # Limit the number of documents to keep vector store lightweight
        max_docs = int(load_env_var("VECTORSTORE_MAX_DOCS", "1000") or "1000")
        if len(docs) > max_docs:
            # Keep most recent documents (assuming they're at the end)
            docs = docs[-max_docs:]
        
        embeddings = create_embeddings()
        persist_dir = get_faiss_persist_dir()
        ensure_directory(persist_dir)
        # Ensure all page_content values are strings for the embedding model.
        texts: List[str] = []
        metadatas: List[dict] = []
        for d in docs:
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
        if not texts:
            raise RuntimeError("No valid text content available to build the vector store.")
        vector_store = FAISS.from_texts(texts, embeddings, metadatas=metadatas)
        vector_store.save_local(persist_dir)
        return vector_store
    except Exception as exc:
        # Fallback to dummy embeddings if the tokenizer rejects inputs.
        if "TextEncodeInput" in str(exc):
            try:
                dummy = create_dummy_embeddings()
                vector_store = FAISS.from_texts(texts, dummy, metadatas=metadatas)
                vector_store.save_local(persist_dir)
                return vector_store
            except Exception as exc2:
                raise RuntimeError(f"Failed to build FAISS vector store: {exc2}") from exc2
        raise RuntimeError(f"Failed to build FAISS vector store: {exc}") from exc


def load_vector_store() -> FAISS | None:
    """Load a persisted FAISS vector store if present."""
    try:
        embeddings = create_embeddings()
        persist_dir = get_faiss_persist_dir()
        if not os.path.exists(persist_dir):
            return None

        # Expect index files inside the directory; if missing, return None.
        index_path = os.path.join(persist_dir, "index.faiss")
        if not os.path.exists(index_path):
            return None

        store = FAISS.load_local(persist_dir, embeddings, allow_dangerous_deserialization=True)
        # Ensure we always use the safe embedding wrapper for new queries/additions.
        store.embedding_function = embeddings
        return store
    except Exception as exc:
        raise RuntimeError(f"Failed to load FAISS vector store: {exc}") from exc
