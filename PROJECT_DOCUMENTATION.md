# AI Research Assistant Project Documentation

## 1. Project Overview

### Project name
AI Research Assistant

### Project type
Academic assistant application with:
- A Streamlit app
- A FastAPI backend
- LLM-powered research workflows
- Local vector search and cached PDF storage

### Core intention of the project
This project is designed to help a researcher, student, or academic user work faster across the early and middle stages of research. The system combines paper discovery, paper review, question answering, reference generation, PDF caching, and corpus-based assistant chat into one interface.

The intention is not just to answer generic questions. The system tries to become a research workspace with grounded outputs based on:
- live scholarly sources such as arXiv, Semantic Scholar, OpenAlex, CORE, DOAJ, Europe PMC, and SerpAPI-backed search
- locally cached PDFs
- a locally trained lightweight assistant index built from stored paper content

### Main user-facing capabilities
1. Research exploration across multiple scholarly sources
2. Structured paper review from an uploaded PDF
3. Question answering on an uploaded paper
4. Reference generation for a topic
5. Guided research paper writing workflow
6. Local assistant chat grounded in cached research materials
7. PDF downloading, caching, and reuse for later retrieval

## 2. High-Level Architecture

### Frontend
The primary user interface is a Streamlit application. The entrypoint is `streamlit_app.py`, and most of the UI logic now lives in a small `ui/` package so beginners can follow the code more easily. It mirrors the original chat-style product flow with:
- sidebar chat sessions
- a role-based mode dropdown
- chat-style message rendering
- reviewer PDF upload flow
- explorer, reviewer, and writer modes

### Backend
The backend is a FastAPI application in `backend/`. It exposes API endpoints for all research workflows, while the Streamlit app calls the same workflows directly for deployment simplicity.

### Data layer
The application stores data in two main local forms:
- SQLite metadata database for downloaded PDFs
- FAISS vector store for semantic retrieval

### Intelligence layer
The project uses:
- Groq-hosted LLMs as the primary generation backend
- optional OpenAI-compatible `gpt-oss` endpoint as an alternative
- Hugging Face sentence-transformer embeddings for vector search
- BM25 plus FAISS hybrid retrieval for the local assistant

## 3. Request Flow Summary

### A. Research assistant chat
1. User sends a prompt from the chat-style Streamlit interface or an API client.
2. The app calls the assistant workflow.
3. Backend retrieves relevant local chunks using BM25 + FAISS.
4. If the local corpus looks relevant, the assistant answers from the VectorDB-backed corpus first.
5. If the local corpus is not relevant enough, the backend falls back to external search.
6. External results are returned to the user and incremental learning starts in the background by downloading PDFs and retraining the local assistant index.
7. The UI renders the answer and cited sources.

### B. Research exploration
1. User submits a research topic.
2. Backend fetches papers from multiple sources.
3. Abstracts and optionally full texts are combined.
4. A prompt asks the LLM to return structured JSON.
5. Output is validated, scored, filtered for relevance, cached, and returned.

### C. Paper review
1. User uploads a PDF.
2. Backend extracts text from the PDF.
3. The text is chunked and summarized chunk by chunk.
4. A reviewer chain produces structured review JSON.
5. A readable review summary is returned to the UI.

### D. Writer workflow
1. The UI starts a guided conversational flow.
2. The backend keeps a lightweight state machine in the request payload.
3. The user is walked through title, mode, section choice, and drafting steps.

## 4. Folder Structure

```text
AI-Research-Assistant/
├─ backend/
│  ├─ app.py
│  ├─ main.py
│  ├─ assistant_model.py
│  ├─ chains.py
│  ├─ retriever.py
│  ├─ embeddings.py
│  ├─ storage.py
│  ├─ pdf_utils.py
│  ├─ helpers.py
│  └─ __init__.py
├─ frontend/
│  ├─ package.json
│  ├─ public/
│  └─ src/
│     ├─ App.js
│     ├─ api.js
│     ├─ App.css
│     ├─ index.js
│     └─ index.css
├─ data/
├─ paper_db/
├─ vectorstore/
├─ requirements.txt
├─ streamlit_app.py
└─ .env
```

## 5. Backend Explanation

### `backend/app.py`
This is the main FastAPI entry point.

Its responsibilities are:
- create the API app
- configure CORS
- manage response caching
- define all HTTP routes
- trigger background assistant training on startup

Important implementation ideas:
- Heavy backend logic is lazily imported through `_backend_main()` so startup stays lighter.
- Research explorer results are cached in memory and on disk using `api_cache.json`.
- Request and response models live in `backend/schemas.py`.
- Explorer caching is isolated in `backend/explorer_cache.py`.
- Explorer URL cleanup, relevance filtering, and fallback shaping live in `backend/explorer_utils.py`.
- Startup optionally launches assistant training in a background thread.
- The file also contains the server-driven writer workflow endpoint.

Main API routes:
- `GET /api/health`
- `POST /api/review/upload`
- `POST /api/review/qa`
- `POST /api/research/explore`
- `POST /api/reference`
- `POST /api/download`
- `GET /api/assistant/status`
- `POST /api/assistant/train`
- `POST /api/assistant/chat`
- `POST /api/writer/step`

### `backend/main.py`
This file is the orchestration core of the system.

Its responsibilities are:
- initialize LLMs
- choose between Groq and optional OSS model backends
- fetch papers from live sources
- download full PDFs when possible
- build and update the vector store
- run research explorer, paper reviewer, paper QA, and reference generation workflows
- validate structured outputs with Pydantic
- use LangGraph retry loops when generated output fails validation

Important design intention:
This file separates workflow logic from HTTP route definitions. `app.py` handles transport and response shaping, while `main.py` handles research operations.

Key patterns used here:
- fallback model strategy for rate-limit situations
- topic relevance filtering
- document normalization
- structured JSON repair when model output is malformed
- graph-based retry loops for stricter response validation

### `backend/chains.py`
This file defines all prompt templates and converts them into LangChain runnable chains.

Why it exists:
Prompt engineering is centralized here so the project can keep:
- workflow logic in one place
- model prompts in another place

Defined prompt families include:
- research explanation and structured table generation
- peer review generation
- paper QA
- chunk summarization
- reference generation
- JSON repair
- research gap synthesis
- assistant grounded QA

The prompt design is intentionally strict. Several prompts explicitly require:
- JSON only
- no hallucinated papers
- no extra markdown
- fixed schema outputs

### `backend/retriever.py`
This file is responsible for external retrieval and vector store management.

It provides:
- arXiv retrieval through LangChain
- direct API integrations for Semantic Scholar, OpenAlex, CORE, DOAJ, Europe PMC
- SerpAPI-based retrieval for Google Scholar, ResearchGate, general web, and ScienceDirect discovery
- conversion utilities between LangChain documents and flat table rows
- FAISS vector store creation and loading

Design intention:
The project does not rely on a single source. It tries to improve breadth and resilience by combining multiple scholarly sources, then normalizing the results into a shared structure.

### `backend/assistant_model.py`
This module powers the local trained assistant.

Its responsibilities are:
- collect text chunks from cached PDFs and the FAISS docstore
- deduplicate them
- save an assistant index to disk
- build a BM25 retriever over the stored chunks
- combine BM25 results with FAISS similarity results
- rerank and diversify sources
- answer questions with grounded context

Why this module matters:
This is what turns the project from a live-search tool into a reusable local research memory. Once papers are cached and indexed, the assistant can answer based on the accumulated corpus instead of depending only on live search every time.

Retrieval strategy:
- BM25 handles lexical overlap well
- FAISS handles semantic similarity
- hybrid reranking improves relevance
- diversity filtering avoids overusing only one paper

### `backend/embeddings.py`
This file manages embeddings.

It uses `sentence-transformers/all-MiniLM-L6-v2` and wraps it with `_SafeEmbeddings` so non-string or malformed inputs do not crash retrieval. If embedding initialization fails, the code falls back to dummy zero-vector embeddings to keep the application running.

Design intention:
Favor graceful degradation over hard crashes.

### `backend/storage.py`
This module implements local paper storage.

It manages:
- SQLite database initialization
- PDF caching lookup by `pdf_url`
- PDF file persistence
- metadata upsert for paper records
- listing stored records

Why it exists:
Downloaded papers are not just temporary files. They become reusable local assets for later retrieval and assistant training.

### `backend/pdf_utils.py`
This file handles PDF text extraction and chunking.

Main functions:
- `extract_text(pdf_file)` loads page text with `PyPDFLoader`
- `chunk_text(text)` splits long content for retrieval and summarization

### `backend/helpers.py`
This module contains defensive utility functions.

Examples:
- environment variable loading
- robust JSON cleanup and parsing
- APA reference formatting
- author normalization
- heuristic extraction of datasets and models
- directory creation
- HTML stripping

Its intention is to keep the rest of the code cleaner and more fault tolerant.

## 6. Frontend Explanation

### `streamlit_app.py`
This is the main Streamlit interface.

Its responsibilities are:
- provide a chat-style UI similar to the original frontend
- manage chat sessions in the sidebar
- let the user switch roles with a dropdown
- call the existing backend workflows directly
- handle reviewer PDF upload and writer-state progression

Notable design decisions:
- the Streamlit app keeps the original role-based chat flow instead of a tabbed workspace
- assistant and writer flows use session state to preserve context between interactions
- deployment stays simple because the UI reuses existing Python workflows directly

## 7. Data and Persistence

### SQLite
The SQLite database stores paper metadata in a `papers` table with fields such as:
- title
- authors
- url
- pdf_url
- source
- file_path
- added_at

### PDF cache
Downloaded PDFs are stored under the paper database directory, typically inside a `pdfs/` folder.

### FAISS vector store
The vector store stores embedded text chunks for semantic similarity search.

### Assistant index
The trained local assistant stores:
- `assistant_index.json`
- `assistant_config.json`

These artifacts represent the lightweight trained corpus used by the assistant chat feature.

## 8. LLM and Retrieval Strategy

### Model strategy
The system supports:
- Groq LLMs as the primary path
- a secondary Groq model for fallback on rate limits
- optional OpenAI-compatible `gpt-oss` endpoint

### Why structured prompts are used
Many workflows need machine-readable output, especially:
- research tables
- review results
- references

That is why the prompts aggressively ask for strict JSON and the code validates outputs with Pydantic and repair chains.

### Why hybrid retrieval is used
Pure vector search can miss exact keyword matches.
Pure keyword search can miss semantically relevant content.

Combining BM25 and FAISS gives the assistant a more balanced retrieval system.

## 9. API Documentation

### `POST /api/assistant/chat`
Purpose: answer a user prompt using the local trained corpus.

Input:
- `prompt`
- `chat_history` optional

Output:
- model metadata
- grounded answer
- supporting sources

### `POST /api/research/explore`
Purpose: discover papers, summarize them, identify gaps, and generate an idea.

Input:
- `topic`
- `chat_history` optional
- `focus_topic` optional
- `use_live` optional
- `force_refresh` optional

Output:
- structured paper table
- research gaps
- assistant reply
- generated research idea
- implementation steps

### `POST /api/review/upload`
Purpose: upload a PDF and generate a structured review.

### `POST /api/review/qa`
Purpose: ask questions about an uploaded paper's extracted text.

### `POST /api/reference`
Purpose: generate references for a topic.

### `POST /api/download`
Purpose: download and cache papers for a topic, then refresh assistant training.

### `POST /api/writer/step`
Purpose: move the guided writer flow forward one step at a time.

## 10. Environment Variables

The project depends on several environment variables. Important ones include:

### Model and inference
- `GROQ_API_KEY`
- `GROQ_MODEL_ID_PRIMARY`
- `GROQ_MODEL_ID_SECONDARY`
- `GROQ_REASONING_EFFORT`
- `GROQ_MAX_TOKENS`
- `USE_GPT_OSS`
- `OSS_BASE_URL`
- `OSS_MODEL_ID`
- `OSS_API_KEY`

### Retrieval and storage
- `FAISS_PERSIST_DIR`
- `PAPER_DB_DIR`
- `ASSISTANT_MODEL_DIR`
- `MAX_PDF_DOWNLOADS`
- `DOWNLOAD_ARXIV_PDFS`
- `DOWNLOAD_EXTERNAL_PDFS`

### Search provider keys
- `SEMANTIC_SCHOLAR_API_KEY`
- `SERPAPI_API_KEY`
- `CORE_API_KEY`

### Frontend and runtime behavior
- `CORS_ORIGINS`
- `ASSISTANT_ONLY`
- `ASSISTANT_TRAIN_ON_STARTUP`
- `ASSISTANT_MODEL_ONLY`
- `LOCAL_ONLY`
- `FAST_MODE`
- `FAST_MAX_PRIMARY`
- `FAST_MAX_SECONDARY`

### Model cache paths
- `SENTENCE_TRANSFORMERS_HOME`
- `HF_HOME`
- `TRANSFORMERS_CACHE`

## 11. Deployment

### Local development
Python app:
- install Python dependencies from `requirements.txt`
- run Streamlit with `streamlit run streamlit_app.py`

Optional API server:
- run FastAPI with Uvicorn

### Streamlit deployment
The project now uses `streamlit_app.py` as the primary deployment entrypoint.

Typical deployment flow is:
1. install Python dependencies from `requirements.txt`
2. provide the required environment variables
3. start the app with `streamlit run streamlit_app.py`

## 12. Code Intentions by Feature

### Research Explorer intention
Help the user survey a topic quickly by turning multiple paper sources into:
- a paper table
- concise summaries
- research gaps
- a synthesized future idea

### Reviewer intention
Turn a raw PDF into a structured peer-review style response that is easier to consume than reading the whole paper at once.

### Writer intention
Guide users who may not know how to structure a conference or journal paper, using a conversational academic workflow.

### Assistant intention
Create a grounded assistant that becomes more useful as the local paper corpus grows over time.

## 13. Strengths of the Current Design

1. Clear separation between API layer, orchestration layer, retrieval layer, storage layer, and UI.
2. Good fault tolerance through fallback models, JSON repair, relevance filtering, and defensive helpers.
3. Multiple live retrieval sources improve source diversity.
4. Local PDF caching makes the system more useful over time.
5. Hybrid retrieval makes the assistant more grounded than a simple chat wrapper.
6. Output validation adds structure and reliability to LLM workflows.

## 14. Current Limitations and Risks

1. There are no visible automated tests in the current repository.
2. Some workflows rely heavily on prompt quality and may still produce imperfect structured data.
3. Search quality depends on external API availability and keys.
4. Dummy embedding fallback preserves uptime but can reduce retrieval quality significantly.
5. The writer workflow is currently template-driven rather than fully generative in all branches.
6. `backend/main.py` is still the largest and most advanced file in the project, so it is the next place where deeper modularization would help beginners.
7. Caching and indexing behavior exist, but observability and admin tooling are limited.

## 15. File-by-File Intent Summary

### Backend
- `backend/app.py`: HTTP interface, caching, startup tasks, and writer endpoint
- `backend/schemas.py`: shared request and response models
- `backend/explorer_cache.py`: small cache helper for explorer responses
- `backend/explorer_utils.py`: explorer URL fixing, filtering, and fallback helpers
- `backend/main.py`: research workflow orchestration and validation
- `backend/chains.py`: prompts and chain builders
- `backend/retriever.py`: source retrieval and FAISS management
- `backend/assistant_model.py`: local corpus training and grounded assistant QA
- `backend/embeddings.py`: safe embedding wrappers and persistence path resolution
- `backend/storage.py`: SQLite and cached PDF metadata management
- `backend/pdf_utils.py`: PDF extraction and chunking
- `backend/helpers.py`: shared defensive utilities

### Root
- `requirements.txt`: Python dependency list
- `streamlit_app.py`: small Streamlit entrypoint

### UI
- `ui/config.py`: page config, modes, and CSS
- `ui/state.py`: Streamlit session state helpers
- `ui/helpers.py`: UI utility helpers such as chat history and URL fixing
- `ui/services.py`: UI-to-backend workflow actions
- `ui/rendering.py`: reusable rendering functions

## 16. How to Explain This Project in a Viva, Demo, or Report

A strong short explanation would be:

"This project is an AI Research Assistant that helps users discover research papers, review PDFs, ask questions about papers, generate references, and build research drafts. The backend uses FastAPI, LangChain, LangGraph, external scholarly APIs, and a local FAISS-based retrieval system. The main user interface now runs in Streamlit, which gives the project a simpler integrated workspace for the core research flows. One of the key ideas of the project is that it gradually builds a reusable local knowledge base from downloaded PDFs, so the assistant becomes grounded in the user's own research corpus over time."

## 17. Suggested Future Improvements

1. Add automated tests for API routes, retrieval normalization, and JSON validation.
2. Continue the same modularization style inside `backend/main.py`, especially around research explorer orchestration.
3. Add authentication if the app is intended for multiple users.
4. Add structured logging and monitoring around retrieval failures and model errors.
5. Add a document management screen for cached papers and training status.
6. Add export options for research tables, reviews, and generated writing.
7. Add stronger citation grounding and source attribution for generated ideas.

## 18. Final Summary

This project is more than a simple chatbot. It is a research workflow system that combines discovery, analysis, review, retrieval, caching, and guided writing into one application. The backend is designed around modular responsibilities, the Streamlit interface provides a practical chat-based user experience, and the local corpus pipeline gives the system a long-term memory built from real research materials.

If this document is used for submission or presentation, it already covers:
- project overview
- intention
- architecture
- code explanation
- module responsibilities
- feature explanation
- deployment
- strengths
- limitations
- future work
