Deployment guide

Recommended approach: deploy the FastAPI backend to Render (free/quick) or Railway, and keep the Streamlit frontend on Streamlit Cloud.

Steps (Render):

1. Push your repo to GitHub (if not already).

2. Create a new Web Service on Render (https://dashboard.render.com/new/web-service).
   - Connect your GitHub repo and choose the `main` branch.
   - Build command: `pip install -r requirements.txt`
   - Start command: `uvicorn backend.app:app --host 0.0.0.0 --port $PORT --workers 1`
   - Instance type: choose according to needs.

3. Set environment variables on Render:
   - `CORS_ORIGINS`: set to `https://aira26.streamlit.app` (or `https://aira26.streamlit.app,http://localhost:8501` if testing locally)
   - Any backend-specific envs (e.g., `ASSISTANT_MIN_VECTOR_HITS`, `ASSISTANT_TRAIN_ON_STARTUP`, API keys)

4. Deploy and note the public URL (e.g., `https://aira-backend.onrender.com`).

Steps (Streamlit Cloud):

1. In your Streamlit app settings, set the following secrets/env vars:
   - `BACKEND_URL`: `https://<your-backend-host>` (e.g., `https://aira-backend.onrender.com`)
   - `CORS_ORIGINS`: (optional) ensure it includes the Streamlit URL

2. Re-deploy the Streamlit app (or restart). The UI now calls the backend via `BACKEND_URL` and will use remote APIs.

Local testing (before deploying):

1. Run the backend locally:

```bash
pip install -r requirements.txt
uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload
```

2. In a separate terminal run Streamlit locally:

```bash
streamlit run streamlit_app.py
```

3. Set `BACKEND_URL` locally before running Streamlit (optional):

```bash
export BACKEND_URL=http://localhost:8000
# On Windows PowerShell:
# $env:BACKEND_URL = 'http://localhost:8000'
```

Notes and tips:
- If your backend requires building a vector index, ensure `data/vectorstore` is available or allow the service to run the research pipeline (set `ASSISTANT_MIN_VECTOR_HITS` low if you want fallbacks).
- For production, consider using a managed DB or object store for downloaded PDFs and the vectorstore.
- Use HTTPS endpoint for `BACKEND_URL` to avoid mixed-content issues on Streamlit Cloud.
