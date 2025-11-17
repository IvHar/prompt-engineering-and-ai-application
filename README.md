# AI Data Assistant

Minimal Streamlit app that generates synthetic datasets from PostgreSQL DDL (using Vertex AI Gemini when available) and lets you query them using natural language (NL→SQL) executed against a PostgreSQL instance.

**Key Components**
- **`app.py`**: Streamlit UI with three workspaces: Data Generation, Talk to your data (NL→SQL), and Query history.
- **`synthetic_factory.py`**: Parses PostgreSQL DDL and builds synthetic tables. Uses Vertex AI Gemini when available; falls back to a deterministic local generator otherwise.
- **`sql_agent.py`**: Converts natural language to a single PostgreSQL `SELECT` using Gemini and executes it against a running PostgreSQL (loads generated tables into the DB first).
- **`observability.py`**: Optional Langfuse tracing helpers for data generation and NL→SQL pipelines.
- **`schemas/`**: Example DDL schemas you can upload to the Data Generation tab (company, library, restaurants).
- **`Dockerfile` & `docker-compose.yml`**: Containerised app + PostgreSQL for easy local runs.

**Quick start (recommended: Docker Compose)**
- Build and run the app and a local PostgreSQL:

```bash
docker-compose up --build
```

- Open the Streamlit UI at `http://localhost:8501`.

**Environment variables**
- `GOOGLE_CLOUD_PROJECT`, `GOOGLE_REGION`: for Vertex AI (optional).
- `GOOGLE_APPLICATION_CREDENTIALS` or `gcloud auth application-default login`: to authenticate with Vertex AI if using LLM features.
- `LANGFUSE_SECRET_KEY`, `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_HOST`: enable Langfuse tracing (optional).
- `POSTGRES_HOST`, `POSTGRES_PORT`, `POSTGRES_DB`, `POSTGRES_USER`, `POSTGRES_PASSWORD`: PostgreSQL connection for `sql_agent.py` (defaults configured in `docker-compose.yml`).

**Usage**
- Data Generation: upload a DDL file (from `schemas/` or your own), provide high-level instructions and settings, then click "Generate data". The app uses Gemini when configured, otherwise a local fallback generator produces deterministic rows.
- Talk to your data: after generating data, ask natural-language questions. The app builds a single `SELECT` statement (PostgreSQL dialect) and executes it against the loaded tables, returning a table and quick chart options.
- Query history: view recent NL→SQL queries and results.

**Troubleshooting**
- If you get authentication errors for Google services, ensure ADC is configured (`gcloud auth application-default login`) or set `GOOGLE_APPLICATION_CREDENTIALS` to a service account JSON.
- If Langfuse tracing fails, the app logs a warning and continues without tracing.

**License & attribution**
- This repository is an educational/demo project — adapt and use as needed.

----
For more details, inspect `app.py`, `synthetic_factory.py`, and `sql_agent.py`.
