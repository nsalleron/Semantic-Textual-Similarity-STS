# syntax=docker/dockerfile:1
FROM python:3.11-slim

# System deps kept minimal; torch CPU wheels bundle their own libs.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models_cache

WORKDIR /app

# Install Python dependencies first for better layer caching.
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy the application code and any prebuilt artifacts.
COPY api ./api
COPY scripts ./scripts
COPY pytest.ini ./
# Artifacts (tfidf vectorizer, bert checkpoint, thresholds.json) if present.
COPY artifact[s] ./artifacts

EXPOSE 8000

# By default load all models at startup. Override with SSA_MODEL_IDS to load a
# subset, or SSA_SKIP_MODEL_LOAD=1 to skip loading entirely.
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
