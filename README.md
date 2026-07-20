# Semantic Similarity API

A production-ready **FastAPI** service that scores the semantic similarity of two
sentences using several interchangeable models. For each request it returns a
similarity score, a binary "same meaning" prediction (using an F1-optimized
threshold), a confidence level, the inference time, the embedding dimension and
the compute device.

## Features

- **5 models** behind one uniform interface:
  | id | alias | kind | embedding dim |
  |----|-------|------|---------------|
  | `tfidf` | – | TF-IDF + cosine | vocab size |
  | `bert_finetuned` | `bert` | fine-tuned BERT bi-encoder | 768 |
  | `all-MiniLM-L6-v2` | `minilm` | Sentence-Transformers | 384 |
  | `all-mpnet-base-v2` | `mpnet` | Sentence-Transformers | 768 |
  | `e5-base-v2` | `e5` | Sentence-Transformers (query/passage prefixes) | 768 |
- Models are loaded **once at startup** via a `ModelManager` singleton (memory cache).
- **Optimal decision thresholds** per model, computed by F1 grid search on STS Benchmark.
- Automatic **device detection**: CUDA → MPS (Apple Silicon) → CPU.
- Enriched responses: `confidence` (High/Medium/Low), `embedding_dim`, `device`, `inference_time_ms`.
- Fully **test-driven** (pytest) and **Dockerized**.

## Architecture

```
api/
  app.py                 # FastAPI app + startup lifespan (model loading)
  core/
    config.py            # settings, model registry, alias resolution
    device.py            # cuda/mps/cpu detection
    model_manager.py     # singleton: loads models + thresholds once
  models/
    base.py              # SimilarityModel interface: predict(t1, t2) -> [0,1]
    tfidf_model.py
    bert_finetuned_model.py
    sentence_transformer_model.py
  routers/               # health, models, predict
  schemas/               # Pydantic request/response models
  utils/                 # cosine, confidence bands, timing
scripts/
  fit_tfidf.py           # fit + save the TF-IDF vectorizer
  train_bert.py          # fine-tune + save the BERT checkpoint
  compute_thresholds.py  # F1 grid search -> thresholds.json
artifacts/               # generated: vectorizer, bert checkpoint, thresholds.json
tests/
```

## Endpoints

### `GET /health`
```json
{ "status": "OK" }
```

### `GET /models`
Returns the loaded models with their thresholds, embedding dims and aliases:
```json
{
  "device": "cpu",
  "models": [
    { "id": "all-mpnet-base-v2", "kind": "sentence_transformer",
      "embedding_dim": 768, "threshold": 0.81, "aliases": ["mpnet"] }
  ]
}
```

### `POST /predict`
Request:
```json
{ "model": "mpnet", "text1": "I love machine learning", "text2": "I enjoy AI and ML" }
```
Response:
```json
{
  "model": "mpnet",
  "similarity": 0.87,
  "prediction": true,
  "threshold": 0.81,
  "confidence": "High",
  "inference_time_ms": 23.0,
  "embedding_dim": 768,
  "device": "cpu"
}
```
Errors: `404` unknown model · `503` model known but not loaded (missing artifact)
· `422` empty/blank text or missing field.

## Setup

Requires Python 3.11+ (developed and tested on 3.13).

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Build the model artifacts

The neural Sentence-Transformers models (`minilm`, `mpnet`, `e5`) download
automatically from the Hugging Face Hub on first use. The TF-IDF and
fine-tuned BERT models must be built locally, and thresholds computed:

```bash
# 1. Fit the TF-IDF vectorizer on STS Benchmark -> artifacts/tfidf/vectorizer.joblib
python -m scripts.fit_tfidf

# 2. Fine-tune BERT on STS Benchmark -> artifacts/bert_finetuned/
#    Full run (≈ 8 epochs). Use --subset / --epochs for a quick smoke build.
python -m scripts.train_bert                     # full training
python -m scripts.train_bert --subset 2000 --epochs 1   # quick

# 3. Compute F1-optimal thresholds -> artifacts/thresholds.json
python -m scripts.compute_thresholds             # all loaded models
python -m scripts.compute_thresholds --models tfidf all-MiniLM-L6-v2
```

Models whose artifacts are missing are **skipped** at startup (the app still
serves the available ones), and requesting them returns `503`.

## Run

```bash
uvicorn api.app:app --host 0.0.0.0 --port 8000
```

Environment variables:
- `SSA_MODEL_IDS` — comma-separated subset to load at startup
  (e.g. `tfidf,all-MiniLM-L6-v2`) to avoid downloading every model.
- `SSA_SKIP_MODEL_LOAD=1` — start without loading any model.

Example requests:
```bash
curl localhost:8000/health
curl localhost:8000/models
curl -X POST localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"model":"mpnet","text1":"I love machine learning","text2":"I enjoy AI and ML"}'
```

## Docker

```bash
docker compose up --build
# then, in another shell:
curl localhost:8000/health
```
The compose file persists the Hugging Face cache and mounts `./artifacts`, so
locally built TF-IDF/BERT models and thresholds are used inside the container.
Set `SSA_MODEL_IDS` in `docker-compose.yml` to control which models load.

## Tests

```bash
# Fast suite (no downloads / no training):
pytest -m "not slow and not download"

# Heavy tests (download models, run a tiny training):
pytest -m slow
```

## Design notes

- **Similarity** is the cosine of the two embeddings, clamped to `[0, 1]`
  (matching STSB's normalized gold scores).
- **Binary label for tuning**: an STSB pair is "same meaning" when its
  normalized gold score ≥ `0.6` (raw ≥ 3/5). Configurable in `core/config.py`.
- **Confidence** reflects the distance of the similarity from the threshold:
  `≥ 0.15` → High, `≥ 0.05` → Medium, else Low.
- **e5** models require `query:` / `passage:` prefixes and L2 normalization,
  handled inside the wrapper.

## Security note

The API ships **without authentication** and binds to all interfaces. Do not
expose it directly to untrusted networks; place it behind an authenticating
reverse proxy / API gateway, or add an API-key dependency, before deployment.
