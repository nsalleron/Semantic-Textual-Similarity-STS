"""FastAPI application entrypoint.

Models are loaded once at startup via the ModelManager singleton (lifespan).
The ``/predict`` router is mounted in Task 7.
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI

from api.core.config import settings
from api.core.model_manager import ModelManager
from api.routers import health, models, predict

logging.basicConfig(level=logging.INFO)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load models once at startup. Set SSA_SKIP_MODEL_LOAD=1 to skip (e.g. for
    # lightweight tests of the app object without the ML stack). Set
    # SSA_MODEL_IDS to a comma-separated list to load only a subset (useful to
    # avoid downloading every model in constrained environments).
    if os.getenv("SSA_SKIP_MODEL_LOAD") != "1":
        subset = os.getenv("SSA_MODEL_IDS")
        model_ids = [m.strip() for m in subset.split(",")] if subset else None
        ModelManager.instance().load(model_ids)
    yield


def create_app() -> FastAPI:
    app = FastAPI(title=settings.app_name, version="0.1.0", lifespan=lifespan)
    app.include_router(health.router)
    app.include_router(models.router)
    app.include_router(predict.router)
    return app


app = create_app()
