"""
FastAPI application entrypoint.

Run with:
    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

(from inside the backend/ directory, with the virtual environment active).
"""

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.config import settings
from app.database.session import init_db

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("crop_disease_api")

app = FastAPI(
    title="Crop Disease Detection and Analysis System",
    description=(
        "Upload a crop/plant leaf image to receive a real deep-learning-based "
        "disease classification (Denoising Autoencoder + ResNet18), confidence "
        "score, top-3 predictions and general disease information. "
        "This is an academic project and is NOT a substitute for professional "
        "agricultural diagnosis."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.on_event("startup")
def on_startup():
    try:
        init_db()
        logger.info("Database tables verified/created successfully.")
    except Exception as exc:  # noqa: BLE001
        # We do not crash the app on startup if MySQL is not reachable yet —
        # /api/health will report the degraded status, and /api/predict will
        # return a clear 503 when it tries to save a record. This makes the
        # API usable for local frontend development even before MySQL is
        # fully configured.
        logger.warning(
            f"Could not initialize the database on startup: {exc}. "
            "The API will still start, but database-backed endpoints will "
            "fail until MySQL is reachable. Run `python scripts/init_db.py` "
            "manually, or check your .env configuration."
        )


@app.get("/")
def root():
    return {
        "name": "Crop Disease Detection and Analysis System API",
        "docs": "/docs",
        "health": "/api/health",
    }
