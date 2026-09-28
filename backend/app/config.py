"""
Application configuration.

All secrets and environment-specific values are read from environment
variables (loaded from a local `.env` file via python-dotenv during
development). Nothing here is hardcoded — see `.env.example` for the
expected variables.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

BACKEND_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = BACKEND_ROOT.parent

load_dotenv(BACKEND_ROOT / ".env")


class Settings:
    # --- Database ---
    DB_HOST: str = os.getenv("DB_HOST", "localhost")
    DB_PORT: str = os.getenv("DB_PORT", "3306")
    DB_USER: str = os.getenv("DB_USER", "root")
    DB_PASSWORD: str = os.getenv("DB_PASSWORD", "")
    DB_NAME: str = os.getenv("DB_NAME", "crop_disease_detection")

    @property
    def database_url(self) -> str:
        return (
            f"mysql+pymysql://{self.DB_USER}:{self.DB_PASSWORD}"
            f"@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"
        )

    # --- CORS ---
    CORS_ORIGINS: list = [
        origin.strip()
        for origin in os.getenv("CORS_ORIGINS", "http://localhost:5173").split(",")
        if origin.strip()
    ]

    # --- Uploads ---
    MAX_UPLOAD_MB: int = int(os.getenv("MAX_UPLOAD_MB", "10"))
    MAX_UPLOAD_BYTES: int = MAX_UPLOAD_MB * 1024 * 1024
    ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/jpg"}
    ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}

    # --- ML model paths (shared with the ml/ pipeline) ---
    ML_ROOT: Path = PROJECT_ROOT / "ml"
    AUTOENCODER_WEIGHTS_PATH: str = str(ML_ROOT / "models" / "autoencoder.pth")
    CLASSIFIER_WEIGHTS_PATH: str = str(ML_ROOT / "models" / "resnet18_classifier.pth")
    CLASS_NAMES_PATH: str = str(ML_ROOT / "models" / "class_names.json")
    DISEASE_INFO_PATH: str = str(BACKEND_ROOT / "app" / "data" / "disease_info.json")
    IMAGE_SIZE: int = 128


settings = Settings()
