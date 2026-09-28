"""
SQLAlchemy ORM model for stored predictions.

Table: predictions (in the `crop_disease_detection` MySQL database)
"""

from datetime import datetime

from sqlalchemy import JSON, DateTime, Float, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from app.database.session import Base


class Prediction(Base):
    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    image_filename: Mapped[str] = mapped_column(String(255), nullable=False)
    predicted_disease: Mapped[str] = mapped_column(String(255), nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)

    # Stored as JSON: a list of {"class": str, "confidence": float} dicts.
    # MySQL's native JSON type is used (falls back to TEXT-backed JSON on
    # backends that don't support it natively, which SQLAlchemy handles
    # transparently).
    top_predictions: Mapped[list] = mapped_column(JSON, nullable=False)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "image_filename": self.image_filename,
            "predicted_disease": self.predicted_disease,
            "confidence": self.confidence,
            "top_predictions": self.top_predictions,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }
