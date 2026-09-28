"""
Pydantic schemas used by the FastAPI routes for request/response validation
and OpenAPI documentation.
"""

from datetime import datetime
from typing import List

from pydantic import BaseModel, Field


class ClassPredictionSchema(BaseModel):
    class_: str = Field(alias="class")
    confidence: float

    model_config = {"populate_by_name": True}


class DiseaseInfoSchema(BaseModel):
    disease_name: str
    crop: str
    description: str
    symptoms: List[str]
    prevention: List[str]
    management: List[str]


class PredictResponseSchema(BaseModel):
    prediction: str
    confidence: float
    top_predictions: List[ClassPredictionSchema]
    disease_info: DiseaseInfoSchema
    disclaimer: str
    original_image_base64: str
    denoised_image_base64: str


class HistoryItemSchema(BaseModel):
    id: int
    image_filename: str
    predicted_disease: str
    confidence: float
    top_predictions: List[ClassPredictionSchema]
    created_at: datetime


class HistoryResponseSchema(BaseModel):
    total: int
    items: List[HistoryItemSchema]


class HealthResponseSchema(BaseModel):
    status: str
    database_connected: bool
    models_available: bool
    message: str


class DeleteHistoryResponseSchema(BaseModel):
    deleted_count: int
    message: str


class ErrorResponseSchema(BaseModel):
    detail: str
