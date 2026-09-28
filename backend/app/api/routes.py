"""
API routes.

    GET    /api/health    - service/database/model status
    POST   /api/predict   - run a real prediction on an uploaded leaf image
    GET    /api/history   - list past predictions from MySQL
    DELETE /api/history   - clear prediction history in MySQL
"""

import logging

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy import delete, func, select
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session

from app.database.session import engine, get_db
from app.models.prediction import Prediction
from app.schemas.prediction_schema import (
    DeleteHistoryResponseSchema,
    HealthResponseSchema,
    HistoryResponseSchema,
    PredictResponseSchema,
)
from app.services import disease_info
from app.services.inference_service import inference_service
from app.utils.image_validation import UploadValidationError, validate_and_read_upload
from app.utils.file_utils import unique_filename

logger = logging.getLogger("crop_disease_api")

router = APIRouter(prefix="/api", tags=["crop-disease"])


def _check_database_connected() -> bool:
    try:
        with engine.connect() as conn:
            conn.execute(select(1))
        return True
    except Exception as exc:  # noqa: BLE001 - we deliberately want to catch any DB error here
        logger.warning(f"Database health check failed: {exc}")
        return False


@router.get("/health", response_model=HealthResponseSchema)
def health_check():
    db_ok = _check_database_connected()
    models_ok = inference_service.models_available()

    if db_ok and models_ok:
        status_str, message = "ok", "API, database and models are all available."
    elif not db_ok and not models_ok:
        status_str, message = "degraded", "Database is unreachable and trained models are missing."
    elif not db_ok:
        status_str, message = "degraded", "Database is unreachable. Check MySQL connection settings in .env."
    else:
        status_str, message = "degraded", "Trained model weights are missing. Train the models before predicting."

    return HealthResponseSchema(
        status=status_str,
        database_connected=db_ok,
        models_available=models_ok,
        message=message,
    )


@router.post("/predict", response_model=PredictResponseSchema)
async def predict(file: UploadFile = File(...), db: Session = Depends(get_db)):
    # 1-3: validate file type, size, and that it is a real decodable image
    try:
        validated = await validate_and_read_upload(file)
    except UploadValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    # 4-7: preprocess, run autoencoder, run ResNet18, compute probabilities
    try:
        # ml/src is put on sys.path by services/inference_service.py. This
        # import fails if torch/torchvision are not installed.
        from inference import InvalidImageError, ModelsNotAvailableError
    except ImportError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Deep learning dependencies (PyTorch/torchvision) are not installed: {exc}",
        ) from exc

    try:
        result = inference_service.predict(validated.content)
    except ModelsNotAvailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except InvalidImageError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        # e.g. torch/torchvision not installed
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    # 8: look up disease information for the predicted class
    try:
        info = disease_info.get_disease_info(result.predicted_class)
    except disease_info.DiseaseNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    top_predictions_payload = [
        {"class": p.class_name, "confidence": p.confidence} for p in result.top_predictions
    ]

    # 9: save the prediction to MySQL
    stored_filename = unique_filename(validated.safe_filename)
    try:
        record = Prediction(
            image_filename=stored_filename,
            predicted_disease=result.predicted_class,
            confidence=result.confidence,
            top_predictions=top_predictions_payload,
        )
        db.add(record)
        db.commit()
    except OperationalError as exc:
        db.rollback()
        logger.error(f"Failed to save prediction to MySQL: {exc}")
        raise HTTPException(
            status_code=503,
            detail="Prediction succeeded but could not be saved to the database. "
            "Check that MySQL is running and reachable.",
        ) from exc

    # 10: return the JSON response
    return PredictResponseSchema(
        prediction=result.predicted_class,
        confidence=result.confidence,
        top_predictions=top_predictions_payload,
        disease_info=info,
        disclaimer=disease_info.get_disclaimer(),
        original_image_base64=result.original_image_base64,
        denoised_image_base64=result.denoised_image_base64,
    )


@router.get("/history", response_model=HistoryResponseSchema)
def get_history(db: Session = Depends(get_db)):
    try:
        total = db.execute(select(func.count()).select_from(Prediction)).scalar_one()
        records = db.execute(select(Prediction).order_by(Prediction.created_at.desc())).scalars().all()
    except OperationalError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Could not read prediction history from MySQL: {exc}",
        ) from exc

    return HistoryResponseSchema(total=total, items=[r.to_dict() for r in records])


@router.delete("/history", response_model=DeleteHistoryResponseSchema)
def delete_history(db: Session = Depends(get_db)):
    try:
        count = db.execute(select(func.count()).select_from(Prediction)).scalar_one()
        db.execute(delete(Prediction))
        db.commit()
    except OperationalError as exc:
        db.rollback()
        raise HTTPException(
            status_code=503,
            detail=f"Could not delete prediction history in MySQL: {exc}",
        ) from exc

    return DeleteHistoryResponseSchema(deleted_count=count, message=f"Deleted {count} prediction record(s).")
