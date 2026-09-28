# Backend — Crop Disease Detection API

FastAPI service that serves real predictions from the trained Denoising
Autoencoder + ResNet18 pipeline (see `../ml/`) and stores prediction history
in MySQL.

## Setup

```bash
cd backend
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# edit .env with your MySQL credentials
```

## Database

Create the database once (MySQL must be running):

```bash
mysql -u root -p -e "CREATE DATABASE crop_disease_detection;"
python scripts/init_db.py
```

## Trained models

The API loads weights from `../ml/models/`:

- `autoencoder.pth`
- `resnet18_classifier.pth`
- `class_names.json`

These are produced by the training scripts in `../ml/src/` (see the main
project README). If they are missing, `GET /api/health` will report
`models_available: false` and `POST /api/predict` will return `503` with a
clear message — the API will never return a fabricated prediction.

## Run

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Interactive API docs: http://localhost:8000/docs

## Tests

```bash
pip install pytest
pytest tests/ -v
```

`test_health.py` and `test_validation.py` run without MySQL or trained
models. `test_history.py` exercises the MySQL-backed endpoints and is most
meaningful once MySQL is configured.

## Endpoints

| Method | Path           | Description                              |
|--------|----------------|-------------------------------------------|
| GET    | `/api/health`  | Service, database and model status        |
| POST   | `/api/predict` | Upload an image, get a real prediction    |
| GET    | `/api/history` | List stored predictions from MySQL        |
| DELETE | `/api/history` | Clear stored prediction history           |
