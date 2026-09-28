# Crop Disease Detection and Analysis System

An MCA-level Deep Learning academic project that classifies crop leaf
diseases from a photograph using a **convolutional denoising autoencoder**
feeding into a **ResNet18** classifier, served through a **FastAPI** backend
with **MySQL** persistence, and a **React + TypeScript** frontend.

> **Honesty notice (read first):** this repository was assembled in a
> sandboxed build environment with no internet access, no PyTorch
> installation, and no MySQL server available. All code is complete and was
> syntax-checked (and, where possible without a DL stack, actually executed
> and tested — see `docs/results.md`), but **the models have not yet been
> trained and no real accuracy/precision/recall/F1 numbers exist yet.** See
> `PROJECT_STATUS.md` and `docs/results.md` for exactly what was and wasn't
> run, and the commands to generate real results yourself.

---

## 1. Project Overview

Upload a tomato or potato leaf image and receive a real (non-hardcoded)
disease prediction, a confidence score, the top-3 candidate classes,
general disease information, and a side-by-side comparison of the original
image against its autoencoder-denoised reconstruction. Every prediction is
logged to MySQL and viewable in a history page.

## 2. Problem Statement

Manual identification of crop diseases requires expert knowledge that is
not always readily accessible to farmers and students. This project
explores whether a relatively lightweight, transfer-learning-based deep
learning pipeline can classify common tomato and potato leaf diseases from
a single photograph, and whether an additional denoising preprocessing step
helps or hurts that classification.

## 3. Objectives

1. Implement and train a convolutional denoising autoencoder on leaf images.
2. Implement and fine-tune a ResNet18 classifier via transfer learning.
3. Honestly measure whether autoencoder preprocessing improves or degrades
   classification (Experiment A vs. Experiment B).
4. Serve real-time inference through a REST API backed by MySQL history.
5. Present results through a clean, usable web interface.

## 4. Features

- Real deep learning inference (no hardcoded predictions/metrics)
- Denoising autoencoder + ResNet18 pipeline
- Top-3 predictions with confidence scores
- Structured disease information (description, symptoms, prevention, management)
- Original vs. denoised image comparison
- MySQL-backed prediction history (view + clear)
- Documented, honestly-reported comparison experiment
- Full training/evaluation scripts, reproducible via CLI

## 5. Technology Stack

**Deep Learning:** Python, PyTorch, torchvision, OpenCV, NumPy, Pandas,
scikit-learn, Matplotlib, Seaborn
**Backend:** Python, FastAPI, Uvicorn, SQLAlchemy, PyMySQL, python-dotenv
**Database:** MySQL (`crop_disease_detection`)
**Frontend:** React, TypeScript, Vite, Tailwind CSS

## 6. System Architecture

See `docs/architecture.md` for full Mermaid diagrams. Summary:

```
User → React Frontend → FastAPI → Preprocessing → Denoising Autoencoder
     → ResNet18 Classifier → Prediction + Disease Info → MySQL History
```

## 7. Machine Learning Workflow

1. Discover classes and build a leak-free stratified train/val/test split
   (`ml/src/dataset.py`).
2. Train the denoising autoencoder (`ml/src/train_autoencoder.py`).
3. Train the baseline classifier on original images (`ml/src/train_classifier.py`).
4. Train a second classifier on autoencoder-denoised images
   (`ml/src/train_classifier.py --use-autoencoder`).
5. Evaluate both on the same held-out test split (`ml/src/evaluate.py`).
6. Compare them honestly (`ml/src/compare_models.py`).
7. Serve the trained autoencoder + classifier through the shared inference
   module (`ml/src/inference.py`), reused as-is by the backend.

Full detail: `docs/methodology.md`.

## 8. Autoencoder Explanation

A 3-block convolutional encoder/decoder trained to reconstruct clean leaf
images from artificially noised versions, using MSE reconstruction loss.
See `docs/methodology.md` §4 and `docs/viva.md` for the conceptual
explanation.

## 9. CNN / ResNet18 Explanation

torchvision's ResNet18, pretrained on ImageNet, with its final layer
replaced for 6-class output and fine-tuned on the project's leaf dataset.
See `docs/methodology.md` §5 and `docs/viva.md`.

## 10. Dataset

**Exact classes used** (6, across 2 crops):

| Crop | Classes |
|---|---|
| Tomato | Healthy, Early Blight, Late Blight |
| Potato | Healthy, Early Blight, Late Blight |

Source: PlantVillage (or an equivalent publicly available dataset). **The
dataset is not included in this repository** — see Dataset Setup below.
PlantVillage images use controlled, uniform backgrounds; real-world field
images will likely see reduced accuracy (see `docs/limitations.md`).

## 11. Dataset Setup

1. Download PlantVillage (search "PlantVillage dataset" — commonly mirrored
   on Kaggle) or another crop-disease leaf dataset of your choice.
2. Arrange it as:
   ```
   ml/data/raw/
   ├── Tomato_Healthy/*.jpg
   ├── Tomato_Early_Blight/*.jpg
   ├── Tomato_Late_Blight/*.jpg
   ├── Potato_Healthy/*.jpg
   ├── Potato_Early_Blight/*.jpg
   └── Potato_Late_Blight/*.jpg
   ```
   (Folder names become class names. If your dataset uses different folder
   names, either rename them or update `DEFAULT_CLASSES` in `ml/src/config.py`
   — the loader actually uses whatever folders are present.)
3. For a quick, no-download pipeline smoke test only (**not real data**),
   you can instead generate a tiny synthetic set:
   ```bash
   cd ml/src
   python generate_synthetic_demo_data.py --out ../data/demo_synthetic
   ```
   and pass `--data-dir ../data/demo_synthetic` to the training scripts.

## 12. MySQL Setup

```bash
mysql -u root -p -e "CREATE DATABASE crop_disease_detection;"
cd backend
cp .env.example .env   # then edit with your MySQL credentials
python scripts/init_db.py
```

This creates the `predictions` table (id, image_filename, predicted_disease,
confidence, top_predictions, created_at). SQLite is never used anywhere in
this project.

## 13. Environment Configuration

Copy `backend/.env.example` to `backend/.env` and set:

```
DB_HOST=localhost
DB_PORT=3306
DB_USER=root
DB_PASSWORD=your_password
DB_NAME=crop_disease_detection
CORS_ORIGINS=http://localhost:5173,http://127.0.0.1:5173
MAX_UPLOAD_MB=10
```

No credentials are hardcoded anywhere in the codebase.

## 14. Installation

```bash
git clone <this-repo>
cd crop-disease-detection-system

# ML / backend (shared Python environment recommended)
cd backend
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
cd ..

# Frontend
cd frontend
npm install
cd ..
```

## 15. Autoencoder Training

```bash
cd ml/src
python train_autoencoder.py --data-dir ../data/raw --epochs 20 --batch-size 16
```
Outputs: `ml/models/autoencoder.pth`, `ml/outputs/autoencoder_training_curve.png`,
`ml/outputs/autoencoder_examples.png`.

## 16. Classifier Training

```bash
cd ml/src
python train_classifier.py --data-dir ../data/raw --epochs 15                       # Experiment A
python train_classifier.py --data-dir ../data/raw --epochs 15 --use-autoencoder \
    --autoencoder-weights ../models/autoencoder.pth                                 # Experiment B
```
Outputs: `ml/models/resnet18_baseline.pth`, `ml/models/resnet18_classifier.pth`,
`ml/models/class_names.json`, training curve plots.

## 17. Evaluation

```bash
python evaluate.py --data-dir ../data/raw --weights ../models/resnet18_baseline.pth --tag baseline
python evaluate.py --data-dir ../data/raw --weights ../models/resnet18_classifier.pth \
    --use-autoencoder --autoencoder-weights ../models/autoencoder.pth --tag denoised
```
Outputs real accuracy/precision/recall/F1, classification report, and a
confusion matrix image for each.

## 18. Comparison Experiment

```bash
python compare_models.py
```
Loads both evaluation results and honestly reports whether denoising helped,
hurt, or made no meaningful difference — see `docs/results.md`.

## 19. Backend Setup

See `backend/README.md`. In short: `pip install -r requirements.txt`,
configure `.env`, `python scripts/init_db.py`, then run the server (§21).

## 20. Frontend Setup

See `frontend/README.md`. In short: `npm install`, optionally set
`VITE_API_BASE_URL`, then `npm run dev`.

## 21. Running the Application

```bash
# Terminal 1 — backend
cd backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Terminal 2 — frontend
cd frontend
npm run dev
```

Frontend: http://localhost:5173  •  API docs: http://localhost:8000/docs

## 22. API Documentation

| Method | Path | Description |
|---|---|---|
| GET | `/api/health` | Service, database and model availability status |
| POST | `/api/predict` | Upload an image (`file`, multipart/form-data) → real prediction |
| GET | `/api/history` | List stored predictions from MySQL |
| DELETE | `/api/history` | Clear all stored prediction history |

Interactive schema and try-it-out UI: `/docs` (Swagger) once the backend is
running. Full request/response examples are also in `backend/README.md`.

## 23. Results

**Not yet available in this delivery** — see the honesty notice at the top
of this file and `docs/results.md` for exactly what was verified (11/11
preprocessing unit tests, synthetic-data pipeline smoke test, full syntax
and type checking) versus what still requires you to run training with
PyTorch installed and real data in place.

## 24. Limitations

See `docs/limitations.md` for the full list (dataset domain gap, class
coverage, uncalibrated confidence, no authentication, etc.).

## 25. Future Scope

- Expand to more crops/diseases, and incorporate real field-condition images.
- Add confidence calibration (e.g. temperature scaling) and an
  out-of-distribution / "unknown" category.
- Add user accounts and per-user history.
- Persist uploaded images (with consent) to allow later review from history.
- Explore lightweight architectures (MobileNet, EfficientNet-lite) for
  on-device/offline mobile inference in low-connectivity farming areas.
- Active-learning loop: let agronomists confirm/correct predictions to
  build a higher-quality, field-condition training set over time.

---

## Repository layout

```
crop-disease-detection-system/
├── backend/     FastAPI app + MySQL models + tests   (see backend/README.md)
├── ml/          Datasets, models, training/eval CLIs (see docs/methodology.md)
├── frontend/    React + TS + Vite + Tailwind UI       (see frontend/README.md)
├── docs/        methodology, architecture, results, limitations, viva
├── PROJECT_STATUS.md
└── README.md    (this file)
```
