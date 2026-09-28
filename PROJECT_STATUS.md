# PROJECT_STATUS

Last updated: 28 September 2026

## Headline (please read)

The full codebase is written, but **the deep learning models have NOT been
trained, and the backend and frontend have NOT been run**. The build
environment had no internet access, so `torch`, `torchvision`, `fastapi`,
`sqlalchemy`, `pymysql`, `uvicorn` and the npm packages could not be
installed. There was also no MySQL server and no real dataset.

Consequences:

- **No model weights are included.** `ml/models/` is empty.
- **No real metrics exist** (accuracy, precision, recall, F1, confusion
  matrix, autoencoder loss). `docs/results.md` says so and contains no
  invented numbers.
- Until you train the models, `POST /api/predict` returns a clear `503`
  ("models must be trained first"). It never fabricates a prediction.

Everything needed to get real results is included; the steps are below.

## Implemented features

- ML: dataset discovery, stratified leak-free split, augmentation, class
  distribution; convolutional denoising autoencoder; ResNet18 transfer
  learning; training scripts (configurable dataset path, image size, batch
  size, epochs, LR, workers, output path; CPU or CUDA); evaluation
  (accuracy, macro/weighted P/R/F1, classification report, confusion
  matrix); Experiment A vs B comparison that reports the result honestly;
  reusable inference module.
- Backend: FastAPI with `GET /api/health`, `POST /api/predict`,
  `GET /api/history`, `DELETE /api/history`; upload validation (extension,
  content-type, size, real-image decode, filename sanitising); MySQL via
  SQLAlchemy + PyMySQL + python-dotenv; `scripts/init_db.py`; CORS; disease
  information JSON for all 6 classes.
- Frontend: Dashboard, Analyze (drag-and-drop, preview, loading/error
  states, original vs denoised, top-3, disease info), History (from MySQL,
  clear), Methodology pages.
- Docs: README, methodology, architecture (Mermaid), results, limitations, viva.

## Project structure

```
backend/   app/{api,services,models,database,schemas,utils,data}, scripts/init_db.py, tests/
ml/        src/, tests/, data/{raw,demo_synthetic}, models/, outputs/, notebooks/
frontend/  src/{api,components,pages,types,styles}
docs/      methodology, architecture, results, limitations, viva
```

## Dataset setup

Download PlantVillage and arrange it under `ml/data/raw/<ClassName>/*.jpg`
with these six classes: `Tomato_Healthy`, `Tomato_Early_Blight`,
`Tomato_Late_Blight`, `Potato_Healthy`, `Potato_Early_Blight`,
`Potato_Late_Blight`. The dataset is not in the ZIP. A tiny **synthetic**
smoke-test set (not real leaf images, ~756 KB) is included in
`ml/data/demo_synthetic/`; results from it are meaningless scientifically.

## MySQL setup

```bash
mysql -u root -p -e "CREATE DATABASE crop_disease_detection;"
cd backend && cp .env.example .env    # edit credentials
python scripts/init_db.py
```

## Exact installation commands

```bash
cd backend
python -m venv venv && source venv/bin/activate     # Windows: venv\Scripts\activate
pip install -r requirements.txt
cd ../frontend && npm install
```

## Exact training commands

```bash
cd ml/src
python train_autoencoder.py --data-dir ../data/raw
python train_classifier.py --data-dir ../data/raw
python train_classifier.py --data-dir ../data/raw --use-autoencoder --autoencoder-weights ../models/autoencoder.pth
python evaluate.py --data-dir ../data/raw --weights ../models/resnet18_baseline.pth --tag baseline
python evaluate.py --data-dir ../data/raw --weights ../models/resnet18_classifier.pth --use-autoencoder --autoencoder-weights ../models/autoencoder.pth --tag denoised
python compare_models.py
```

For a quicker CPU run add e.g. `--epochs 5 --batch-size 8 --num-workers 0`
(and optionally `--freeze-backbone` for the classifier). First classifier
run needs internet once to download ImageNet weights; if that fails it
warns and trains from scratch.

## Exact backend / frontend run commands

```bash
cd backend && uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
cd frontend && npm run dev        # http://localhost:5173
```

## Model files available

None. Expected after training: `ml/models/autoencoder.pth`,
`resnet18_baseline.pth`, `resnet18_classifier.pth`, `class_names.json`.
Note: the backend serves `resnet18_classifier.pth` (the denoised-input
classifier), matching the required pipeline. `class_names.json` is written
by `train_classifier.py`.

## Tests performed

- `ml/tests/test_preprocessing.py`: **11/11 passed** (executed).
- `ml/src/generate_synthetic_demo_data.py`: executed successfully.
- Dataset helper `float_array_to_uint8` checked on a real synthetic image.

## Build checks performed

- `python -m py_compile` on every ML and backend `.py` file: passed
  (syntax only; imports of missing packages are not checked by this).
- `tsc --noEmit` on the frontend source, with stub declarations for the
  uninstalled npm packages: 0 errors. This does **not** verify types against
  the real React / react-router / lucide packages, and `vite build` was not run.
- JSON files (disease info, notebook) parse correctly; all six classes have
  disease information entries.

## NOT executed (be aware)

- `train_autoencoder.py`, `train_classifier.py`, `evaluate.py`,
  `compare_models.py`, `inference.py` (no PyTorch). They are unrun code and
  may contain bugs that only appear at runtime; one such bug (float array
  passed to `ToPILImage`) was found by review and fixed, so expect that
  others are possible. Try the synthetic set first with 1-2 epochs.
- Backend server and `backend/tests/*` (no FastAPI/SQLAlchemy/MySQL). The
  tests were written to accept 503 when models/MySQL are absent.
- Frontend `npm install`, `npm run build`, and running in a browser.
- No MySQL connection of any kind was tested.

## Known limitations

See `docs/limitations.md`. Key points: PlantVillage is controlled-background;
six classes only; predictions are not diagnoses; confidence is uncalibrated;
no authentication; uploaded images are not stored.

## Manual steps still required

1. Install dependencies (Python + npm).
2. Download the dataset into `ml/data/raw/`.
3. Create the MySQL database, fill `backend/.env`, run `init_db.py`.
4. Train, evaluate and compare (commands above); paste real numbers into
   `docs/results.md`.
5. Start backend and frontend, and fix anything that surfaces at first run.
