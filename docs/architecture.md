# Architecture

## System overview

```mermaid
flowchart TD
    A[User] --> B[React Frontend<br/>Vite + TypeScript + Tailwind]
    B -->|HTTP / multipart upload| C[FastAPI REST API]
    C --> D[Image Validation<br/>type, size, decodability]
    D --> E[Preprocessing<br/>resize, normalize]
    E --> F[Denoising Convolutional Autoencoder]
    F --> G[ResNet18 Classifier]
    G --> H[Prediction + Confidence + Top-3]
    H --> I[Disease Information Lookup]
    I --> J[MySQL: predictions table]
    J --> C
    C -->|JSON response| B
```

## Component responsibilities

- **React Frontend** (`frontend/`): upload UI, result display, history view,
  methodology page. Talks to the backend only over HTTP/JSON.
- **FastAPI backend** (`backend/`): request validation, orchestration of the
  ML inference module, MySQL persistence, disease information lookup.
  Contains no model-training or model-definition code itself — it imports
  the shared `ml/src` package.
- **ML pipeline** (`ml/`): dataset handling, model definitions, training
  scripts, evaluation, and the `inference.py` module reused by the backend.
  Fully independent of the web stack; can be run and tested from the command
  line with no server involved.
- **MySQL** (`crop_disease_detection` database): durable storage for
  prediction history (`predictions` table).

## ML training pipeline

```mermaid
flowchart LR
    RAW[ml/data/raw/&lt;ClassName&gt;/*.jpg] --> IDX[Build file index<br/>+ stratified split]
    IDX --> AETRAIN[train_autoencoder.py]
    AETRAIN --> AEW[ml/models/autoencoder.pth]

    IDX --> CLFA[train_classifier.py<br/>Experiment A: original images]
    CLFA --> CLFAW[ml/models/resnet18_baseline.pth]

    AEW --> CLFB[train_classifier.py --use-autoencoder<br/>Experiment B: denoised images]
    IDX --> CLFB
    CLFB --> CLFBW[ml/models/resnet18_classifier.pth]

    CLFAW --> EVALA[evaluate.py]
    CLFBW --> EVALB[evaluate.py --use-autoencoder]
    EVALA --> COMPARE[compare_models.py]
    EVALB --> COMPARE
    COMPARE --> REPORT[ml/outputs/comparison_report.json<br/>+ comparison_summary.md]
```

## Request lifecycle for POST /api/predict

```mermaid
sequenceDiagram
    participant U as User (browser)
    participant F as React Frontend
    participant A as FastAPI
    participant P as CropDiseasePredictor (ml/src)
    participant DB as MySQL

    U->>F: Select/drop leaf image
    F->>A: POST /api/predict (multipart file)
    A->>A: Validate type, size, decodability
    A->>P: predict(image_bytes)
    P->>P: Resize + normalize
    P->>P: Autoencoder forward pass (denoise)
    P->>P: ResNet18 forward pass (classify)
    P-->>A: prediction, confidence, top-3, images
    A->>DB: INSERT INTO predictions
    A-->>F: JSON response
    F-->>U: Render result, images, disease info
```

## Directory structure

```
crop-disease-detection-system/
├── backend/        FastAPI app, MySQL models, tests
├── ml/             Dataset, autoencoder, classifier, training/eval scripts
├── frontend/       React + TypeScript + Vite + Tailwind UI
├── docs/           This documentation
├── README.md
└── PROJECT_STATUS.md
```
