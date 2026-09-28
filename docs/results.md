# Results

## Status of this document: metrics not yet generated

**Read this before anything else.** The code in this repository was written
and packaged in a sandboxed environment with **no internet access and no
PyTorch/torchvision installation available**, and **no real PlantVillage
dataset present**. Because of this, **the autoencoder and classifier have
not actually been trained in that environment, and no real accuracy,
precision, recall, F1, or confusion matrix numbers currently exist.**

This is stated here explicitly, and deliberately, because the project's own
requirements say: *"If training has not been performed, clearly indicate
that metrics are unavailable"* and *"NEVER invent metrics."* Any numbers
that are not the direct output of a completed training/evaluation run in
**your** environment would be fabricated, so none are shown below.

## What to do to generate real results

1. Install dependencies: `pip install -r backend/requirements.txt` (this
   includes `torch`/`torchvision`) or a dedicated `ml/requirements.txt`
   subset if you prefer a lighter environment for training only.
2. Download PlantVillage (or an equivalent dataset) and place it under
   `ml/data/raw/<ClassName>/*.jpg` as documented in `README.md` → *Dataset
   Setup*. Alternatively, for a quick no-download smoke test of the code
   path (not meaningful for real conclusions), run:
   ```bash
   cd ml/src
   python generate_synthetic_demo_data.py --out ../data/demo_synthetic
   ```
3. Train and evaluate both experiments:
   ```bash
   cd ml/src
   python train_autoencoder.py --data-dir ../data/raw
   python train_classifier.py --data-dir ../data/raw                                    # Experiment A (baseline)
   python train_classifier.py --data-dir ../data/raw --use-autoencoder \
       --autoencoder-weights ../models/autoencoder.pth                                  # Experiment B (denoised)
   python evaluate.py --data-dir ../data/raw --weights ../models/resnet18_baseline.pth --tag baseline
   python evaluate.py --data-dir ../data/raw --weights ../models/resnet18_classifier.pth \
       --use-autoencoder --autoencoder-weights ../models/autoencoder.pth --tag denoised
   python compare_models.py
   ```
4. This will populate `ml/outputs/` with:
   - `autoencoder_training_curve.png`, `autoencoder_examples.png`
   - `classifier_training_curve_baseline.png`, `classifier_training_curve_denoised.png`
   - `evaluation_baseline.json`, `evaluation_denoised.json`
   - `confusion_matrix_baseline.png`, `confusion_matrix_denoised.png`
   - `comparison_table.csv`, `comparison_chart.png`, `comparison_report.json`,
     `comparison_summary.md`
5. Copy the real numbers from those files into the tables below.

## Pipeline correctness: what WAS verified in this environment

Even without a deep learning stack installed, the parts of the pipeline that
don't require `torch` were run and passed, to give confidence the rest of
the code is correct rather than untested:

- `ml/tests/test_preprocessing.py`: **11/11 tests passed** — covering class
  discovery, file indexing, class distribution, stratified split (including
  an explicit no-data-leakage check and a reproducibility check), and image
  loading/resizing.
- `ml/src/generate_synthetic_demo_data.py` was run successfully and produced
  a small synthetic image set, confirming the data-loading code operates
  correctly end to end on a real directory of images.
- Every `.py` file in `ml/` and `backend/` passed `python -m py_compile`
  (syntax validation).
- The frontend's TypeScript source (`frontend/src/**/*.tsx`) was type-checked
  with `tsc --noEmit` (against stub type declarations for the missing
  offline `npm` packages) with zero errors.

What was **not** run in this environment: `train_autoencoder.py`,
`train_classifier.py`, `evaluate.py`, `compare_models.py`, the FastAPI
server, and the frontend dev/build server — all because the required
packages (`torch`, `torchvision`, `fastapi`, `sqlalchemy`, `pymysql`,
`uvicorn`, and the npm dependencies in `frontend/package.json`) could not be
installed without internet access, and no MySQL server was available.

## Autoencoder — training/validation loss

| Epoch | Train loss (MSE) | Val loss (MSE) |
|---|---|---|
| _Run `train_autoencoder.py` and paste values here_ | | |

Best validation loss: _(not yet available)_

## Classifier — Experiment A (original images)

| Metric | Value |
|---|---|
| Test accuracy | _(not yet available)_ |
| Precision (macro) | _(not yet available)_ |
| Recall (macro) | _(not yet available)_ |
| F1-score (macro) | _(not yet available)_ |

Full classification report and confusion matrix: see
`ml/outputs/evaluation_baseline.json` and `ml/outputs/confusion_matrix_baseline.png`
after running evaluation.

## Classifier — Experiment B (autoencoder-denoised images)

| Metric | Value |
|---|---|
| Test accuracy | _(not yet available)_ |
| Precision (macro) | _(not yet available)_ |
| Recall (macro) | _(not yet available)_ |
| F1-score (macro) | _(not yet available)_ |

Full classification report and confusion matrix: see
`ml/outputs/evaluation_denoised.json` and `ml/outputs/confusion_matrix_denoised.png`
after running evaluation.

## Comparison: did the autoencoder help?

_(not yet available — run `compare_models.py` after both evaluations above,
then paste the contents of `ml/outputs/comparison_summary.md` here. Report
the outcome honestly whichever direction it goes, per the project's own
requirements — do not adjust or omit an unfavorable result.)_
