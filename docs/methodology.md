# Methodology

## 1. Problem framing

Given a photograph of a tomato or potato leaf, predict which of six classes
it belongs to (three per crop: Healthy, Early Blight, Late Blight), report a
confidence score, and provide general, non-prescriptive prevention and
management information for the predicted class.

## 2. Dataset

- **Source**: PlantVillage (or an equivalent publicly available crop-disease
  leaf image dataset).
- **Classes used** (6):
  - `Tomato_Healthy`, `Tomato_Early_Blight`, `Tomato_Late_Blight`
  - `Potato_Healthy`, `Potato_Early_Blight`, `Potato_Late_Blight`
- **Why this subset**: it keeps training tractable on student hardware
  (CPU-only is workable) while still requiring the model to distinguish
  between visually similar diseases (early vs. late blight) and between two
  different crops, which is a meaningfully harder task than a binary
  healthy/diseased split.
- **Known limitation**: PlantVillage images are captured against uniform,
  controlled backgrounds in near-identical lighting. A model trained purely
  on this data should be expected to perform noticeably worse on real field
  photographs, which have variable backgrounds, lighting, occlusion by other
  leaves, and often multiple leaves per frame. This project does not attempt
  to correct for that domain gap; it is documented as an explicit limitation
  (see `limitations.md`).

## 3. Preprocessing

- Images are loaded, converted to RGB, and resized to 128x128.
- Pixel values are scaled to [0, 1] before autoencoder processing, then
  normalized with ImageNet mean/std before being passed to the ResNet18
  classifier (matching the normalization ResNet18 was pretrained with).
- **Split strategy**: a stratified 70/15/15 train/validation/test split is
  performed independently per class, directly on the list of file paths,
  *before* any Dataset object or augmentation is created. This guarantees no
  image file can appear in more than one split (no data leakage), and that
  small classes are still represented in every split.
- **Augmentation** (train split only): random horizontal flip, random
  rotation (±15°), and mild brightness/contrast/saturation jitter. Vertical
  flips and heavy hue shifts are deliberately avoided, since they would
  produce leaf orientations and colorations that would not occur in real
  photographs and could teach the model to rely on unrealistic cues.
- **Reproducibility**: a fixed random seed (42, configurable) is used for
  the split, weight initialization, and all `numpy`/`torch` random state.

## 4. Denoising convolutional autoencoder

- **Architecture**: 3 convolutional encoder blocks (32 → 64 → 128 channels,
  each followed by batch norm, ReLU and max-pooling) producing a 16×16×128
  latent representation, mirrored by 3 transposed-convolution decoder
  blocks back to a 128×128×3 reconstruction with a final sigmoid activation.
- **Training objective**: Gaussian noise (configurable standard deviation,
  default 0.15 on a [0,1] pixel scale) is added to each clean image; the
  network is trained to reconstruct the *clean* image from the *noisy* one,
  using Mean Squared Error (MSE) reconstruction loss. MSE is the standard
  choice for a real-valued, pixel-wise reconstruction objective and
  penalizes large deviations more heavily than L1, which is appropriate
  since large reconstruction errors correspond to structurally wrong
  reconstructions we most want to avoid.
- **Training controls**: configurable epochs, batch size, learning rate
  (Adam optimizer), and noise level. The best checkpoint (lowest validation
  loss) is saved automatically.

## 5. ResNet18 classifier (transfer learning)

- torchvision's ResNet18, pretrained on ImageNet, with the final fully
  connected layer replaced by a new linear layer sized to 6 classes.
- Cross-entropy loss, Adam optimizer, configurable learning rate/epochs.
  The best checkpoint (highest validation accuracy) is saved.
- Optionally, the convolutional backbone can be frozen
  (`--freeze-backbone`) to train faster on limited/CPU hardware, at some
  cost to final accuracy.

## 6. Comparison experiment

Two independently trained classifiers are evaluated on the *same* held-out
test split:

- **Experiment A**: original image → ResNet18
- **Experiment B**: original image → autoencoder → denoised image → ResNet18

`ml/src/evaluate.py` computes real accuracy/precision/recall/F1 and a
confusion matrix for each. `ml/src/compare_models.py` then loads both sets
of already-computed metrics and reports the difference honestly — it does
not recompute or adjust anything, so the comparison can never diverge from,
or improve on, the individually reported numbers. See `results.md` for the
actual outcome of this project's training run.

## 7. Inference pipeline

`ml/src/inference.py`'s `CropDiseasePredictor` is the single source of
truth for turning an uploaded image into a prediction: load → resize →
autoencoder → ResNet18 → softmax → top-3. It is imported directly by the
FastAPI backend (not reimplemented), and raises clear, typed errors
(`ModelsNotAvailableError`, `InvalidImageError`) instead of ever returning a
fabricated result.
