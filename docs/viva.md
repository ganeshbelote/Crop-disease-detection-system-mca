# Viva Questions and Answers

## Deep Learning fundamentals

**Q: What is Deep Learning?**
A subfield of machine learning using multi-layered neural networks to
automatically learn hierarchical representations from raw data (e.g. pixels),
rather than relying on hand-engineered features.

**Q: What is a CNN (Convolutional Neural Network)?**
A neural network architecture that applies learnable convolutional filters
across an image, exploiting spatial locality and translation invariance to
detect features like edges, textures and shapes at increasing levels of
abstraction through its layers.

**Q: Why use a CNN for image classification instead of a plain fully
connected network?**
A fully connected network on raw pixels would need an impractically large
number of parameters and ignores spatial structure. Convolutional filters
share weights across the image, drastically reducing parameters, and are
naturally suited to detecting local patterns (like a lesion) regardless of
where they appear in the image.

## Autoencoder

**Q: What is an Autoencoder?**
An unsupervised neural network trained to reconstruct its own input. It
consists of an encoder that compresses the input into a lower-dimensional
latent representation, and a decoder that reconstructs the input from that
representation.

**Q: What is a Denoising Autoencoder, specifically?**
A variant where the network is given a corrupted (noisy) version of the
input but trained to reconstruct the original clean version. This forces
the latent representation to capture meaningful structure rather than
noise, since noise alone gives no useful information for reconstruction.

**Q: Why use an Autoencoder in this project?**
To explore whether denoising a leaf image before classification improves or
harms downstream ResNet18 performance — this project implements and reports
that comparison directly (see Experiment A vs. B in `results.md`), rather
than assuming a benefit.

**Q: What is the difference between the encoder and the decoder?**
The encoder maps a high-dimensional input to a compact latent
representation (downsampling); the decoder maps that latent representation
back to the original input's dimensionality (upsampling), typically via
transposed convolutions.

**Q: What is the "latent representation" or "latent space"?**
The compressed, lower-dimensional output of the encoder — in this project a
16×16×128 tensor — that ideally captures the essential structure of the
input needed to reconstruct it.

**Q: What is reconstruction loss?**
A measure of how different the network's reconstruction is from the target
(here, the clean image), used as the training signal.

**Q: Why use MSE (Mean Squared Error) as the reconstruction loss here?**
MSE is the standard choice for real-valued, pixel-wise image reconstruction.
It penalizes larger errors quadratically, pushing the network to avoid
large structural mistakes, and its gradient is simple and well-behaved for
training.

## Transfer learning and ResNet

**Q: What is transfer learning?**
Reusing a model (or its learned weights) trained on one task/dataset as a
starting point for a different but related task, rather than training from
randomly initialized weights. It is especially useful when the target
dataset is comparatively small.

**Q: Why ResNet18 specifically?**
It offers a strong accuracy-to-compute tradeoff, is well supported by
pretrained ImageNet weights in torchvision, and — being one of the smaller
ResNet variants — is realistic to fine-tune on student-grade (including
CPU-only) hardware, which matches this project's academic scope.

**Q: What is residual learning / what problem does it solve?**
Very deep plain networks suffer from vanishing/exploding gradients and
degradation (accuracy plateauing or worsening as depth increases). Residual
("skip") connections let a block learn a residual function relative to its
input rather than the full mapping, which keeps gradients flowing more
directly to earlier layers and allows much deeper networks to train
effectively.

## Training and evaluation concepts

**Q: What is overfitting?**
When a model learns patterns specific to the training data (including
noise) rather than the underlying generalizable relationship, resulting in
strong training performance but weaker performance on unseen data.

**Q: Why use a train/validation/test split rather than training on
everything?**
The training set is used to fit the model, the validation set to tune
hyperparameters and select the best checkpoint without touching the final
test data, and the test set to give an unbiased estimate of real-world
performance. Without this separation, reported performance would be
optimistic and not representative of unseen data.

**Q: What is data augmentation, and why was it limited here?**
Artificially expanding the effective training set by applying realistic
transformations (flips, rotation, color jitter) to existing images. It was
limited to biologically plausible transformations in this project (no
vertical flips, no extreme hue shifts) since unrealistic augmentation could
teach the model to rely on cues that would never appear in a real leaf
photograph.

**Q: What is Softmax?**
A function that converts a vector of raw class scores (logits) into a
probability distribution over classes — non-negative values that sum to 1 —
used here to obtain the confidence score and top-3 predictions.

**Q: What is Cross Entropy Loss?**
A loss function commonly used for classification that measures the
difference between the predicted probability distribution (from softmax)
and the true one-hot class label; it heavily penalizes confident wrong
predictions.

**Q: What is Precision?**
Of all instances the model predicted as a given class, the proportion that
were actually that class (TP / (TP + FP)).

**Q: What is Recall?**
Of all instances that actually belong to a given class, the proportion the
model correctly identified (TP / (TP + FN)).

**Q: What is F1-score?**
The harmonic mean of precision and recall, useful as a single balanced
metric when both false positives and false negatives matter and classes may
be imbalanced.

**Q: What is a Confusion Matrix?**
A table showing the counts of predicted vs. actual classes, letting you see
not just overall accuracy but exactly which classes are being confused with
each other (e.g. Early Blight predicted as Late Blight).

## Dataset

**Q: Why PlantVillage?**
It's a widely used, publicly available, labeled dataset of crop leaf images
covering many crop/disease combinations, making it a practical and
well-precedented choice for an academic crop-disease classification
project.

**Q: What are PlantVillage's limitations?**
Its images are captured under controlled, uniform backgrounds and lighting,
which does not reflect the variability of real field photographs — models
trained on it alone typically generalize less well to real-world images.

**Q: Why might the Autoencoder improve classification?**
If real sensor/compression noise in the images was hurting the classifier,
denoising could remove that noise and let the classifier focus on the
actual disease-relevant texture and color patterns, potentially also acting
as a form of regularization.

**Q: Why might the Autoencoder reduce classification performance?**
The reconstruction process can smooth over fine-grained details (like small
lesion textures or subtle color variations) that the classifier actually
relies on to distinguish visually similar diseases, effectively discarding
useful signal along with the noise.

## Backend / systems

**Q: What is FastAPI?**
A modern Python web framework for building APIs, built on top of Starlette
and Pydantic, offering automatic request/response validation and
interactive OpenAPI documentation with relatively little boilerplate.

**Q: Why MySQL rather than SQLite for this project?**
MySQL is a production-grade relational database supporting concurrent
connections and closer to what a real deployed system would use; the
project's requirements specifically call for MySQL to reflect that.

**Q: Why SQLAlchemy?**
It provides an ORM layer that lets the project define Python classes
(`Prediction`) mapped to database tables, handle sessions/transactions
safely, and remain reasonably portable across SQL backends without writing
raw SQL throughout the codebase.

**Q: How does the frontend communicate with the backend?**
The React frontend makes HTTP requests (JSON and multipart form-data) to
the FastAPI backend's REST endpoints (`/api/health`, `/api/predict`,
`/api/history`) using the browser's `fetch` API, with CORS configured on
the backend to allow the frontend's origin.

**Q: Describe the end-to-end flow when a user analyzes an image.**
The user uploads an image in the React UI, which sends it as multipart
form-data to `POST /api/predict`. FastAPI validates the file, passes its
bytes to the shared `CropDiseasePredictor` (in `ml/src/inference.py`), which
resizes/normalizes the image, runs it through the trained denoising
autoencoder, then through the trained ResNet18 classifier, computes softmax
probabilities and the top-3 classes, looks up structured disease
information for the top prediction, saves a record to MySQL, and returns a
JSON response that the frontend renders — including both the original and
denoised images for comparison.
