# Limitations

## Dataset limitations

- **Controlled-background bias.** PlantVillage images are photographed
  against uniform backgrounds under consistent lighting. Models trained
  purely on this data tend to partly learn background/lighting cues rather
  than purely disease-relevant features, and typically show a measurable
  accuracy drop on real field photographs (variable backgrounds, multiple
  leaves, occlusion, shadows, dust, insect damage).
- **Limited class coverage.** Only six classes across two crops are
  supported. Any other crop, disease, pest, nutrient deficiency, or a
  non-leaf image will still be forced into one of the six known classes —
  the system has no "unknown/out-of-distribution" category.
- **Class imbalance.** Real-world disease datasets are rarely perfectly
  balanced across classes; whatever imbalance exists in the source data
  used will bias the model somewhat toward the majority class(es). The
  stratified split mitigates uneven representation across train/val/test
  but does not correct dataset-level imbalance itself.

## Modeling limitations

- **A prediction is not a diagnosis.** The system reports what its trained
  model finds most statistically similar to patterns seen during training —
  it does not "know" plant pathology, has no mechanism to detect diseases it
  was never trained on, and can be confidently wrong.
- **Confidence is not calibrated probability of correctness.** A high
  softmax confidence score reflects the model's certainty relative to the
  classes it knows, not a guarantee of being right, and is not formally
  calibrated (e.g. via temperature scaling) in this project.
- **The autoencoder may help or hurt.** As required by this project's own
  brief, the comparison experiment (`docs/results.md`) reports whichever
  outcome actually occurs. It is entirely possible for the denoising step to
  reduce classification accuracy if it smooths away fine lesion texture the
  classifier relies on — this would be a legitimate, informative research
  finding, not a defect to be hidden.
- **Small-scale training.** ResNet18 at 128×128 resolution with a six-class,
  student-hardware-scale dataset will not match the accuracy of
  larger-scale, higher-resolution, ensemble, or production agricultural
  diagnostic systems trained on much larger datasets.

## Engineering limitations

- **Single-image inference only.** The system analyzes one leaf per request;
  it does not perform multi-leaf, whole-plant, or video-based assessment.
- **No authentication.** The API and history endpoints have no user
  accounts or access control — this is appropriate for a local academic
  project but would need to be added before any public/multi-tenant
  deployment.
- **No image storage.** Uploaded images are processed in memory and are not
  persisted to disk (only the filename, prediction and confidence are
  stored in MySQL) — the original photo cannot be re-viewed later from
  history alone.
- **Local development scope.** CORS, environment variables and MySQL setup
  are configured for local development; production deployment (HTTPS,
  managed MySQL, secrets management, rate limiting) is out of scope for this
  academic deliverable.

## What this project does NOT claim

- It does not claim 100% accuracy under any circumstances.
- It does not claim to replace an agricultural extension officer, plant
  pathologist, or agronomist.
- It does not claim the autoencoder improves results — see `results.md` for
  the actual, honestly reported outcome once training has been run.
