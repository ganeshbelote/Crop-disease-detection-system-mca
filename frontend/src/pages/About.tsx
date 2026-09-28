import Section from "../components/Section";

export default function About() {
  return (
    <div className="space-y-6">
      <div>
        <h1 className="font-serif text-2xl text-ink mb-2">Methodology</h1>
        <p className="text-ink/70 text-sm max-w-2xl">
          A summary of the dataset, models, workflow and known limitations of
          this project. See the repository's <code className="font-mono text-xs bg-canopy-50 px-1 py-0.5 rounded-sm">docs/</code>{" "}
          folder for the full technical write-up.
        </p>
      </div>

      <Section title="Dataset">
        <p className="text-sm text-ink/75 leading-relaxed">
          A subset of the PlantVillage dataset covering six classes across two
          crops: Tomato (Healthy, Early Blight, Late Blight) and Potato
          (Healthy, Early Blight, Late Blight). PlantVillage images are
          captured under controlled, uniform backgrounds — real field
          photographs (variable lighting, backgrounds, occlusion, multiple
          leaves) will likely see reduced accuracy compared to the reported
          test-set metrics.
        </p>
      </Section>

      <Section title="Denoising autoencoder">
        <p className="text-sm text-ink/75 leading-relaxed">
          A convolutional encoder compresses each leaf image into a compact
          latent representation, and a mirrored decoder reconstructs it. During
          training the network is shown a noisy version of each image and
          learns to reconstruct the clean original, which encourages the
          latent space to capture leaf structure rather than sensor or
          compression noise. At inference time every uploaded image is passed
          through this autoencoder before classification.
        </p>
      </Section>

      <Section title="ResNet18 classifier">
        <p className="text-sm text-ink/75 leading-relaxed">
          A ResNet18 convolutional network, pretrained on ImageNet, with its
          final layer replaced and retrained on the six crop-disease classes
          above. Residual connections let the network train effectively at
          this depth without vanishing gradients, which is a large part of why
          ResNet architectures transfer well to new image domains with a
          comparatively small amount of labeled data.
        </p>
      </Section>

      <Section title="Evaluation and comparison experiment">
        <p className="text-sm text-ink/75 leading-relaxed">
          Two classifiers are trained and evaluated independently: one on raw
          images, one on autoencoder-denoised images. Accuracy, precision,
          recall and F1-score are computed on a held-out test split for both,
          and compared directly — see{" "}
          <code className="font-mono text-xs bg-canopy-50 px-1 py-0.5 rounded-sm">docs/results.md</code>{" "}
          for the actual numbers from this project's training run. The
          comparison is reported as-is, whichever direction it goes.
        </p>
      </Section>

      <Section title="Limitations">
        <ul className="text-sm text-ink/75 space-y-2 list-disc list-inside">
          <li>Trained on a controlled-background dataset; field performance will vary.</li>
          <li>Limited to six classes across two crops — anything else will be misclassified into the nearest known class.</li>
          <li>A model prediction is not a guaranteed correct diagnosis.</li>
          <li>Confidence scores reflect the model's certainty, not biological ground truth.</li>
        </ul>
      </Section>
    </div>
  );
}
