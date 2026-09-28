import { Link } from "react-router-dom";
import { ArrowRight, ScanLine, Sparkles, Database } from "lucide-react";
import Section from "../components/Section";

const SUPPORTED_CLASSES = [
  { crop: "Tomato", condition: "Healthy" },
  { crop: "Tomato", condition: "Early Blight" },
  { crop: "Tomato", condition: "Late Blight" },
  { crop: "Potato", condition: "Healthy" },
  { crop: "Potato", condition: "Early Blight" },
  { crop: "Potato", condition: "Late Blight" },
];

const WORKFLOW_STEPS = [
  { title: "Upload a leaf image", detail: "JPEG or PNG, photographed against a reasonably plain background." },
  { title: "Denoising autoencoder", detail: "A trained convolutional autoencoder reconstructs a cleaner version of the image." },
  { title: "ResNet18 classification", detail: "The denoised image is classified into one of the supported crop/disease categories." },
  { title: "Result & guidance", detail: "You receive the predicted disease, a confidence score, and general prevention/management notes." },
];

export default function Dashboard() {
  return (
    <div className="space-y-10">
      <div>
        <p className="font-mono text-xs text-canopy-700 mb-2">Academic Deep Learning Project — MCA</p>
        <h1 className="font-serif text-3xl md:text-4xl text-ink leading-tight mb-4">
          Crop Disease Detection and Analysis System
        </h1>
        <p className="text-ink/75 leading-relaxed max-w-2xl">
          Upload a photograph of a tomato or potato leaf. The system runs it through a
          convolutional denoising autoencoder and a ResNet18 classifier to identify
          likely disease, and returns a confidence score, the top three candidate
          classes, and general prevention and management guidance.
        </p>
        <Link
          to="/analyze"
          className="inline-flex items-center gap-2 mt-6 bg-canopy-600 text-white px-5 py-2.5 rounded-sm hover:bg-canopy-700 transition-colors text-sm font-medium"
        >
          Analyze a leaf image
          <ArrowRight className="w-4 h-4" aria-hidden />
        </Link>
      </div>

      <div className="grid md:grid-cols-2 gap-6">
        <Section title="Supported crops and classes">
          <ul className="divide-y divide-canopy-100">
            {SUPPORTED_CLASSES.map((item) => (
              <li key={`${item.crop}-${item.condition}`} className="flex justify-between py-2 text-sm">
                <span className="text-ink/80">{item.crop}</span>
                <span className="text-ink font-medium">{item.condition}</span>
              </li>
            ))}
          </ul>
          <p className="text-xs text-ink/50 mt-3">
            Trained on a subset of the PlantVillage dataset. See Methodology for details
            and limitations.
          </p>
        </Section>

        <Section title="How it works">
          <ol className="space-y-4">
            {WORKFLOW_STEPS.map((step, i) => (
              <li key={step.title} className="flex gap-3">
                <span className="font-mono text-xs text-canopy-600 w-4 pt-0.5">{i + 1}</span>
                <div>
                  <p className="text-sm font-medium text-ink">{step.title}</p>
                  <p className="text-sm text-ink/60">{step.detail}</p>
                </div>
              </li>
            ))}
          </ol>
        </Section>
      </div>

      <div className="grid sm:grid-cols-3 gap-4">
        <div className="flex items-start gap-3 p-4 border border-canopy-200 rounded-sm bg-white">
          <ScanLine className="w-5 h-5 text-canopy-600 shrink-0 mt-0.5" aria-hidden />
          <div>
            <p className="text-sm font-medium text-ink">Real inference</p>
            <p className="text-xs text-ink/60">Every prediction runs the actual trained models — nothing here is hardcoded.</p>
          </div>
        </div>
        <div className="flex items-start gap-3 p-4 border border-canopy-200 rounded-sm bg-white">
          <Sparkles className="w-5 h-5 text-canopy-600 shrink-0 mt-0.5" aria-hidden />
          <div>
            <p className="text-sm font-medium text-ink">Denoising step</p>
            <p className="text-xs text-ink/60">You can compare the original and autoencoder-reconstructed image side by side.</p>
          </div>
        </div>
        <div className="flex items-start gap-3 p-4 border border-canopy-200 rounded-sm bg-white">
          <Database className="w-5 h-5 text-canopy-600 shrink-0 mt-0.5" aria-hidden />
          <div>
            <p className="text-sm font-medium text-ink">Stored history</p>
            <p className="text-xs text-ink/60">Every analyzed image is logged to MySQL and viewable on the History page.</p>
          </div>
        </div>
      </div>
    </div>
  );
}
