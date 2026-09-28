import { useCallback, useRef, useState } from "react";
import { UploadCloud, ImageOff, Loader2, RotateCcw, AlertTriangle } from "lucide-react";
import Section from "../components/Section";
import ConfidenceBar from "../components/ConfidenceBar";
import { ApiError, predictImage } from "../api/client";
import type { PredictResponse } from "../types/api";

type Status = "idle" | "ready" | "loading" | "error" | "done";

const ACCEPTED_TYPES = ["image/jpeg", "image/png", "image/jpg"];
const MAX_SIZE_BYTES = 10 * 1024 * 1024;

export default function Analyze() {
  const [status, setStatus] = useState<Status>("idle");
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [result, setResult] = useState<PredictResponse | null>(null);
  const [errorMessage, setErrorMessage] = useState<string>("");
  const [isDragging, setIsDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const reset = useCallback(() => {
    setStatus("idle");
    setFile(null);
    setPreviewUrl(null);
    setResult(null);
    setErrorMessage("");
  }, []);

  const handleFile = useCallback((candidate: File | undefined | null) => {
    if (!candidate) return;

    if (!ACCEPTED_TYPES.includes(candidate.type)) {
      setStatus("error");
      setErrorMessage("Unsupported file type. Please upload a JPEG or PNG image.");
      return;
    }
    if (candidate.size > MAX_SIZE_BYTES) {
      setStatus("error");
      setErrorMessage("File is too large. Maximum size is 10 MB.");
      return;
    }

    setFile(candidate);
    setPreviewUrl(URL.createObjectURL(candidate));
    setResult(null);
    setStatus("ready");
  }, []);

  const handleDrop = useCallback(
    (e: React.DragEvent<HTMLDivElement>) => {
      e.preventDefault();
      setIsDragging(false);
      handleFile(e.dataTransfer.files?.[0]);
    },
    [handleFile]
  );

  const handleAnalyze = useCallback(async () => {
    if (!file) return;
    setStatus("loading");
    setErrorMessage("");
    try {
      const response = await predictImage(file);
      setResult(response);
      setStatus("done");
    } catch (err) {
      setStatus("error");
      if (err instanceof ApiError) {
        setErrorMessage(err.message);
      } else {
        setErrorMessage("Could not reach the analysis service. Is the backend running?");
      }
    }
  }, [file]);

  return (
    <div className="space-y-8">
      <div>
        <h1 className="font-serif text-2xl text-ink mb-2">Analyze a leaf image</h1>
        <p className="text-ink/70 text-sm max-w-2xl">
          Upload a clear photograph of a single tomato or potato leaf, ideally
          against a plain background and with even lighting.
        </p>
      </div>

      <Section>
        {!previewUrl ? (
          <div
            onDragOver={(e) => {
              e.preventDefault();
              setIsDragging(true);
            }}
            onDragLeave={() => setIsDragging(false)}
            onDrop={handleDrop}
            className={`flex flex-col items-center justify-center gap-3 border-2 border-dashed rounded-sm py-14 px-6 text-center transition-colors ${
              isDragging ? "border-canopy-500 bg-canopy-50" : "border-canopy-200"
            }`}
          >
            <UploadCloud className="w-8 h-8 text-canopy-500" aria-hidden />
            <div>
              <p className="text-sm text-ink">Drag and drop a leaf image here</p>
              <p className="text-xs text-ink/50 mt-1">JPEG or PNG, up to 10 MB</p>
            </div>
            <button
              onClick={() => inputRef.current?.click()}
              className="mt-1 text-sm text-canopy-700 border border-canopy-300 px-4 py-1.5 rounded-sm hover:bg-canopy-50"
            >
              Choose a file
            </button>
            <input
              ref={inputRef}
              type="file"
              accept="image/jpeg,image/png"
              className="hidden"
              onChange={(e) => handleFile(e.target.files?.[0])}
            />
          </div>
        ) : (
          <div className="flex flex-col items-center gap-4">
            <img
              src={previewUrl}
              alt="Uploaded leaf preview"
              className="max-h-72 rounded-sm border border-canopy-200 object-contain"
            />
            <div className="flex gap-3">
              <button
                onClick={handleAnalyze}
                disabled={status === "loading"}
                className="inline-flex items-center gap-2 bg-canopy-600 text-white px-5 py-2 rounded-sm hover:bg-canopy-700 disabled:opacity-60 disabled:cursor-not-allowed text-sm font-medium"
              >
                {status === "loading" && <Loader2 className="w-4 h-4 animate-spin" aria-hidden />}
                {status === "loading" ? "Analyzing..." : "Analyze image"}
              </button>
              <button
                onClick={reset}
                className="inline-flex items-center gap-2 text-ink/70 border border-canopy-200 px-4 py-2 rounded-sm hover:bg-canopy-50 text-sm"
              >
                <RotateCcw className="w-4 h-4" aria-hidden />
                Choose another
              </button>
            </div>
          </div>
        )}

        {status === "error" && (
          <div className="mt-4 flex items-start gap-2 bg-rust/5 border border-rust/30 text-rust text-sm rounded-sm p-3">
            <AlertTriangle className="w-4 h-4 shrink-0 mt-0.5" aria-hidden />
            <span>{errorMessage}</span>
          </div>
        )}
      </Section>

      {status === "done" && result && (
        <div className="space-y-6">
          <div className="grid md:grid-cols-2 gap-6">
            <Section title="Original image">
              <img
                src={`data:image/png;base64,${result.original_image_base64}`}
                alt="Original uploaded leaf"
                className="w-full rounded-sm border border-canopy-100"
              />
            </Section>
            <Section title="Denoised (autoencoder output)">
              <img
                src={`data:image/png;base64,${result.denoised_image_base64}`}
                alt="Autoencoder-reconstructed leaf"
                className="w-full rounded-sm border border-canopy-100"
              />
            </Section>
          </div>

          <Section title="Prediction">
            <div className="mb-5">
              <p className="text-xs text-ink/50 mb-1">Predicted class</p>
              <p className="font-serif text-2xl text-ink">{result.prediction.replace(/_/g, " ")}</p>
            </div>
            <p className="text-xs text-ink/50 mb-2">Top 3 predictions</p>
            {result.top_predictions.map((p, i) => (
              <ConfidenceBar
                key={p.class}
                label={p.class.replace(/_/g, " ")}
                confidence={p.confidence}
                emphasized={i === 0}
              />
            ))}
          </Section>

          <Section title={`${result.disease_info.disease_name} — ${result.disease_info.crop}`}>
            <p className="text-sm text-ink/80 mb-5 leading-relaxed">{result.disease_info.description}</p>

            <div className="grid sm:grid-cols-3 gap-6">
              <div>
                <p className="text-xs font-medium text-canopy-700 mb-2">Symptoms</p>
                <ul className="text-sm text-ink/75 space-y-1.5 list-disc list-inside">
                  {result.disease_info.symptoms.map((s) => (
                    <li key={s}>{s}</li>
                  ))}
                </ul>
              </div>
              <div>
                <p className="text-xs font-medium text-canopy-700 mb-2">Prevention</p>
                <ul className="text-sm text-ink/75 space-y-1.5 list-disc list-inside">
                  {result.disease_info.prevention.map((s) => (
                    <li key={s}>{s}</li>
                  ))}
                </ul>
              </div>
              <div>
                <p className="text-xs font-medium text-canopy-700 mb-2">Management</p>
                <ul className="text-sm text-ink/75 space-y-1.5 list-disc list-inside">
                  {result.disease_info.management.map((s) => (
                    <li key={s}>{s}</li>
                  ))}
                </ul>
              </div>
            </div>

            <p className="text-xs text-ink/45 mt-6 pt-4 border-t border-canopy-100">
              {result.disclaimer}
            </p>
          </Section>
        </div>
      )}

      {status === "idle" && !file && (
        <div className="flex items-center gap-2 text-xs text-ink/40">
          <ImageOff className="w-3.5 h-3.5" aria-hidden />
          No image selected yet.
        </div>
      )}
    </div>
  );
}
