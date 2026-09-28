import { useCallback, useEffect, useState } from "react";
import { Trash2, RefreshCw, AlertTriangle } from "lucide-react";
import Section from "../components/Section";
import { ApiError, clearHistory, fetchHistory } from "../api/client";
import type { HistoryItem } from "../types/api";

type Status = "loading" | "loaded" | "error";

export default function History() {
  const [status, setStatus] = useState<Status>("loading");
  const [items, setItems] = useState<HistoryItem[]>([]);
  const [errorMessage, setErrorMessage] = useState("");
  const [deleting, setDeleting] = useState(false);

  const load = useCallback(async () => {
    setStatus("loading");
    try {
      const data = await fetchHistory();
      setItems(data.items);
      setStatus("loaded");
    } catch (err) {
      setStatus("error");
      setErrorMessage(
        err instanceof ApiError
          ? err.message
          : "Could not reach the backend. Is the API running and MySQL reachable?"
      );
    }
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  const handleClear = useCallback(async () => {
    if (!window.confirm("Delete all prediction history? This cannot be undone.")) return;
    setDeleting(true);
    try {
      await clearHistory();
      await load();
    } catch (err) {
      setStatus("error");
      setErrorMessage(err instanceof ApiError ? err.message : "Failed to delete history.");
    } finally {
      setDeleting(false);
    }
  }, [load]);

  return (
    <div className="space-y-6">
      <div className="flex items-start justify-between gap-4">
        <div>
          <h1 className="font-serif text-2xl text-ink mb-2">Prediction history</h1>
          <p className="text-ink/70 text-sm max-w-xl">
            Every analysis you run is stored in MySQL. This list is read directly
            from the database — nothing here is generated locally.
          </p>
        </div>
        <div className="flex gap-2 shrink-0">
          <button
            onClick={load}
            className="inline-flex items-center gap-1.5 text-sm border border-canopy-200 px-3 py-1.5 rounded-sm hover:bg-canopy-50 text-ink/70"
          >
            <RefreshCw className="w-3.5 h-3.5" aria-hidden />
            Refresh
          </button>
          <button
            onClick={handleClear}
            disabled={deleting || items.length === 0}
            className="inline-flex items-center gap-1.5 text-sm border border-rust/30 text-rust px-3 py-1.5 rounded-sm hover:bg-rust/5 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <Trash2 className="w-3.5 h-3.5" aria-hidden />
            Clear history
          </button>
        </div>
      </div>

      <Section>
        {status === "loading" && <p className="text-sm text-ink/50 py-6 text-center">Loading history…</p>}

        {status === "error" && (
          <div className="flex items-start gap-2 bg-rust/5 border border-rust/30 text-rust text-sm rounded-sm p-3">
            <AlertTriangle className="w-4 h-4 shrink-0 mt-0.5" aria-hidden />
            <span>{errorMessage}</span>
          </div>
        )}

        {status === "loaded" && items.length === 0 && (
          <p className="text-sm text-ink/50 py-6 text-center">
            No predictions yet. Analyze a leaf image to see it appear here.
          </p>
        )}

        {status === "loaded" && items.length > 0 && (
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="text-left text-xs text-ink/50 border-b border-canopy-100">
                  <th className="py-2 pr-4 font-medium">Filename</th>
                  <th className="py-2 pr-4 font-medium">Predicted disease</th>
                  <th className="py-2 pr-4 font-medium">Confidence</th>
                  <th className="py-2 pr-4 font-medium">Date</th>
                </tr>
              </thead>
              <tbody>
                {items.map((item) => (
                  <tr key={item.id} className="border-b border-canopy-50 last:border-0">
                    <td className="py-2.5 pr-4 font-mono text-xs text-ink/70">{item.image_filename}</td>
                    <td className="py-2.5 pr-4 text-ink">{item.predicted_disease.replace(/_/g, " ")}</td>
                    <td className="py-2.5 pr-4 font-mono text-xs">{(item.confidence * 100).toFixed(1)}%</td>
                    <td className="py-2.5 pr-4 text-ink/60 text-xs">
                      {new Date(item.created_at).toLocaleString()}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Section>
    </div>
  );
}
