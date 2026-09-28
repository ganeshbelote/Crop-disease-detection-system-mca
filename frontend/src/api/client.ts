import type {
  HealthResponse,
  HistoryResponse,
  PredictResponse,
} from "../types/api";

// Configurable via a Vite env variable (VITE_API_BASE_URL) so the frontend
// can point at a different backend host in production without a rebuild of
// the source. Defaults to the local FastAPI dev server.
const API_BASE_URL: string =
  (import.meta as any).env?.VITE_API_BASE_URL || "http://localhost:8000";

export class ApiError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
    this.name = "ApiError";
  }
}

async function handleResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    let detail = `Request failed with status ${response.status}`;
    try {
      const body = await response.json();
      if (body?.detail) detail = body.detail;
    } catch {
      // response body was not JSON; keep the generic message
    }
    throw new ApiError(response.status, detail);
  }
  return response.json() as Promise<T>;
}

export async function fetchHealth(): Promise<HealthResponse> {
  const response = await fetch(`${API_BASE_URL}/api/health`);
  return handleResponse<HealthResponse>(response);
}

export async function predictImage(file: File): Promise<PredictResponse> {
  const formData = new FormData();
  formData.append("file", file);

  const response = await fetch(`${API_BASE_URL}/api/predict`, {
    method: "POST",
    body: formData,
  });
  return handleResponse<PredictResponse>(response);
}

export async function fetchHistory(): Promise<HistoryResponse> {
  const response = await fetch(`${API_BASE_URL}/api/history`);
  return handleResponse<HistoryResponse>(response);
}

export async function clearHistory(): Promise<{ deleted_count: number; message: string }> {
  const response = await fetch(`${API_BASE_URL}/api/history`, { method: "DELETE" });
  return handleResponse(response);
}
