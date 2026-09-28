export interface ClassPrediction {
  class: string;
  confidence: number;
}

export interface DiseaseInfo {
  disease_name: string;
  crop: string;
  description: string;
  symptoms: string[];
  prevention: string[];
  management: string[];
}

export interface PredictResponse {
  prediction: string;
  confidence: number;
  top_predictions: ClassPrediction[];
  disease_info: DiseaseInfo;
  disclaimer: string;
  original_image_base64: string;
  denoised_image_base64: string;
}

export interface HistoryItem {
  id: number;
  image_filename: string;
  predicted_disease: string;
  confidence: number;
  top_predictions: ClassPrediction[];
  created_at: string;
}

export interface HistoryResponse {
  total: number;
  items: HistoryItem[];
}

export interface HealthResponse {
  status: "ok" | "degraded";
  database_connected: boolean;
  models_available: boolean;
  message: string;
}

export interface ApiErrorPayload {
  detail: string;
}
