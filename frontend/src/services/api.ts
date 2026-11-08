import type {
  HealthResponse,
  LoadResponse,
  ModelConfigResponse,
  ModelDetail,
  ModelListResponse,
  ParametersResponse,
  SystemInfo,
  UnloadResponse,
  CompareRequest,
  CompareResponse,
  AttentionRequest, 
  AttentionResponse, 
  DecodeResponse, 
  EncodeResponse, 
  VocabularyResponse,
  ArchitectureResponse, 
  ForwardPassResponse,
} from "../types/api";

export const API_BASE = (import.meta.env.VITE_API_URL ?? "http://localhost:8000/api/v1").replace(
  /\/$/,
  "",
);

export class ApiError extends Error {
  readonly status: number;
  constructor(status: number, message: string) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

function detailMessage(body: unknown, fallback: string): string {
  if (typeof body === "object" && body !== null && "detail" in body) {
    const detail = (body as { detail: unknown }).detail;
    if (typeof detail === "string") return detail;
    if (Array.isArray(detail)) {
      return detail
        .map((d) => (typeof d === "object" && d !== null && "msg" in d ? String(d.msg) : String(d)))
        .join("; ");
    }
  }
  return fallback;
}

export async function errorFromResponse(res: Response): Promise<ApiError> {
  let body: unknown = null;
  try {
    body = await res.json();
  } catch {
    /* non-JSON error body */
  }
  return new ApiError(res.status, detailMessage(body, res.statusText || `HTTP ${res.status}`));
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${API_BASE}${path}`, {
      ...init,
      headers: { "Content-Type": "application/json", ...init?.headers },
    });
  } catch {
    throw new ApiError(0, `Cannot reach the backend at ${API_BASE}`);
  }
    if (!res.ok) throw await errorFromResponse(res);
  return (await res.json()) as T;
}

const modelPath = (id: string) => `/models/${encodeURIComponent(id)}`;
const post = (body: unknown): RequestInit => ({ method: "POST", body: JSON.stringify(body) });

export const api = {
  health: () => request<HealthResponse>("/health"),
  systemInfo: () => request<SystemInfo>("/system/info"),
  listModels: () => request<ModelListResponse>("/models"),
  modelDetail: (id: string) => request<ModelDetail>(modelPath(id)),
  modelConfig: (id: string) => request<ModelConfigResponse>(`${modelPath(id)}/config`),
  modelParameters: (id: string) => request<ParametersResponse>(`${modelPath(id)}/parameters`),
  loadModel: (id: string) => request<LoadResponse>("/models/load", post({ model_id: id })),
  unloadModel: (id: string) => request<UnloadResponse>("/models/unload", post({ model_id: id })),
  compare: (body: CompareRequest) => request<CompareResponse>("/generate/compare", post(body)),
  encode: (modelId: string, text: string) =>request<EncodeResponse>("/tokenizer/encode", post({ model_id: modelId, text })),
  decode: (modelId: string, ids: number[]) =>request<DecodeResponse>("/tokenizer/decode", post({ model_id: modelId, ids })),
  vocabulary: (modelId: string) =>request<VocabularyResponse>(`/tokenizer/vocabulary?model_id=${encodeURIComponent(modelId)}`),
  attention: (body: AttentionRequest) =>request<AttentionResponse>("/visualize/attention", post(body)),
  architecture: (modelId: string) =>request<ArchitectureResponse>(`/visualize/architecture?model_id=${encodeURIComponent(modelId)}`),
  forwardPass: (modelId: string, prompt: string) =>request<ForwardPassResponse>("/visualize/forward-pass",post({ model_id: modelId, prompt, max_tokens: 64 }),),
};