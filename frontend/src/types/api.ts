export interface HealthResponse {
  status: string;
  version: string;
  uptime_seconds: number;
}

export interface GPUInfo {
  index: number;
  name: string;
  total_memory_bytes: number;
  compute_capability: string;
}

export interface SystemInfo {
  app_version: string;
  python_version: string;
  platform: string;
  torch_version: string;
  cuda_available: boolean;
  cuda_version: string | null;
  device: string;
  gpus: GPUInfo[];
  cpu_count: number | null;
  runs_dir_exists: boolean;
  checkpoints_found: number;
  models_loaded: number;
  limits: { max_new_tokens: number; max_loaded_models: number };
}

export interface ModelSummary {
  model_id: string;
  run: string;
  kind: "best" | "last";
  size_bytes: number;
  modified: string;
  loaded: boolean;
}

export interface ModelListResponse {
  models: ModelSummary[];
}

export interface ModelDetail extends ModelSummary {
  model_type: string;
  iteration: number;
  parameters: number;
  vocab_size: number;
  run_best_val_loss: number | null;
  run_final_val_loss: number | null;
  run_training_time_s: number | null;
}

export interface ModelConfigResponse {
  model_id: string;
  model_type: string;
  model_kwargs: Record<string, unknown>;
  head_size: number | null;
  context_length: number | null;
}

export interface ParametersResponse {
  model_id: string;
  total: number;
  breakdown: Record<string, number> | null;
  breakdown_matches_actual: boolean | null;
  tensors: { name: string; shape: number[]; numel: number }[];
}

export interface LoadResponse {
  model_id: string;
  loaded: boolean;
  device: string;
  loaded_at: string;
  parameters: number;
}

export interface UnloadResponse {
  model_id: string;
  unloaded: boolean;
}

export interface GenerateRequest {
  model_id: string;
  prompt: string;
  max_new_tokens: number;
  temperature: number;
  top_k: number | null;
  seed: number | null;
}

export interface GenerateResponse {
  model_id: string;
  prompt: string;
  text: string;
  completion: string;
  tokens_generated: number;
  prompt_tokens: number;
  seconds: number;
  tokens_per_second: number;
  device: string;
  seed: number | null;
  seed_note: string | null;
}

export interface CompareVariant {
  label: string;
  model_id: string | null;
  temperature: number | null;
  top_k: number | null;
}

export interface CompareRequest {
  model_id: string;
  prompt: string;
  max_new_tokens: number;
  temperature: number;
  top_k: number | null;
  seed: number;
  variants: CompareVariant[];
}

export interface CompareResponse {
  seed: number;
  results: { label: string; result: GenerateResponse }[];
}

export interface StreamStart {
  model_id: string;
  device: string;
  seed: number | null;
  max_new_tokens: number;
}
export interface StreamToken {
  id: number;
  text: string;
  index: number;
}
export interface StreamDone {
  tokens_generated: number;
  seconds: number;
  tokens_per_second: number;
}