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

export interface EncodedToken {
  char: string;
  id: number;
}

export interface EncodeResponse {
  model_id: string;
  ids: number[];
  tokens: EncodedToken[];
  length: number;
  vocab_size: number;
  block_size: number | null;
}

export interface DecodeResponse {
  model_id: string;
  text: string;
}

export interface VocabEntry {
  id: number;
  char: string;
  codepoint: string;
}

export interface VocabularyResponse {
  model_id: string;
  vocab_size: number;
  entries: VocabEntry[];
}

export interface AttentionRequest {
  model_id: string;
  prompt: string;
  layer: number;
  max_tokens: number;
}

export interface AttentionResponse {
  model_id: string;
  tokens: EncodedToken[];
  layer: number;
  n_layers: number;
  n_heads: number;
  block_size: number;
  cropped: boolean;
  /** weights[head][query][key] */
  weights: number[][][];
}

export interface ArchitectureStage {
  key: string;
  label: string;
  input_shape: string;
  output_shape: string;
  parameters: number;
  details: Record<string, number | string>;
}

export interface ArchitectureResponse {
  model_id: string;
  config: Record<string, number>;
  stages: ArchitectureStage[];
  parameters_total: number;
  stage_sum_matches_total: boolean;
}

export interface TensorStat {
  name: string;
  shape: number[];
  mean: number;
  std: number;
  min: number;
  max: number;
}

export interface ForwardPassResponse {
  model_id: string;
  tokens: EncodedToken[];
  cropped: boolean;
  stages: TensorStat[];
}

export type JobStatus = "QUEUED" | "RUNNING" | "COMPLETED" | "FAILED" | "STOPPED";

export interface MetricRow {
  iter: number;
  train_loss: number;
  val_loss: number;
  elapsed_s: number;
}

export interface TrainingJob {
  job_id: string;
  run_name: string;
  model_type: string;
  model_id: string;
  status: JobStatus;
  created_at: string;
  started_at: string | null;
  finished_at: string | null;
  device: string;
  config: { model: Record<string, number>; training: Record<string, number> };
  error: string | null;
  resumed_from: string | null;
  stop_requested: boolean;
  latest: MetricRow | null;
  progress: number | null;
  result: Record<string, unknown> | null;
  kind: string; 
  detail: Record<string, unknown> | null;
}

export interface TrainingJobRequest {
  name: string;
  model_type: "gpt" | "bigram";
  n_embd: number;
  n_head: number;
  n_layer: number;
  block_size: number;
  dropout: number;
  batch_size: number;
  learning_rate: number;
  max_iters: number;
  eval_interval: number;
  eval_iters: number;
  seed: number;
}

export interface JobLogs {
  job_id: string;
  status: JobStatus;
  lines: string[];
}

export interface ExperimentVariantInfo {
  name: string;
  label: string;
  model_type: string;
  overrides: Record<string, unknown>;
  model: Record<string, unknown>;
}
export interface ExperimentInfo {
  key: string;
  title: string;
  variants: ExperimentVariantInfo[];
}
export interface ExperimentList {
  base_model: Record<string, unknown>;
  experiments: ExperimentInfo[];
}
export interface ExperimentRunRequest {
  experiments: string[] | null;
  max_iters: number;
  seed: number;
  force: boolean;
}
export interface BatchSummary {
  batch_id: string;
  max_iters: number;
  seed: number;
  variants_finished: number;
  variants_total: number;
  modified: string;
}
export interface BatchRow {
  run: string;
  label: string;
  parameters: number;
  training_time_s: number;
  final_train_loss: number;
  final_val_loss: number;
  best_val_loss: number;
  best_iter: number;
  val_minus_train: number;
}
export interface CurvePoint {
  iter: number;
  train_loss: number;
  val_loss: number;
}
export interface BatchDetail {
  batch_id: string;
  max_iters: number;
  seed: number;
  rows: BatchRow[];
  curves: Record<string, CurvePoint[]>;
  batch: Record<string, unknown> | null;
}
export interface ExperimentCompareVariant {
  variant: string;
  label: string;
  best_val_loss: Record<string, number | null>;
  parameters: Record<string, number | null>;
}
export interface ExperimentCompareResponse {
  run_ids: string[];
  variants: ExperimentCompareVariant[];
}
export interface DistributionRequest {
  model_id: string;
  prompt: string;
  temperature: number;
  top_k: number | null;
  top_n: number;
}
export interface DistributionEntry {
  id: number;
  char: string;
  probability: number;
}
export interface DistributionResponse {
  model_id: string;
  prompt_tokens: number;
  cropped: boolean;
  temperature: number;
  top_k: number | null;
  shown_mass: number;
  entries: DistributionEntry[];
}