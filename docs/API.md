# API

Base URL `http://localhost:8000/api/v1`. Interactive docs (OpenAPI) at `/docs`.

Errors return `{"detail": "..."}` with: **404** unknown id, **409** conflict (name taken, job
still running, model limit), **422** invalid request, **429** queue full, **403** training
disabled. Model IDs look like `gpt_small:best`.

## System
| Method | Path | Notes |
|---|---|---|
| GET | `/health` | status, version, uptime |
| GET | `/system/info` | real torch/CUDA/GPU info, limits, `training_enabled` |

## Models
| Method | Path | Notes |
|---|---|---|
| GET | `/models` | checkpoint files; does not load them |
| GET | `/models/{id}` | metadata, parameters, run losses from `summary.json` |
| GET | `/models/{id}/config` | constructor arguments, head size, context length |
| GET | `/models/{id}/parameters` | total, per-component breakdown, per-tensor shapes |
| POST | `/models/load`, `/models/unload` | body `{"model_id": "..."}` |

## Generation
| Method | Path | Notes |
|---|---|---|
| POST | `/generate` | text, timing, tokens/s, device, seed |
| POST | `/generate/stream` | Server-Sent Events: `start`, `token`, `done`, `error` |
| POST | `/generate/compare` | 2 to 4 variants, one shared seed; variants may override model, temperature, top-k |
| POST | `/generate/distribution` | next-token probabilities after temperature and top-k; nothing is sampled |

Request: `model_id`, `prompt`, `max_new_tokens` (capped by `GPTLAB_MAX_NEW_TOKENS`),
`temperature` (0 < t <= 5), `top_k`, `seed`. A seed reproduces output on the same device type
only (CPU and CUDA differ). Models load on first use.

## Tokenizer
`POST /tokenizer/encode`, `POST /tokenizer/decode`, `GET /tokenizer/vocabulary?model_id=`.
Character-level: one token per character.

## Visualization (GPT checkpoints only; a bigram returns 422)
| Method | Path | Notes |
|---|---|---|
| GET | `/visualize/architecture?model_id=` | stages with shapes and parameter counts from the checkpoint |
| POST | `/visualize/attention` | real weights for one layer, all heads; capped at 128 tokens |
| POST | `/visualize/forward-pass` | per-module output shapes and statistics for a prompt |

## Training jobs
| Method | Path | Notes |
|---|---|---|
| POST | `/training/jobs` | queue a job (201) |
| GET | `/training/jobs`, `/training/jobs/{id}` | status, progress, latest metrics |
| POST | `/training/jobs/{id}/stop` | queued jobs stop at once; running jobs stop within one step |
| POST | `/training/jobs/{id}/resume` | new job continuing the run; optional `max_iters` to extend |
| GET | `/training/jobs/{id}/metrics?after_iter=` | evaluation rows |
| GET | `/training/jobs/{id}/logs?tail=` | worker output |

Statuses: `QUEUED`, `RUNNING`, `COMPLETED`, `FAILED`, `STOPPED`. There is no pause endpoint
(see ARCHITECTURE.md).

## Experiments
| Method | Path | Notes |
|---|---|---|
| GET | `/experiments` | definitions from `experiments/definitions.py` |
| POST | `/experiments/run` | queue a batch as a job (201); track via `/training/jobs/{id}` |
| GET | `/experiments/runs`, `/experiments/runs/{batch_id}` | batches; results, curves, `batch.json` |
| GET | `/experiments/compare?run_ids=a&run_ids=b` | 2 to 4 batches (for example, two seeds) |
| GET | `/experiments/export/{batch_id}` | CSV download |

## WebSocket
`/ws/training/{job_id}` sends `{"event": ..., "data": ...}`: `training.started`,
`training.metrics` (job_id, iter, train_loss, val_loss, elapsed_s, status),
`training.checkpoint_saved` (a new best validation loss), `training.completed`,
`training.failed`, `training.stopped`, `ping` every 15 s of silence, and `error` for an unknown
job (close code 4404). A disallowed origin is closed with 4403. Generation streaming uses SSE,
not a WebSocket.