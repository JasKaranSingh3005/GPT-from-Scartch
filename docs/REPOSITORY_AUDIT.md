# Repository Audit

Baseline: the state at the end of the Streamlit/experiments work (113 passing pytest tests).
Verified against `git ls-files`; the file list matches this audit.

## 1. Existing architecture

| Layer | Modules | Notes |
|---|---|---|
| Models | `models/tokenizer.py`, `bigram.py`, `attention.py`, `transformer.py`, `gpt.py`, `sampling.py` | Pure PyTorch, no web or Streamlit imports |
| Training | `training/train.py`, `evaluate.py`, `checkpoint.py` | Library functions; `train.py` (root) is the CLI |
| Experiments | `experiments/definitions.py`, `run_experiments.py`, `plot_results.py` | 14 variants, 6 experiments |
| Config / utils | `utils/config.py`, `data.py`, `device.py`, `estimate.py`, `config/config.yaml` | Dataclass config with validation |
| UI (existing) | `app/streamlit_app.py`, `app/components/*` | 8 Streamlit pages; must be preserved |
| Tests | `tests/` | 113 tests; `pytest.ini` sets `pythonpath = .` |

## 2. Reusable interfaces (the integration contract)

| Purpose | Signature | Returns |
|---|---|---|
| Build GPT | `GPT(vocab_size, n_embd, n_head, n_layer, block_size, dropout)` / `GPT.from_config(cfg, vocab_size)` | model |
| Forward | `GPT.forward(idx, targets=None, return_weights=False)` | `(logits, loss)` or `(logits, loss, [per-layer (B, n_head, T, T)])` |
| Generate | `GPT.generate(idx, max_new_tokens, temperature, top_k)` | `(B, T+n)` tensor; crops context to `block_size`; restores train/eval mode |
| Param count | `GPT.num_parameters()`, `utils.estimate.gpt_parameter_breakdown(...)` | int / dict (verified equal by tests) |
| Bigram | `BigramLanguageModel(vocab_size)`; same forward/generate shape | no `block_size` attribute |
| Tokenizer | `CharTokenizer(chars)`, `.from_text`, `.encode`, `.decode`, `.stoi`, `.itos`, `.chars`, `.vocab_size` | raises `ValueError` on unknown chars/IDs |
| Sampling | `next_token_probs`, `sample_next_token`, `encode_prompt`, `generate_text(model, tok, prompt, n, temperature, top_k, device, seed)` | probs / token / text |
| Checkpoint I/O | `save_checkpoint(path, model, *, model_type, model_kwargs, chars, iteration, history)`, `load_checkpoint`, `load_model` | `load_model` -> `(model in eval mode, tokenizer, ckpt dict)` |
| Train loop | `train_model(model, splits, *, block_size, batch_size, learning_rate, max_iters, eval_interval, eval_iters, device, weight_decay, grad_clip, writer, on_best, on_eval)` | history list of dicts |
| Run | `run_experiment(cfg, model_type, run_name, out_root, use_tensorboard, device, on_eval)` | summary dict |
| Experiments | `VARIANTS`, `EXPERIMENTS`, `build_config(base, variant, max_iters)`, `selected_variants`, `run_all(..., run_fn=)` | configs / summaries |
| Results | `plot_results.collect_table(out_dir, names)` | DataFrame from `summary.json` |

Checkpoint format: a dict with `model_type`, `model_kwargs`, `chars`, `iteration`, `history`,
`state_dict`; loaded with `weights_only=True`. Run folders: `outputs/runs/<name>/{best.pt, last.pt,
history.csv, summary.json}`. Experiment folders: `outputs/experiments/it<N>_s<seed>/<variant>/`.

## 3. Current limitations relevant to GPT Lab

1. **No stop hook in `train_model`.** A running job cannot be interrupted cleanly.
2. **No resume.** Checkpoints contain no optimizer state, scheduler state or RNG state. There is
   no scheduler at all (constant learning rate).
3. **No pause.** Nothing supports it; it will not be exposed (see section 5).
4. **Generation is non-streaming.** `generate` returns only the final tensor.
5. **Seeding is global.** `generate_text(seed=...)` calls `torch.manual_seed`, and
   `sample_next_token` has no `generator` parameter. Concurrent requests would corrupt each
   other's reproducibility.
6. **`Head.last_weights` is mutable shared state.** The attention endpoint must use the
   `return_weights=True` path, never `last_weights`.
7. **`app/components/common.py` imports Streamlit** at module level, so the backend must not
   import `discover_checkpoints` or `RUNS_DIR` from it. `app/components/runs.py` imports from
   `common.py` for the same reason.
8. **Paths are CWD-relative** (`data/input.txt`, `outputs/runs`). The backend must resolve
   them against a project root setting, as the in-app trainer already had to.
9. **Run-name validation lives in the Streamlit layer** (`validate_run_name`), so the API
   needs its own, to prevent path traversal.
10. **No intermediate-tensor access** for the forward-pass endpoint. The model returns only
    logits and attention weights.
11. **No KV cache.** Heads are a Python loop, so a cache would be a separate, optional code path.
12. **No CI exists yet.** `.github/workflows/` was never created, so there is nothing to "extend".
13. **`outputs/` is git-ignored.** A fresh clone, including a deployed demo, has no checkpoints.
14. **Timing noise.** Recorded `training_time_s` varied up to 4x between identically sized
    models, so speed numbers in the UI need a caveat, and benchmarks must run under controlled conditions.

## 4. Proposed integration points

- **Model registry:** wraps `load_model`; model ID = `<run>:<best|last>`; lock around load/unload
  and inference; metadata cache keyed on path + mtime.
- **Inference:** new `generate_stream` generator in `models/sampling.py`. It uses the same
  `sample_next_token` and the same context cropping as `GPT.generate`, with a test proving
  identical output under the same seed. Add an optional `generator: torch.Generator | None`
  parameter, default `None`, to the sampling functions so existing behavior is unchanged.
- **Attention:** call `model(idx, return_weights=True)`. No model change needed.
- **Forward-pass shapes:** `register_forward_hook` on existing modules, no model edits.
- **Training worker:** a subprocess running a new thin module (`training/worker.py`) that calls
  the existing `run_experiment`/`train_model`. Add one optional `should_stop` callback to
  `train_model` (default `None`). Progress goes through an append-only JSON-lines metrics file;
  the API and WebSocket tail it.
- **Experiments:** call `run_all` from the worker. Do not copy it. Results via `collect_table`.
- **Storage:** keep `outputs/` as the artifact root, configurable via `GPTLAB_DATA_DIR`,
  rather than creating a parallel `storage/` tree that would duplicate existing runs. SQLite
  holds only job metadata. Metrics stay in CSV/JSONL, checkpoints on disk.

## 5. Potential breaking changes and how they are avoided

| Change | Risk | Mitigation |
|---|---|---|
| `should_stop` in `train_model` | signature | optional, default `None` |
| Optimizer/RNG state in checkpoints (needed for resume) | `_REQUIRED_KEYS`, `weights_only=True` | add as optional keys; old checkpoints still load; test both |
| `generator` parameter in sampling | signature | optional, default `None`; existing tests unchanged |
| LR scheduler | changes training results | opt-in config field, off by default |
| KV cache | changes generation path | separate method, tested for equality against baseline |

**Pause/resume decision.** Training runs in a separate process, and pausing would need
signal handling that is unreliable across platforms (this project is developed on Windows).
Proposed: expose **stop**, plus **resume from the last checkpoint** once optimizer and RNG state
are saved. Do not expose pause, and do not add a PAUSED status the pipeline cannot honor.

## 6. Recommended order

M1 foundation (registry, health/system, frontend shell) -> M2 playground (generate, stream,
tokenizer) -> M3 visualization -> M4 training (stop hook, worker, WebSocket) -> M5 experiments ->
M6 production (Docker, CI, deploy). Each milestone keeps all 113 existing tests green.

## 7. Environment findings

- `git ls-files` matches this audit. `models/__init__.py` and `training/__init__.py` were
  untracked (namespace packages); added.
- The development machine had a CPU-only PyTorch build (`2.15.0+cpu`, `cuda_available False`),
  so all recorded runs were CPU. Any GPU claim in GPT Lab must come from
  `torch.cuda.is_available()` at runtime.
- Node v22 is available.