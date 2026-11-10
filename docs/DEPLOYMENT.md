# Development

## Setup
```bash
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r backend/requirements.txt
cd frontend && npm install
```
Run the backend from the project root and the frontend from `frontend/`. Copy
`frontend/.env.example` to `frontend/.env` to point at a different API.

## Layout
```
models/ training/ experiments/ utils/ config/   original PyTorch code (reused as is)
backend/        FastAPI: api/v1, schemas, services, websocket, core
frontend/src/   pages, features, hooks, services, stores, theme
app/            original Streamlit app (kept)
tests/          pytest (tests/backend for the API)
docs/ docker/ demo/ .github/
```

## Commands
| Task | Command |
|---|---|
| Python tests | `pytest` |
| Frontend tests, types, build | `cd frontend && npm test && npm run typecheck && npm run build` |
| API server | `uvicorn backend.main:app` |
| UI dev server | `cd frontend && npm run dev` |

## Conventions
- Python: type hints, small services, no hidden global state. Optional parameters must default to
  the old behavior, and old checkpoints must keep loading.
- TypeScript: strict mode. Avoid `any`.
- Never show a number the backend did not produce. If data is missing, show a dash or an honest
  empty state.
- Colors come from `frontend/src/theme/theme.ts`, never hard-coded in components.
- Tests that need local files (dataset, checkpoints) must skip when they are absent, so CI stays green.

## Gotchas
- **No `--reload` with training.** A reload can orphan a worker process.
- **One server process.** The job store assumes it.
- **Name clash:** the top-level `experiments/` package and `backend/api/v1/endpoints/experiments.py`
  share a name. Import the endpoint module with an alias.
- **Settings are read at startup.** Restart the server after changing environment variables.
- **`outputs/` is git-ignored.** The only committed checkpoint is the small one in `demo/`.
- **Shell line endings:** `docker/entrypoint.sh` must use LF (`.gitattributes` enforces it).
- **GPU tests:** exact-equality resume tests run on CPU only.