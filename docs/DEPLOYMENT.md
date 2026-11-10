# Deployment

Nothing here has been verified on a live host yet. Treat it as a checklist to test.

## Local with Docker

```bash
docker compose up --build
```

Frontend http://localhost:8080, API http://localhost:8000, docs http://localhost:8000/docs.
`./outputs` is mounted as the data directory, so existing checkpoints and experiment batches
are visible. The image is CPU-only; for GPU training run the backend directly on the host.

## Environment variables (backend)

| Variable | Default | Meaning |
|---|---|---|
| `GPTLAB_DATA_DIR` | `<project>/outputs` (`/data` in Docker) | Holds `runs/`, `experiments/`, `jobs/`, `jobs.sqlite3` |
| `GPTLAB_DEVICE` | `auto` | `auto`, `cpu` or `cuda` (fails at startup if CUDA is unusable) |
| `GPTLAB_CORS_ORIGINS` | localhost:5173 | Comma-separated allowed origins, with scheme, no trailing slash. Also checked for the training WebSocket. Setting it replaces the default list. |
| `GPTLAB_TRAINING_ENABLED` | `true` | `false` makes training and experiment endpoints return 403 |
| `GPTLAB_MAX_NEW_TOKENS` | `1000` | Per-request generation cap |
| `GPTLAB_MAX_LOADED` | `3` | Models kept in memory |
| `GPTLAB_MAX_QUEUED` | `4` | Queued training jobs |
| `GPTLAB_MAX_TRAIN_ITERS` | `20000` | Cap on `max_iters` |

Frontend: `VITE_API_URL` (build time), for example `https://your-api.onrender.com/api/v1`.

## Public demo (suggested settings)

`GPTLAB_DEVICE=cpu`, `GPTLAB_TRAINING_ENABLED=false`, `GPTLAB_MAX_NEW_TOKENS=300`,
`GPTLAB_MAX_LOADED=1`. Users cannot upload models or start training.

## Backend on Render

1. New Web Service, runtime Docker, from this repository. Dockerfile path
   `backend/Dockerfile`, build context `.` (repository root).
2. Health check path `/api/v1/health`.
3. Set the variables above, including `GPTLAB_CORS_ORIGINS` with the exact frontend origin.
4. Deploy. Check `https://<service>.onrender.com/docs`.

## Frontend on Vercel

1. Import the repository, set the root directory to `frontend`.
2. Build command `npm run build`, output directory `dist`.
3. Environment variable `VITE_API_URL=https://<service>.onrender.com/api/v1`, then redeploy
   (it is baked in at build time).
4. Add the Vercel domain to the backend's `GPTLAB_CORS_ORIGINS`. Preview deployments get new
   URLs, so each one must be added by hand.

## Limitations

- **Memory.** CPU PyTorch plus the app uses a few hundred MB. A 512 MB free instance may not
  fit. This has not been measured.
- **Cold starts.** Free instances sleep when idle, so the first request after a pause is slow.
- **Ephemeral disk.** Without a persistent disk, anything written to `/data` is lost on
  redeploy. The demo checkpoint is re-seeded from the image on every start.
- **Single process.** Run exactly one backend worker; training jobs are tracked in SQLite and
  a local process table.
- **No GPU** in these images.
- **Seeded generation** reproduces only on the same device type (CPU vs CUDA differ).