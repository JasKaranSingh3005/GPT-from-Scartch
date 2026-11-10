FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# CPU-only PyTorch first (much smaller than the default CUDA wheel), then the rest.
COPY requirements-api.txt .
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install -r requirements-api.txt

COPY backend ./backend
COPY models ./models
COPY training ./training
COPY experiments ./experiments
COPY utils ./utils
COPY config ./config
COPY demo ./demo
COPY docker/entrypoint.sh /entrypoint.sh

RUN useradd --system --uid 10001 --create-home gptlab \
 && mkdir -p /data /app/data \
 && chown -R gptlab /data /app/data \
 && chmod +x /entrypoint.sh
USER gptlab

ENV GPTLAB_DATA_DIR=/data \
    GPTLAB_DEVICE=cpu

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/v1/health', timeout=4)" || exit 1

ENTRYPOINT ["/entrypoint.sh"]
# One worker only: the job store assumes a single server process. No --reload.
CMD ["sh", "-c", "exec uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8000}"]