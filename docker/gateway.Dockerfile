FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    ADAPTOS_CONFIG_DIR=/app/configs \
    ADAPTOS_ARTIFACTS_DIR=/app/cache/artifacts \
    HF_HOME=/app/cache/huggingface

WORKDIR /app

# CPU-only PyTorch first (the default Linux wheel would pull ~3 GB of CUDA libraries).
RUN pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu

# Dependencies before code, so code edits don't invalidate this layer.
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install ".[train,router]" \
        "transformers==5.23.0" "peft==0.22.0" "sentence-transformers==6.3.0"

COPY configs ./configs

# Model weights (Hugging Face) and registry downloads (MLflow) live in /app/cache,
# a volume, so they are fetched once and survive container rebuilds.
RUN useradd --create-home --uid 1000 app && mkdir -p /app/cache && chown app:app /app/cache
USER app

EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=3s --retries=5 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=2)"

CMD ["uvicorn", "adaptos.serving.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
