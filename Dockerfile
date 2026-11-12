# syntax=docker/dockerfile:1

# ---------- Stage 1: train ----------
FROM python:3.11.9-slim-bookworm AS trainer
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 PYTHONDONTWRITEBYTECODE=1
WORKDIR /build
COPY requirements-train.txt .
RUN pip install -r requirements-train.txt
COPY params.yaml .
COPY training/ training/
COPY data/raw/breast_cancer.csv data/raw/breast_cancer.csv
ARG MODEL_VERSION=dev
RUN python -m training.train \
      --data-path data/raw/breast_cancer.csv \
      --output-dir artifacts \
      --model-version ${MODEL_VERSION}

# ---------- Stage 2: serve ----------
FROM python:3.11.9-slim-bookworm
ARG MODEL_VERSION=dev
ARG GIT_SHA=unknown
LABEL org.opencontainers.image.version=${MODEL_VERSION} \
      org.opencontainers.image.revision=${GIT_SHA} \
      org.opencontainers.image.title="breast-cancer-triage-api"

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MODEL_VERSION=${MODEL_VERSION} GIT_SHA=${GIT_SHA} \
    MODEL_PATH=artifacts/model.joblib \
    DECISION_THRESHOLD=0.5 LOG_LEVEL=INFO REDIS_PORT=6379

RUN groupadd --system --gid 10001 app && \
    useradd --system --uid 10001 --gid app --no-create-home app

WORKDIR /srv
COPY requirements-serve.txt .
RUN pip install -r requirements-serve.txt
COPY --chown=app:app app/ app/
COPY --from=trainer --chown=app:app /build/artifacts/ artifacts/

USER app
EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=3s --start-period=10s --retries=3 \
  CMD ["python", "-c", "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2).status == 200 else 1)"]

ENTRYPOINT ["uvicorn", "app.main:app"]
CMD ["--host", "0.0.0.0", "--port", "8000"]