FROM python:3.12-slim

# git is needed because CodeAtlas clones the repositories it analyses
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir .

# trained risk model (must exist in the repo: models/risk_model.joblib)
COPY models ./models

ENV CODEATLAS_DATA_DIR=/data \
    CODEATLAS_MODEL_PATH=/app/models/risk_model.joblib \
    PYTHONUNBUFFERED=1
EXPOSE 8000
CMD ["uvicorn", "codeatlas.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
