# macOS-compatible image — no CUDA, no nvidia-smi.
# MPS is a host-only runtime (Apple Silicon); Docker on macOS cannot access MPS,
# so the container runs on CPU. For MPS inference run the app natively outside Docker.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    POETRY_VERSION=1.8.3 \
    POETRY_HOME=/opt/poetry \
    POETRY_VIRTUALENVS_IN_PROJECT=true \
    PATH="/opt/poetry/bin:$PATH" \
    DEVICE=cpu

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl build-essential git \
    && rm -rf /var/lib/apt/lists/*

RUN curl -sSL https://install.python-poetry.org | python3 -

WORKDIR /app
COPY pyproject.toml poetry.lock* ./

RUN poetry install --extras "local-mps" --no-root --no-interaction

COPY . .
RUN poetry install --extras "local-mps" --no-interaction

EXPOSE 8000
CMD ["poetry", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
