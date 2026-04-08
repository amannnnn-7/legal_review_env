FROM ghcr.io/meta-pytorch/openenv-base:latest

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app/env

COPY pyproject.toml README.md openenv.yaml Dockerfile ./
COPY server ./server
COPY docs ./docs
COPY src ./src
COPY inference.py ./inference.py
COPY scripts ./scripts

RUN python -m pip install --no-cache-dir --upgrade pip uv \
    && uv pip install --system .

EXPOSE 8000

CMD ["uvicorn", "legal_review_env.server.app:app", "--host", "0.0.0.0", "--port", "8000"]