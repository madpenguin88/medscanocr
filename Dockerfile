# ─── Stage 1: Build/deps ──────────────────────────────────────────────────────
FROM python:3.12-slim AS builder

WORKDIR /app

# Instalează dependențele într-un venv izolat
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ─── Stage 2: Runtime ─────────────────────────────────────────────────────────
FROM python:3.12-slim AS runtime

WORKDIR /app

# Copiază venv-ul din stage-ul de build
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copiază codul sursă (fără .env — injectat la runtime)
COPY main.py .

ENV PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
