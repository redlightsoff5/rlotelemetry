# Runs on Hugging Face Spaces (16 GB RAM, free) — and any other Docker host.
FROM python:3.11-slim

WORKDIR /app

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    CACHE_DIR=/tmp/fastf1cache \
    MPLCONFIGDIR=/tmp/mpl

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

# HF Spaces serves on 7860; other hosts (Render, etc.) inject $PORT.
EXPOSE 7860
CMD gunicorn app:server --workers 1 --threads 8 --timeout 120 --bind 0.0.0.0:${PORT:-7860}
