# Speech-to-Text container.
# Slim Python base + native audio libs. Model weights are NOT bundled — they
# download on first run into the volume mounted at /root/.cache (see
# docker-compose.yml).

FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive \
    NEMO_LOG_LEVEL=ERROR

# System deps:
# - ffmpeg: pydub audio decode
# - libportaudio2: sounddevice mic input (Linux host only)
# - libsndfile1: soundfile I/O
# - sox + libsox-fmt-all: qwen-asr resampling
# - git: a few pip deps fetch via VCS
# - build-essential, gcc: source builds for any wheel that has no manylinux variant
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libportaudio2 \
        libsndfile1 \
        sox \
        libsox-fmt-all \
        git \
        ca-certificates \
        build-essential \
        gcc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (separate layer for caching)
COPY requirements.txt ./
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy the application
COPY main.py config.json ./
COPY src/ ./src/
COPY static/ ./static/

# Create directories that get mounted as volumes (mountpoints must exist)
RUN mkdir -p /app/transcripts /app/logs /root/.cache

# Web server port (multi-user mode). No-op when not running `serve`.
EXPOSE 8000

# Default to verifying the install. Compose overrides this with the actual command.
CMD ["python", "main.py", "verify"]
