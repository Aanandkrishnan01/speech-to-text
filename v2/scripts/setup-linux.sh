#!/usr/bin/env bash
# One-shot setup for Debian/Ubuntu Linux.
# For other distros, follow PLATFORMS.md manually.
# Idempotent — safe to re-run.
set -euo pipefail

cd "$(dirname "$0")/.."

echo "==> Speech-To-Text setup for Linux (Debian/Ubuntu)"

# 1. apt check
if ! command -v apt-get >/dev/null 2>&1; then
    echo "This script targets Debian/Ubuntu (apt). For other distros see PLATFORMS.md."
    exit 1
fi

# 2. System packages (need sudo)
echo "==> Installing system packages (requires sudo)"
sudo apt-get update
sudo apt-get install -y \
    python3.11 python3.11-venv python3-pip \
    ffmpeg libportaudio2 libsndfile1 \
    sox libsox-fmt-all \
    git build-essential

# 3. Python venv
if [ ! -d "venv" ]; then
    echo "==> Creating venv with python3.11"
    python3.11 -m venv venv
fi

# 4. Pip deps
echo "==> Installing Python dependencies"
# shellcheck disable=SC1091
source venv/bin/activate
pip install --upgrade pip

# 4a. Optional CUDA torch — auto-detect NVIDIA driver
if command -v nvidia-smi >/dev/null 2>&1; then
    echo "==> NVIDIA driver detected — installing CUDA-enabled torch first"
    pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu121 || {
        echo "==> CUDA torch install failed, continuing with default wheel"
    }
fi

pip install -r requirements.txt

# 5. .env scaffold (HF_TOKEN is optional — only needed for gated HF models)
if [ ! -f .env ]; then
    if [ -f .env.example ]; then
        cp .env.example .env
    else
        echo "HF_TOKEN=" > .env
    fi
    echo "==> Created .env (HF_TOKEN is optional, leave empty unless you use gated models)"
fi

# 6. Verify
echo "==> Running verify"
python main.py verify

echo
echo "Setup complete. Activate the venv with:"
echo "  source venv/bin/activate"
echo "Then run:"
echo "  python main.py realtime"
