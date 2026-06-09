#!/usr/bin/env bash
# One-shot setup for macOS (Apple Silicon or Intel).
# Idempotent — safe to re-run.
set -euo pipefail

cd "$(dirname "$0")/.."

echo "==> Speech-To-Text setup for macOS"

# 1. Homebrew check
if ! command -v brew >/dev/null 2>&1; then
    echo "Homebrew not found. Install from https://brew.sh and re-run this script."
    exit 1
fi

# 2. System packages
echo "==> Installing system packages (python@3.11, ffmpeg, portaudio, sox)"
brew install python@3.11 ffmpeg portaudio sox

# 3. Python venv
PY=$(brew --prefix python@3.11)/bin/python3.11
if [ ! -d "venv" ]; then
    echo "==> Creating venv with $PY"
    "$PY" -m venv venv
fi

# 4. Pip deps
echo "==> Installing Python dependencies"
# shellcheck disable=SC1091
source venv/bin/activate
pip install --upgrade pip
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
