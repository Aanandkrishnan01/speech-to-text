# Platform Compatibility & Dependency Matrix

This project depends on a mix of Python wheels, native libraries, and ML model
runtimes that behave differently across operating systems. This file is the
canonical reference for what works where, what to install per OS, and which
limitations to expect.

> **TL;DR**
> - **macOS** (Apple Silicon): fully supported. ASR runs on MPS. Mic input works natively.
> - **Linux** (x86_64, with NVIDIA): fully supported with the best performance (CUDA + vLLM).
> - **Windows 10/11**: supported for batch + realtime via the native install. NeMo training/streaming features are limited; transcription works.
> - **Docker**: supported on all three for batch mode. Realtime mic capture only works on Linux hosts (or WSL2 with audio passthrough configured).

---

## 1. Required Python version

Pin **Python 3.11.x**. Newer (3.12/3.13) breaks `nemo_toolkit==2.7.3`'s pinned deps; older (3.10 and below) is missing typing features used in this codebase.

| OS      | Recommended install                                                                        |
| ------- | ------------------------------------------------------------------------------------------ |
| macOS   | `brew install python@3.11`                                                                 |
| Linux   | `apt install python3.11 python3.11-venv` (Ubuntu 22.04+) or `dnf install python3.11`       |
| Windows | [python.org installer](https://www.python.org/downloads/) — tick "Add to PATH" at install |

---

## 2. Required system libraries (non-pip)

These are **NOT installed by `pip install -r requirements.txt`**. They must
exist on the host before the Python install will work.

| Package        | Purpose                                | macOS                    | Linux (Debian/Ubuntu)                | Windows                                                            |
| -------------- | -------------------------------------- | ------------------------ | ------------------------------------ | ------------------------------------------------------------------ |
| **ffmpeg**     | Audio decoding (used by `pydub`)       | `brew install ffmpeg`    | `sudo apt install ffmpeg`            | `choco install ffmpeg` or download from ffmpeg.org and add to PATH |
| **PortAudio**  | Mic input (used by `sounddevice`)      | `brew install portaudio` | `sudo apt install libportaudio2`     | Bundled in the `sounddevice` wheel — nothing to install            |
| **libsndfile** | Audio I/O (used by `soundfile`)        | bundled in macOS wheels  | `sudo apt install libsndfile1`       | Bundled in the `soundfile` wheel                                   |
| **sox**        | Audio resampling (used by `qwen-asr`)  | `brew install sox`       | `sudo apt install sox libsox-fmt-all` | `choco install sox`                                                |
| **git**        | Some pip deps fetch from git           | `brew install git`       | `sudo apt install git`               | Included with Git for Windows / GitHub Desktop                     |

---

## 3. Python package compatibility

### `torch` (PyTorch)

The default `torch==2.11.0` wheel works everywhere, but the active hardware
acceleration differs:

| OS / hardware                  | Acceleration       | What this project uses                                            |
| ------------------------------ | ------------------ | ----------------------------------------------------------------- |
| macOS + Apple Silicon (M1/M2/M3) | **MPS**            | Auto-detected via `torch.backends.mps.is_available()`             |
| Linux + NVIDIA GPU             | **CUDA**           | Auto-detected via `torch.cuda.is_available()` — fastest path      |
| Windows + NVIDIA GPU           | **CUDA**           | Same as Linux                                                     |
| Linux/Mac/Win without GPU      | **CPU**            | Fallback — usable but slow for the 7B Qwen models                 |
| AMD GPU (ROCm)                 | **Not supported**  | NeMo + qwen-asr stack does not have ROCm wheels                   |

### `nemo_toolkit==2.7.3`

| OS      | Status                                                                                                    |
| ------- | --------------------------------------------------------------------------------------------------------- |
| Linux   | First-class. All features work.                                                                           |
| macOS   | Works for inference. Some training-only Apex/Megatron features are missing (warnings on startup are normal — see below). |
| Windows | Works for inference. PyTorch distributed warnings are shown at startup (`Redirects are currently not supported in Windows or MacOs`) but do not affect transcription. |

The warnings you can safely ignore on Mac/Windows:
```
[NeMo W ... megatron_init: Megatron num_microbatches_calculator not found, using Apex version.]
[NOTE: Redirects are currently not supported in Windows or MacOs.]
OneLogger: Setting error_handling_strategy to DISABLE_QUIETLY_AND_REPORT_METRIC_ERROR ...
```

### `qwen-asr==0.0.6`

| OS      | Status                                                                                            |
| ------- | ------------------------------------------------------------------------------------------------- |
| Linux   | Full support including the `[vllm]` extra (faster inference, streaming).                          |
| macOS   | Inference only — `vllm==0.14.0` is NOT installable on macOS. Do **not** use the `[vllm]` extra.   |
| Windows | Inference only. Same vLLM limitation as macOS.                                                    |

`qwen-asr 0.0.6` strictly pins `accelerate==1.12.0`. The `requirements.txt` in
this repo matches that pin to keep dependency resolution clean.

### `sounddevice==0.5.5`

Mic capture library. Works on all three OSes once PortAudio is present.

### `vllm` (optional, Linux + CUDA only)

If running on a Linux box with NVIDIA GPU and you want faster Qwen inference,
install with:
```bash
pip install -U "qwen-asr[vllm]"
```

This will fail on macOS/Windows — leave it out unless you're on Linux+CUDA.

---

## 4. Per-OS install paths

### macOS (Apple Silicon)

```bash
brew install python@3.11 ffmpeg portaudio sox
git clone <this-repo>
cd Speech-To-Text-STT-
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # HF_TOKEN is optional — only needed for gated HF models
python main.py verify
```

### Linux (Ubuntu 22.04+, Debian 12+)

```bash
sudo apt update
sudo apt install -y python3.11 python3.11-venv python3-pip \
    ffmpeg libportaudio2 libsndfile1 sox libsox-fmt-all git
git clone <this-repo>
cd Speech-To-Text-STT-
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # HF_TOKEN is optional — only needed for gated HF models
python main.py verify
```

For NVIDIA GPU acceleration: install the matching CUDA-enabled PyTorch *before*
the rest of `requirements.txt`:
```bash
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu121
```

### Windows 10/11

In **PowerShell** (run as administrator the first time):
```powershell
# Install Chocolatey if not present: https://chocolatey.org/install
choco install -y python311 ffmpeg sox.portable git

git clone <this-repo>
cd Speech-To-Text-STT-
py -3.11 -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env  # HF_TOKEN is optional — only needed for gated HF models
python main.py verify
```

> Mic permissions: Windows Settings → Privacy → Microphone → allow desktop apps.

---

## 5. Docker

See [`Dockerfile`](Dockerfile) and [`docker-compose.yml`](docker-compose.yml).

| Use case          | Mac     | Linux     | Windows |
| ----------------- | ------- | --------- | ------- |
| Batch mode        | ✅      | ✅        | ✅      |
| Realtime mic mode | ❌ (no host audio passthrough) | ✅ (`--device /dev/snd` + PulseAudio) | ⚠️ Only via WSL2 with extra setup |
| GPU acceleration  | ❌ (Docker Desktop on Mac has no GPU access) | ✅ (NVIDIA Container Toolkit) | ✅ (CUDA via WSL2) |

### Model weights and the image

**The Docker image does NOT bundle model weights.** They are downloaded on
first run into a Docker named volume (`stt-model-cache`) and reused thereafter.

Rationale:
- Bundling Qwen3-ASR-1.7B + NeMo VAD + titanet would push the image past 15 GB.
- Standard ML pattern: ship a small image, mount a cache volume.
- First launch needs internet; subsequent launches work offline.

To pre-warm the cache (download weights without running the app):
```bash
docker compose run --rm stt python -c "from src.transcriber import BaseTranscriber; BaseTranscriber()"
```

---

## 6. Known issues & workarounds

### `ERROR: Cannot install ... accelerate` during `pip install`
You're seeing the `qwen-asr 0.0.6` pin (`accelerate==1.12.0`) clash with an
older requirements pin. Fixed in this repo as of the latest commit — pull
latest, or manually downgrade: `pip install accelerate==1.12.0`.

### Mic input produces no transcripts
RMS of your mic input is below the silence threshold. Test:
```bash
python -c "import sounddevice as sd, numpy as np; d=sd.rec(int(3*16000),samplerate=16000,channels=1,dtype='float32'); sd.wait(); print('RMS:', np.sqrt(np.mean(d**2)))"
```
Lower `realtime.silence_threshold` in `config.json` if RMS is below the current
value. See [logs/](logs/) for diagnostic detail.

### NeMo logs are too noisy
The `logging` section in `config.json` only controls this project's logger.
NeMo's own logger is silenced via `NEMO_LOG_LEVEL=ERROR` set in
[src/asr_backends.py](src/asr_backends.py) and [src/transcriber.py](src/transcriber.py).

### Qwen3-ASR mis-detects language on short utterances
Set `models.qwen.language: "English"` (or whatever language you speak) in
`config.json` instead of `null` (auto-detect).

### Docker on Mac/Windows can't see the mic
Expected. Use the native install path for realtime mode; use Docker for batch
processing of audio files.
