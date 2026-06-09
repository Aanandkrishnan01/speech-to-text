# ✅ Converted to NVIDIA NeMo Only

The project has been completely converted to use **ONLY NVIDIA NeMo** for ASR (Automatic Speech Recognition).

## What Changed

### ❌ Removed
- OpenAI Whisper dependency
- Backend switching logic (`--backend` flag)
- Whisper configuration options
- All Whisper-related code

### ✅ Now Using
- **NVIDIA NeMo** - Exclusive ASR engine
- Simpler, cleaner codebase
- Better accuracy for English transcription
- GPU-optimized performance

## Installation

```bash
# Install dependencies (NeMo only)
pip install -r requirements.txt
```

## Usage

```bash
# Real-time transcription
python main.py realtime

# Batch transcription
python main.py batch audio.mp3

# Use specific NeMo model
python main.py realtime --model stt_en_conformer_ctc_medium
python main.py batch audio.mp3 --model stt_en_conformer_ctc_large
```

## Available NeMo Models

Configured in [config.json](config.json):

| Model | Purpose | Size | Speed |
|-------|---------|------|-------|
| `stt_en_fastconformer_hybrid_large_streaming_80ms` | **Real-time** (default) | ~400MB | Fast |
| `stt_en_conformer_ctc_large` | **Batch** (default) | ~500MB | Best accuracy |
| `stt_en_conformer_ctc_medium` | Balanced | ~300MB | Good |
| `stt_en_conformer_ctc_small` | Quick processing | ~200MB | Very fast |

## Configuration

[config.json](config.json:1-44) - Simple NeMo-only config:

```json
{
  "models": {
    "diarization": "pyannote/speaker-diarization-3.1",
    "nemo": {
      "realtime": "stt_en_fastconformer_hybrid_large_streaming_80ms",
      "batch": "stt_en_conformer_ctc_large"
    }
  },
  "nemo": {
    "batch_size": 8
  }
}
```

## Project Structure

```
src/
├── transcriber.py           # Base NeMo transcriber
├── batch_transcriber.py     # Batch with NeMo
├── realtime_transcriber.py  # Real-time with NeMo
└── config.py                # NeMo configuration

main.py                      # Simple NeMo-only CLI
config.json                  # NeMo model settings
```

## Why NeMo Only?

✅ **Better Accuracy** - State-of-the-art English ASR
✅ **Simpler Code** - No backend switching
✅ **GPU Optimized** - Excellent CUDA/MPS support
✅ **Production Ready** - Used in enterprise applications
✅ **Cleaner** - Single engine, easier to maintain

## First Run

NeMo will download models (~200-500MB) on first use:

```bash
python main.py verify  # Check setup
python main.py realtime  # Start transcription
```

Models cached in `~/.cache/torch/NeMo/`

## Examples

```bash
# Quick real-time test
python main.py realtime

# High-accuracy batch
python main.py batch meeting.mp3 --model stt_en_conformer_ctc_large

# Fast batch processing
python main.py batch audio.mp3 --model stt_en_conformer_ctc_small
```

---

**The project is now 100% NeMo-powered! 🚀**
