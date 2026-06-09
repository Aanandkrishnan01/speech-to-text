# NVIDIA NeMo Setup Guide

NVIDIA NeMo provides state-of-the-art ASR (Automatic Speech Recognition) models with better accuracy than Whisper.

## Installation

```bash
# Install NeMo and dependencies
pip install -r requirements.txt

# This will install:
# - nemo_toolkit[asr]
# - omegaconf
```

## Configuration

The project is already configured to use NeMo by default. Check [config.json](config.json:2):

```json
{
  "asr_backend": "nemo",  // Use NeMo instead of Whisper
  "models": {
    "nemo": {
      "asr_model": "stt_en_conformer_ctc_large"  // Best accuracy
    }
  }
}
```

## Available NeMo Models

| Model | Speed | Accuracy | Use Case |
|-------|-------|----------|----------|
| `stt_en_conformer_ctc_small` | Fast | Good | Quick transcription |
| `stt_en_conformer_ctc_medium` | Moderate | Better | **Recommended** |
| `stt_en_conformer_ctc_large` | Slower | **Best** | High accuracy needed |
| `stt_en_fastconformer_hybrid_large_streaming_80ms` | Fast | Good | Real-time streaming |

## Usage

### Option 1: Use Default (from config.json)

```bash
# Real-time with NeMo (uses config.json setting)
python main.py realtime

# Batch with NeMo
python main.py batch audio.mp3
```

### Option 2: Override Backend

```bash
# Force NeMo backend
python main.py realtime --backend nemo

# Force Whisper backend (switch back)
python main.py realtime --backend whisper
```

### Option 3: Specify Model

```bash
# Use specific NeMo model
python main.py batch audio.mp3 --backend nemo --model stt_en_conformer_ctc_medium

# Use Whisper instead
python main.py batch audio.mp3 --backend whisper --model small
```

## Change Backend

### Switch to NeMo (Better Accuracy)

Edit [config.json](config.json:2):
```json
{
  "asr_backend": "nemo",
  ...
}
```

### Switch to Whisper (Faster)

Edit [config.json](config.json:2):
```json
{
  "asr_backend": "whisper",
  ...
}
```

## Performance Comparison

**Whisper vs NeMo:**

| Feature | Whisper | NeMo |
|---------|---------|------|
| Accuracy | Good | **Better** |
| Speed | Fast | Moderate |
| Languages | 99+ | English (best) |
| GPU Support | ✅ | ✅ |
| Apple Silicon | ✅ | ✅ |

## GPU Acceleration

NeMo automatically uses GPU if available:

- **NVIDIA GPU**: Uses CUDA automatically
- **Apple Silicon**: Uses MPS (Metal Performance Shaders)
- **CPU**: Falls back to CPU (slower)

Check during startup:
```
Loading NeMo ASR model: stt_en_conformer_ctc_large...
Using CUDA for NeMo  ← GPU detected
```

## Troubleshooting

### ImportError: No module named 'nemo'

```bash
pip install nemo_toolkit[asr]
```

### Out of Memory

Use a smaller model:
```json
{
  "models": {
    "nemo": {
      "asr_model": "stt_en_conformer_ctc_small"  // Smaller model
    }
  }
}
```

Or reduce batch size:
```json
{
  "nemo": {
    "batch_size": 4  // Reduce from 8 to 4
  }
}
```

### Slow Transcription

- Use GPU (CUDA or MPS)
- Use smaller model (`stt_en_conformer_ctc_small`)
- For real-time, use streaming model: `stt_en_fastconformer_hybrid_large_streaming_80ms`

## Model Downloads

NeMo models are downloaded automatically on first use:
- Cached in `~/.cache/torch/NeMo/`
- ~200-500MB per model
- Requires internet connection (first time only)

## Examples

### Example 1: High Accuracy Batch

```bash
# Use largest NeMo model for best accuracy
python main.py batch important_meeting.mp3 --backend nemo --model stt_en_conformer_ctc_large
```

### Example 2: Fast Real-time

```bash
# Use streaming model for lowest latency
python main.py realtime --backend nemo --model stt_en_fastconformer_hybrid_large_streaming_80ms
```

### Example 3: Balanced

```bash
# Medium model - good balance
python main.py batch audio.mp3 --backend nemo --model stt_en_conformer_ctc_medium
```

## Why Use NeMo?

✅ **Better Accuracy** - State-of-the-art English ASR
✅ **Production Ready** - Used by NVIDIA in commercial products
✅ **GPU Optimized** - Excellent performance on NVIDIA GPUs
✅ **Active Development** - Regular updates and improvements

## When to Use Whisper Instead?

- Need multilingual support (99+ languages)
- Want faster processing (smaller models)
- Limited GPU memory
- Processing non-English audio

---

**Recommendation:** Use NeMo for English transcription when accuracy is important!
