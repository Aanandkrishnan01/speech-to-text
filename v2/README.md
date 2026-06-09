# Speech-to-Text with Multi-Speaker Support

Real-time and batch transcription with automatic speaker diarization. Choose
between two ASR backends:

- **NVIDIA NeMo** — fast, lightweight, English-only Conformer / FastConformer models
- **Qwen3-ASR** — multilingual (52 languages + 22 Chinese dialects), 0.6B and 1.7B variants

Speaker diarization (NeMo VAD + titanet) runs regardless of which ASR backend
you pick.

## ✨ Features

- 🎤 **Real-time Transcription** — live microphone input with speaker labels
- 📁 **Batch Processing** — pre-recorded audio files (MP3, WAV, etc.)
- 👥 **Speaker Diarization** — automatic per-speaker labeling
- 🔄 **Pluggable ASR backend** — NeMo or Qwen3-ASR via one config flag
- 📝 **Structured logging** — file + console, configurable levels
- ⚙️ **Configurable** via `config.json`

## 📋 Requirements

- Python 3.11
- FFmpeg, PortAudio, libsndfile, sox (system packages — see [PLATFORMS.md](PLATFORMS.md))
- Hugging Face token *(optional — only needed for gated HF models)*

## 🚀 Install (pick one path)

### A. macOS (one-shot script)

```bash
./scripts/setup-mac.sh
```

Installs `python@3.11`, `ffmpeg`, `portaudio`, `sox` via Homebrew, creates a
venv, installs Python deps, and runs `python main.py verify`.

### B. Linux (Debian/Ubuntu, one-shot script)

```bash
./scripts/setup-linux.sh
```

Auto-detects NVIDIA GPU and installs the CUDA-enabled `torch` wheel when
present. For other distros, follow [PLATFORMS.md](PLATFORMS.md) manually.

### C. Windows (one-shot PowerShell script)

```powershell
.\scripts\setup-windows.ps1
```

Requires [Chocolatey](https://chocolatey.org/install). Run from PowerShell
(admin on first run for `choco`).

### D. Docker (cross-platform, batch mode)

```bash
docker compose build
cp .env.example .env  # add HF_TOKEN
docker compose run --rm stt python main.py verify
docker compose run --rm stt python main.py batch /audio/yourfile.mp3
```

Drop audio files in `./audio/` to make them visible inside the container at
`/audio/`. Transcripts and logs land back on the host in `./transcripts/`
and `./logs/`. Model weights are cached in a Docker named volume — see
[PLATFORMS.md § 5](PLATFORMS.md#5-docker) for details.

> **Realtime mic mode in Docker** works on Linux only (with PulseAudio
> passthrough). On macOS/Windows use a native install (A, B, or C) for
> realtime; Docker handles batch fine.

### Manual install (all OSes)

If you prefer not to use the scripts, see [PLATFORMS.md](PLATFORMS.md) for
the per-OS commands.

## ⚙️ Configure

```bash
cp .env.example .env  # HF_TOKEN is optional — leave blank unless you use gated HF models
```

Then in `config.json`, pick your ASR backend:

```jsonc
{
  "asr_backend": "nemo",   // or "qwen"
  ...
}
```

## ✅ Verify

```bash
python main.py verify
```

Confirms Python deps, audio devices, FFmpeg, GPU/MPS availability, and the
active ASR backend.

## ▶️ Run

**Real-time (from microphone):**
```bash
python main.py realtime
```

**Batch (from audio file):**
```bash
python main.py batch audio.mp3
```

## 📁 Project Structure

```
Speech-To-Text-STT-/
├── main.py                    # Main entry point
├── config.json                # Configuration settings
├── .env                       # Environment variables (HF_TOKEN)
├── .env.example               # Example environment file
├── requirements.txt           # Python dependencies
├── README.md                  # This file
├── PLATFORMS.md               # Cross-platform compatibility & dep matrix
├── Dockerfile                 # Container recipe
├── docker-compose.yml         # Compose with cache + I/O volumes
├── .dockerignore
│
├── scripts/                   # One-shot setup scripts per OS
│   ├── setup-mac.sh
│   ├── setup-linux.sh
│   └── setup-windows.ps1
│
├── src/                       # Source code
│   ├── __init__.py
│   ├── config.py              # Configuration loader
│   ├── logger.py              # Logging setup
│   ├── audio_utils.py         # Audio utilities
│   ├── asr_backends.py        # NeMo + Qwen3-ASR backends (pluggable)
│   ├── transcriber.py         # Base transcriber (diarization + ASR delegation)
│   ├── realtime_transcriber.py
│   └── batch_transcriber.py
│
├── transcripts/               # Generated transcripts (auto-created)
├── logs/                      # Application logs (auto-created)
└── demo.ipynb                 # Jupyter notebook examples
```

## 🎯 Usage

### Command-Line Interface

```bash
# Show help
python main.py --help

# Real-time transcription
python main.py realtime

# Real-time with specific NeMo model
python main.py realtime --model stt_en_conformer_ctc_medium

# Batch transcription
python main.py batch audio.mp3

# Batch with custom output file
python main.py batch audio.mp3 -o my_transcript.txt

# Batch with specific NeMo model
python main.py batch meeting.mp3 --model stt_en_conformer_ctc_large

# Verify setup
python main.py verify
```

### Configuration

Edit [config.json](config.json) to customize settings:

```json
{
  "models": {
    "nemo": {
      "realtime": "stt_en_fastconformer_hybrid_large_pc",
      "batch": "stt_en_conformer_ctc_large",
      "vad": "vad_multilingual_marblenet",
      "speaker": "titanet_large"
    }
  },
  "audio": {
    "sample_rate": 16000,
    "channels": 1
  },
  "realtime": {
    "chunk_duration": 3.0,     // Process every N seconds
    "buffer_duration": 15.0,   // Context window size
    "save_transcript": false,  // Auto-save to file
    "print_to_console": true   // Print output live
  },
  "diarization": {
    "min_segment_duration": 1.2,  // Skip short segments
    "merge_gap": 0.6               // Merge nearby same-speaker
  }
}
```

### NeMo Model Selection

| Model | Speed | Quality | Size | Use Case |
|-------|-------|---------|------|----------|
| `stt_en_conformer_ctc_small` | Very Fast | Good | ~200MB | Testing, low-end hardware |
| `stt_en_conformer_ctc_medium` | Fast | Better | ~300MB | Balanced performance |
| `stt_en_conformer_ctc_large` | Moderate | High | ~500MB | **Batch recommended** |
| `stt_en_fastconformer_hybrid_large_pc` | Fast | High | ~400MB | **Real-time recommended** |
| `stt_en_fastconformer_hybrid_large_streaming_80ms` | Very Fast | High | ~400MB | Streaming applications |

## 📤 Output Format

Transcripts are saved with timestamps and speaker labels:

```
Transcript for: meeting_audio.mp3
Generated with: NVIDIA NeMo
Generated: 2024-02-11 14:30:22
============================================================

[0.00s - 3.45s] SPEAKER_00: Hello, how are you doing today?
[3.50s - 6.23s] SPEAKER_01: I'm doing great, thanks for asking!
[6.50s - 10.12s] SPEAKER_00: That's wonderful to hear.
```

Output files are saved to `transcripts/` directory with automatic timestamping.

## 🔧 Advanced Usage

### Using as a Python Module

```python
from src.config import get_config
from src.batch_transcriber import BatchTranscriber

# Get configuration
config = get_config()

# Create transcriber
transcriber = BatchTranscriber(nemo_model="stt_en_conformer_ctc_medium", config=config)

# Transcribe file
results = transcriber.transcribe_file("audio.mp3")

# Access results
for segment in results:
    print(f"{segment['speaker']}: {segment['text']}")
```

### Customizing Configuration Programmatically

```python
from src.config import Config

# Load config
config = Config()

# Override settings
config._config['realtime']['chunk_duration'] = 3.0
config._config['models']['nemo']['batch'] = 'stt_en_conformer_ctc_large'

# Use custom config
from src.batch_transcriber import BatchTranscriber
transcriber = BatchTranscriber(config=config)
```

## 🐛 Troubleshooting

### HF_TOKEN

`HF_TOKEN` is **optional**. NeMo models pull from NVIDIA NGC and the Qwen3-ASR
weights are public on HuggingFace — neither requires authentication. Only set
`HF_TOKEN` in `.env` if you want to use a gated model from HuggingFace Hub.

### FFmpeg Not Found

```bash
# Check if FFmpeg is installed
ffmpeg -version

# macOS installation
brew install ffmpeg

# Ubuntu/Debian
sudo apt-get install ffmpeg
```

### Audio Device Issues

```bash
# List available audio devices
python -c "import sounddevice as sd; print(sd.query_devices())"
```

### Out of Memory

- Use smaller NeMo model (`stt_en_conformer_ctc_small`)
- Reduce `buffer_duration` in config.json
- Close other applications

### Poor Transcription Quality

- Use larger NeMo model (`stt_en_conformer_ctc_large`)
- Ensure good microphone quality
- Reduce background noise
- Speak clearly and at moderate pace

## 🎓 Examples

### Example 1: Quick Meeting Transcript

```bash
# Record your meeting (using any recording app)
# Save as meeting.mp3

# Transcribe with good quality
python main.py batch meeting.mp3 --model stt_en_conformer_ctc_large

# Find output in transcripts/
ls transcripts/
```

### Example 2: Live Interview Transcription

```bash
# Start real-time transcription
python main.py realtime

# Conduct interview (speak into mic)
# Press Ctrl+C when done

# Transcript auto-saved to transcripts/realtime_*.txt
```

### Example 3: Process Multiple Files

```bash
# Batch process all MP3 files
for file in *.mp3; do
    python main.py batch "$file" --model stt_en_conformer_ctc_medium
done
```

## 🔬 Technical Details

### Architecture

- **Speech Recognition**: NVIDIA NeMo ASR (Conformer/FastConformer models)
- **Speaker Diarization**: NVIDIA NeMo (VAD + Speaker Embeddings + Clustering)
- **Audio I/O**: SoundDevice (real-time), PyDub (batch)
- **Configuration**: JSON + python-dotenv

### Processing Pipeline

1. **Audio Input** → Microphone or file
2. **Preprocessing** → Normalize, resample to 16kHz mono
3. **VAD** → Detect speech segments using NeMo VAD
4. **Speaker Embeddings** → Extract speaker features using NeMo TitaNet
5. **Clustering** → Group segments by speaker using spectral clustering
6. **Transcription** → NeMo ASR for each speaker segment
7. **Post-processing** → Merge segments, format output
8. **Output** → Console and/or file

## 📊 Performance

Approximate processing times (M1 MacBook Pro with MPS):

| NeMo Model | 1 min audio | Real-time Factor |
|------------|-------------|------------------|
| conformer_ctc_small  | ~8 sec      | 0.13x            |
| conformer_ctc_medium | ~12 sec     | 0.20x            |
| conformer_ctc_large  | ~20 sec     | 0.33x            |
| fastconformer_hybrid_large_pc | ~10 sec | 0.17x       |
| fastconformer_hybrid_streaming | ~6 sec | 0.10x        |

*Real-time factor: 1.0x = same time as audio duration*

NeMo models are generally faster and more memory-efficient than comparable Whisper models.

## 🗺️ Roadmap

- [x] NVIDIA NeMo integration for ASR and diarization
- [ ] Web UI interface
- [ ] Multiple language support (leveraging NeMo's multilingual models)
- [ ] Custom speaker naming
- [ ] Export to SRT/VTT subtitles
- [ ] Enhanced GPU optimization
- [ ] Streaming API with WebSockets

## 📝 License

[Add your license here]

## 🤝 Contributing

Contributions welcome! Please open an issue or submit a PR.

## 🙏 Acknowledgments

- [NVIDIA NeMo](https://github.com/NVIDIA/NeMo) - ASR and speaker diarization models
- [SoundDevice](https://python-sounddevice.readthedocs.io/) - Audio I/O
- [Scikit-learn](https://scikit-learn.org/) - Clustering algorithms

## 📞 Support

For issues and questions:
- Check [Troubleshooting](#-troubleshooting) section
- Run `python main.py verify` to diagnose setup issues
- Open an issue on GitHub

---

Made with ❤️ using Claude AI
