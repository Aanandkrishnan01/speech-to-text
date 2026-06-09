"""
Configuration management for Speech-to-Text system.
Loads settings from config.json and environment variables from .env
"""

import json
import os
from pathlib import Path
from dotenv import load_dotenv


class Config:
    """Configuration manager."""

    def __init__(self, config_path="config.json", env_path=".env"):
        """
        Initialize configuration.

        Args:
            config_path: Path to config.json
            env_path: Path to .env file
        """
        # Load environment variables from .env
        load_dotenv(env_path)

        # Get project root (parent of src/)
        self.root_dir = Path(__file__).parent.parent
        self.config_path = self.root_dir / config_path

        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        # Load configuration
        with open(self.config_path, 'r') as f:
            self._config = json.load(f)

        # HF token is optional — only required if you load a gated HuggingFace
        # model. NeMo (NGC) and the public Qwen3-ASR weights don't need it.
        self.hf_token = os.getenv("HF_TOKEN") or None

    def get(self, *keys, default=None):
        """Get nested configuration value."""
        value = self._config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value

    @property
    def sample_rate(self):
        """Audio sample rate."""
        return self.get("audio", "sample_rate", default=16000)

    @property
    def channels(self):
        """Audio channels."""
        return self.get("audio", "channels", default=1)

    @property
    def asr_backend(self):
        """ASR backend: 'nemo' or 'qwen'."""
        return self.get("asr_backend", default="nemo")

    @property
    def qwen_realtime_model(self):
        """Qwen3-ASR model id for real-time transcription."""
        return self.get("models", "qwen", "realtime", default="Qwen/Qwen3-ASR-0.6B")

    @property
    def qwen_batch_model(self):
        """Qwen3-ASR model id for batch transcription."""
        return self.get("models", "qwen", "batch", default="Qwen/Qwen3-ASR-1.7B")

    @property
    def qwen_language(self):
        """Qwen3-ASR language hint (None = auto-detect)."""
        return self.get("models", "qwen", "language", default=None)

    @property
    def gemini_realtime_model(self):
        """Gemini model id for real-time transcription."""
        return self.get("models", "gemini", "realtime", default="gemini-2.5-flash")

    @property
    def gemini_batch_model(self):
        """Gemini model id for batch transcription."""
        return self.get("models", "gemini", "batch", default="gemini-2.5-pro")

    @property
    def gemini_language(self):
        """Gemini language hint (None = auto-detect)."""
        return self.get("models", "gemini", "language", default=None)

    @property
    def gemini_api_key(self):
        """Gemini API key from env (GEMINI_API_KEY or GOOGLE_API_KEY)."""
        return os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or None

    @property
    def google_stt_realtime_model(self):
        return self.get("models", "google-stt", "realtime", default="chirp_2")

    @property
    def google_stt_batch_model(self):
        return self.get("models", "google-stt", "batch", default="chirp_3")

    @property
    def google_stt_location(self):
        """GCP region for Speech-to-Text. 'global' is required for chirp_3."""
        return self.get("models", "google-stt", "location", default="global")

    @property
    def google_stt_languages(self):
        """List of language codes (e.g. ['en-US']). Default ['en-US']."""
        langs = self.get("models", "google-stt", "languages", default=["en-US"])
        if isinstance(langs, str):
            langs = [langs]
        return list(langs)

    @property
    def google_stt_project(self):
        """GCP project id. Falls back to env GOOGLE_CLOUD_PROJECT / GCP_PROJECT."""
        return (
            self.get("models", "google-stt", "project", default=None)
            or os.getenv("GOOGLE_CLOUD_PROJECT")
            or os.getenv("GCP_PROJECT")
            or None
        )

    # ── Whisper (faster-whisper) ──────────────────────────────────────────

    @property
    def whisper_realtime_model(self):
        """Whisper model size for realtime: tiny|base|small|medium|large-v3."""
        return self.get("models", "whisper", "realtime", default="base")

    @property
    def whisper_batch_model(self):
        """Whisper model size for batch (typically larger for accuracy)."""
        return self.get("models", "whisper", "batch", default="small")

    @property
    def whisper_language(self):
        """ISO-639-1 code (e.g. 'en') or None for auto-detect."""
        return self.get("models", "whisper", "language", default=None)

    @property
    def whisper_beam_size(self):
        return int(self.get("models", "whisper", "beam_size", default=5))

    @property
    def whisper_vad_filter(self):
        """Apply Whisper's built-in Silero VAD pre-filter to skip silences."""
        return bool(self.get("models", "whisper", "vad_filter", default=True))

    @property
    def available_models(self):
        """
        Returns dict of {backend_name: {model_id: description, ...}} as listed
        in config.json under models.<backend>.available. Used by the web UI to
        populate the model-selector dropdown.
        """
        return {
            "nemo": self.get("models", "nemo", "available", default={}) or {},
            "qwen": self.get("models", "qwen", "available", default={}) or {},
            "gemini": self.get("models", "gemini", "available", default={}) or {},
            "google-stt": self.get("models", "google-stt", "available", default={}) or {},
            "whisper": self.get("models", "whisper", "available", default={}) or {},
        }

    @property
    def nemo_realtime_model(self):
        """NeMo model for real-time transcription."""
        return self.get("models", "nemo", "realtime", default="stt_en_fastconformer_hybrid_large_streaming_80ms")

    @property
    def nemo_batch_model(self):
        """NeMo model for batch transcription."""
        return self.get("models", "nemo", "batch", default="stt_en_conformer_ctc_large")

    @property
    def nemo_params(self):
        """NeMo ASR parameters."""
        return {
            "batch_size": self.get("nemo", "batch_size", default=8),
            "preserve_alignment": self.get("nemo", "preserve_alignment", default=False),
            "verbose": self.get("nemo", "verbose", default=False),
        }


    @property
    def chunk_duration(self):
        """Real-time chunk duration."""
        return self.get("realtime", "chunk_duration", default=2.0)

    @property
    def buffer_duration(self):
        """Real-time buffer duration."""
        return self.get("realtime", "buffer_duration", default=10.0)

    @property
    def silence_threshold(self):
        """Silence detection threshold."""
        return self.get("realtime", "silence_threshold", default=0.01)

    @property
    def save_transcript(self):
        """Whether to save transcript to file."""
        return self.get("realtime", "save_transcript", default=True)

    @property
    def print_to_console(self):
        """Whether to print to console."""
        return self.get("realtime", "print_to_console", default=True)

    # ── Chunk boundary strategy (Task 5) ──────────────────────────────────

    @property
    def boundary_strategy(self):
        """'vad' | 'overlap' | 'none'. Default 'vad'."""
        return (self.get("realtime", "boundary_strategy", default="vad") or "vad").lower()

    @property
    def max_chunk_duration(self):
        """Hard upper bound on a single chunk before force-cutting."""
        return float(self.get("realtime", "max_chunk_duration", default=4.0))

    @property
    def overlap_duration(self):
        """Overlap between consecutive chunks (used by overlap strategy)."""
        return float(self.get("realtime", "overlap_duration", default=0.5))

    @property
    def min_silence_ms(self):
        return int(self.get("realtime", "min_silence_ms", default=200))

    @property
    def handle_overlapping_speakers(self):
        """Whether to handle overlapping speakers in batch mode."""
        return self.get("batch", "handle_overlapping_speakers", default=True)

    @property
    def transcript_dir(self):
        """Directory for saving transcripts."""
        dir_name = self.get("output", "transcript_dir", default="transcripts")
        transcript_path = self.root_dir / dir_name
        transcript_path.mkdir(exist_ok=True)
        return transcript_path

    # ── Recording (Task 4) ────────────────────────────────────────────────

    @property
    def recording_enabled(self):
        """Whether to persist captured audio to disk."""
        return bool(self.get("recording", "enabled", default=False))

    @property
    def recording_format(self):
        """'mp3' (default) or 'wav'."""
        return self.get("recording", "format", default="mp3")

    @property
    def recording_bitrate_kbps(self):
        return int(self.get("recording", "bitrate_kbps", default=128))

    @property
    def recording_max_part_mb(self):
        """Maximum size per recorded part before rolling over."""
        return int(self.get("recording", "max_part_mb", default=10))

    @property
    def recording_max_part_bytes(self):
        return self.recording_max_part_mb * 1024 * 1024

    @property
    def recording_dir(self):
        """Directory for saved recorded audio files."""
        dir_name = self.get("recording", "dir", default="audio")
        recording_path = self.root_dir / dir_name
        recording_path.mkdir(exist_ok=True)
        return recording_path

    @property
    def batch_window_sec(self):
        """Batch-mode window size for streaming a long file through ASR."""
        return float(self.get("recording", "batch_window_sec", default=300))

    # ── Benchmark (Task 6) ────────────────────────────────────────────────

    @property
    def benchmark_cache_dir(self):
        d = self.get("benchmark", "cache_dir", default="benchmark_cache")
        p = self.root_dir / d
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def benchmark_output_dir(self):
        d = self.get("benchmark", "output_dir", default="benchmark_reports")
        p = self.root_dir / d
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def benchmark_datasets(self):
        """Per-dataset config dict (e.g. {'librispeech': {split, hf_repo}})."""
        return self.get("benchmark", "datasets", default={}) or {}

    @property
    def timestamp_format(self):
        """Timestamp format for output files."""
        return self.get("output", "timestamp_format", default="%Y%m%d_%H%M%S")

    @property
    def log_dir(self):
        """Directory for log files."""
        dir_name = self.get("logging", "log_dir", default="logs")
        log_path = self.root_dir / dir_name
        return log_path

    @property
    def log_file_level(self):
        return self.get("logging", "file_level", default="DEBUG")

    @property
    def log_console_level(self):
        return self.get("logging", "console_level", default="WARNING")

    @property
    def log_file_enabled(self):
        return self.get("logging", "file_enabled", default=True)

    @property
    def log_console_enabled(self):
        return self.get("logging", "console_enabled", default=True)


# Singleton instance
_config_instance = None


def get_config(reload=False):
    """Get global configuration instance."""
    global _config_instance
    if _config_instance is None or reload:
        _config_instance = Config()
    return _config_instance
