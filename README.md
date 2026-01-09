# Speech-to-Text with Speaker Diarization

A Python application that transcribes audio files with speaker diarization using OpenAI Whisper and pyannote.audio.

## Features

- **Speaker Diarization**: Identifies and separates different speakers in the audio
- **Speech Transcription**: Converts speech to text using OpenAI Whisper
- **Segment Merging**: Intelligently merges segments from the same speaker
- **Overlap Handling**: Resolves overlapping speaker segments

## Requirements

- Python 3.8+
- PyTorch
- Hugging Face token (for pyannote.audio models)

## Installation

1. Clone this repository or navigate to the project directory.

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Set up your Hugging Face token:
   - Get a token from [Hugging Face](https://huggingface.co/settings/tokens)
   - Accept the terms for the [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1) model
   - Set the token as an environment variable:
     ```bash
     export HF_TOKEN=your_token_here
     ```
   - **Important**: Never commit your token to version control. Always use environment variables.

## Usage

1. Place your audio file in the project directory and update the `AUDIO_FILE` variable in `app.py` (default: `"nithin.mp3"`).

2. Run the application:
```bash
python app.py
```

3. The transcript will be saved to the file specified in `OUT_FILE` (default: `"nithin-transcript.txt"`).

## Configuration

You can adjust the following parameters in `app.py`:

- `MIN_SEG_DUR`: Minimum segment duration in seconds (default: 0.60) - drops tiny diarization segments
- `MERGE_GAP`: Gap threshold in seconds (default: 0.35) - merges same-speaker segments if gap is smaller
- `AUDIO_FILE`: Input audio file path
- `OUT_FILE`: Output transcript file path
- Whisper model: Change `"small.en"` to `"medium.en"` or `"large-v2"` for better quality (requires more resources)

## Output Format

The transcript file contains entries in the following format:
```
[START_TIME -- END_TIME] SPEAKER: transcribed_text
```

Example:
```
[5.23s -- 8.45s] SPEAKER_00: Hello, how are you today?
[8.50s -- 12.30s] SPEAKER_01: I'm doing great, thank you!
```

## Notes

- The audio is automatically converted to mono, 16kHz, 16-bit format for processing
- The application uses memory management techniques to handle large audio files
- Processing time depends on audio length and selected Whisper model size

## License

This project uses libraries with their respective licenses:
- OpenAI Whisper: MIT License
- pyannote.audio: MIT License
- PyTorch: BSD-style License

