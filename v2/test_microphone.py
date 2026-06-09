#!/usr/bin/env python3
"""
Test microphone input levels.
Press Ctrl+C to stop.
"""

import numpy as np
import sounddevice as sd
import sys

SAMPLE_RATE = 16000
BLOCK_SIZE = int(SAMPLE_RATE * 0.1)  # 100ms blocks

def audio_callback(indata, frames, time_info, status):
    """Print audio levels."""
    if status:
        print(f"Status: {status}")

    # Convert to mono
    audio_data = indata[:, 0] if len(indata.shape) > 1 else indata
    audio_data = audio_data.flatten().astype(np.float32)

    # Calculate RMS (root mean square) - volume level
    rms = np.sqrt(np.mean(audio_data ** 2))

    # Create visual bar
    bar_length = int(rms * 500)  # Scale for visibility
    bar = '█' * min(bar_length, 50)

    # Show level and threshold indicators
    print(f"\rVolume: {rms:.4f} |{bar:<50}| ", end='')

    # Show if it would trigger transcription at different thresholds
    if rms > 0.05:
        print("[LOUD - would trigger at 0.05]", end='')
    elif rms > 0.02:
        print("[NORMAL - would trigger at 0.02]", end='')
    elif rms > 0.01:
        print("[QUIET - would trigger at 0.01]", end='')
    else:
        print("[SILENCE - wouldn't trigger]", end='')

print("=" * 80)
print("MICROPHONE LEVEL TEST")
print("=" * 80)
print("\nSpeak into your microphone to see input levels.")
print("Current threshold in config.json: 0.02")
print("\nPress Ctrl+C to stop...\n")

try:
    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        callback=audio_callback,
        blocksize=BLOCK_SIZE,
        dtype=np.float32
    ):
        while True:
            sd.sleep(100)

except KeyboardInterrupt:
    print("\n\nTest complete!")
    print("\nRecommendations:")
    print("- If bars appear when you speak, your mic is working")
    print("- Adjust silence_threshold in config.json based on your levels:")
    print("  • Background noise mostly under 0.01: use threshold 0.015")
    print("  • Background noise 0.01-0.02: use threshold 0.025")
    print("  • Background noise over 0.02: use threshold 0.03-0.05")
    sys.exit(0)

except Exception as e:
    print(f"\n\nError: {e}")
    sys.exit(1)
