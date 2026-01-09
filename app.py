import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pyannote.audio.core.io")

import os
import gc
import numpy as np
import torch
from pyannote.audio import Pipeline
from pydub import AudioSegment
import whisper

AUDIO_FILE = "audio.mp3"
OUT_FILE = "transcript.txt"

# ---------------- TUNING ----------------
MIN_SEG_DUR = 0.60    # seconds: drop tiny diarization segments
MERGE_GAP   = 0.35    # seconds: merge same-speaker if gap <= this
# ----------------------------------------

HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError(
        "HF_TOKEN environment variable is required. "
        "Please set it using: export HF_TOKEN=your_token_here"
    )

pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    token=HF_TOKEN
)

audio = AudioSegment.from_file(AUDIO_FILE)
audio = audio.set_channels(1).set_frame_rate(16000).set_sample_width(2)

model = whisper.load_model("small.en")  # try "medium.en" for better quality


def segment_to_float32(seg: AudioSegment) -> np.ndarray:
    samples = np.array(seg.get_array_of_samples(), dtype=np.int16)
    return samples.astype(np.float32) / 32768.0


def merge_same_speaker(segments, merge_gap=0.35):
    """
    Merge consecutive segments if they have same speaker and the gap is small.
    segments: list of (start, end, speaker) sorted by start.
    """
    if not segments:
        return []
    merged = [[segments[0][0], segments[0][1], segments[0][2]]]
    for s, e, spk in segments[1:]:
        ps, pe, pspk = merged[-1]
        if spk == pspk and (s - pe) <= merge_gap:
            merged[-1][1] = max(pe, e)
        else:
            merged.append([s, e, spk])
    return [(s, e, spk) for s, e, spk in merged]


def split_overlaps(segments):
    """
    Convert possibly-overlapping segments into a clean non-overlapping timeline
    by splitting earlier segments around later segments.

    Input: list of (start, end, speaker) sorted by start.
    Output: list of (start, end, speaker) sorted, non-overlapping.

    Rule:
      - If new segment overlaps the previous kept one:
          * cut the previous segment to end at new.start (prev_before)
          * keep new segment
          * if previous had tail after new.end, re-insert that tail (prev_after)
    """
    out = []
    for s, e, spk in segments:
        if not out:
            out.append([s, e, spk])
            continue

        ps, pe, pspk = out[-1]

        # No overlap
        if s >= pe:
            out.append([s, e, spk])
            continue

        # Overlap exists: s < pe
        # Case 1: New segment is fully inside previous (ps..pe)
        # Split previous into before + (new) + after.
        if s > ps:
            # keep "before" part of previous
            out[-1][1] = s
        else:
            # new starts before or exactly at previous start -> drop previous (it is covered)
            out.pop()

        # add new segment
        out.append([s, e, spk])

        # Add "after" tail of previous if it extends beyond new end
        if pe > e:
            out.append([e, pe, pspk])

        # Important: after inserting tail, there can be overlap with next segments too,
        # but since we're processing in start-time order and out tail starts at e,
        # the next segments will be handled similarly.

    # Remove any invalid or zero-length pieces
    out = [seg for seg in out if seg[1] > seg[0]]
    return [(s, e, spk) for s, e, spk in out]


def build_clean_segments(diarization_annotation):
    # Collect segments, drop tiny ones
    segs = []
    for turn, _, speaker in diarization_annotation.itertracks(yield_label=True):
        s = float(turn.start)
        e = float(turn.end)
        if (e - s) < MIN_SEG_DUR:
            continue
        segs.append((s, e, speaker))

    # Sort
    segs.sort(key=lambda x: x[0])

    # Merge same speaker (reduces fragmentation)
    segs = merge_same_speaker(segs, merge_gap=MERGE_GAP)

    # Split overlaps to create a clean timeline (no nested timestamps)
    segs = split_overlaps(segs)

    # Merge again (splitting can create adjacent same-speaker pieces)
    segs = merge_same_speaker(segs, merge_gap=0.01)

    return segs


# ---- Provide waveform directly to pyannote (bypasses torchcodec decoding) ----
full_wave = segment_to_float32(audio)
full_waveform = torch.from_numpy(full_wave).unsqueeze(0)  # (1, time)

output = pipeline({"waveform": full_waveform, "sample_rate": 16000})
diar = output.speaker_diarization  # pyannote 4.x

segments = build_clean_segments(diar)

with open(OUT_FILE, "w", encoding="utf-8") as f:
    for s, e, speaker in segments:
        start_ms = int(s * 1000)
        end_ms   = int(e * 1000)

        seg_audio = audio[start_ms:end_ms]
        waveform = segment_to_float32(seg_audio)

        # Whisper decoding settings for more stability
        result = model.transcribe(
            waveform,
            fp16=False,
            temperature=0.0,
            beam_size=5,
        )

        text = result.get("text", "").strip()
        if text:
            f.write(f"\n[{s:.2f}s -- {e:.2f}s] {speaker}: {text}")

        del seg_audio, waveform, result
        gc.collect()

print(f"Done. Wrote transcript to: {OUT_FILE}")
