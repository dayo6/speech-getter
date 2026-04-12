"""
Apply FX chain to extracted CLIP_*.mp3 files to make them sound like pro beat intros.

FX chain:
  1. Lo-fi telephone filter (HP 400Hz + LP 4kHz)
  2. Saturation / bitcrush for grit
  3. Slapback delay (~50ms)
  4. Small room reverb (low wet)
  5. Vinyl crackle layer
  6. Normalize to -13.5 LUFS with -1.1dB true peak

Usage:
  python fx_chain.py <clip.mp3>
  python fx_chain.py <run_folder>
"""

import sys
import os
import numpy as np
import soundfile as sf
import pyloudnorm as pyln
from pedalboard import (
    Pedalboard,
    HighpassFilter,
    LowpassFilter,
    Distortion,
    Bitcrush,
    Delay,
    Reverb,
    Limiter,
    Gain,
)
from pedalboard.io import AudioFile

TARGET_LUFS = -13.5
TRUE_PEAK_DB = -1.0
SAMPLE_RATE = 44100


def load_audio(path):
    """Load audio file and return numpy array + sample rate."""
    with AudioFile(path) as f:
        audio = f.read(f.frames)
        sr = f.samplerate
    return audio, sr



def normalize_lufs(audio, sr, target_lufs=TARGET_LUFS, peak_db=TRUE_PEAK_DB):
    """Normalize audio to target LUFS with true peak limiter."""
    meter = pyln.Meter(sr)
    # pyloudnorm expects (samples, channels)
    audio_t = audio.T
    current_lufs = meter.integrated_loudness(audio_t)

    if np.isinf(current_lufs):
        return audio

    # Normalize to target LUFS
    audio_t = pyln.normalize.loudness(audio_t, current_lufs, target_lufs)
    audio = audio_t.T

    # Clip any samples that exceed [-1, 1] before limiting
    audio = np.clip(audio, -1.0, 1.0)

    # Apply limiter for true peak ceiling
    limiter = Pedalboard([Limiter(threshold_db=peak_db, release_ms=100)])
    audio = limiter(audio, sr)

    # Verify final LUFS and re-measure
    final_lufs = meter.integrated_loudness(audio.T)
    if not np.isinf(final_lufs) and abs(final_lufs - target_lufs) > 1.0:
        # Second pass if first normalization drifted too far
        audio_t = pyln.normalize.loudness(audio.T, final_lufs, target_lufs)
        audio = np.clip(audio_t.T, -1.0, 1.0)
        audio = limiter(audio, sr)

    return audio


def apply_fx(input_path, output_path=None):
    """Apply the full FX chain to a clip."""
    if output_path is None:
        base = os.path.splitext(input_path)[0]
        output_path = f"{base}_FX.mp3"

    if os.path.exists(output_path):
        return output_path

    print(f"  Processing: {os.path.basename(input_path)}")

    # Prefer vocals version if it exists
    vocals_path = input_path.replace(".mp3", "_vocals.wav")
    source_path = vocals_path if os.path.exists(vocals_path) else input_path

    # Load
    audio, sr = load_audio(source_path)

    # Resample if needed
    if sr != SAMPLE_RATE:
        from pedalboard import Resample
        resampler = Pedalboard([Resample(target_sample_rate=SAMPLE_RATE)])
        audio = resampler(audio, sr)
        sr = SAMPLE_RATE

    # FX Chain
    board = Pedalboard([
        # 1. Aggressive radio filter — tighter than full range, wider than telephone
        HighpassFilter(cutoff_frequency_hz=250),
        LowpassFilter(cutoff_frequency_hz=5000),

        # 2. Light saturation (soft clip, no bitcrush)
        Distortion(drive_db=5),

        # 3. Slapback delay
        Delay(delay_seconds=0.05, feedback=0.0, mix=0.15),

        # 4. Small room reverb
        Reverb(room_size=0.15, wet_level=0.08, dry_level=1.0, damping=0.7),
    ])
    audio = board(audio, sr)

    # 5. LUFS normalization + true peak limiting
    audio = normalize_lufs(audio, sr)

    # Export
    with AudioFile(output_path, "w", sr, num_channels=audio.shape[0]) as f:
        f.write(audio)

    print(f"  Saved: {os.path.basename(output_path)}")
    return output_path


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python fx_chain.py <clip.mp3 or run_folder>")
        raise SystemExit(1)

    target = sys.argv[1]

    if os.path.isdir(target):
        # Walk subfolders looking for CLIP_*.mp3 files
        clip_dirs = sorted(
            d for d in os.listdir(target)
            if os.path.isdir(os.path.join(target, d))
        )

        if clip_dirs:
            total = 0
            for d in clip_dirs:
                clip_path = os.path.join(target, d)
                clips = sorted(f for f in os.listdir(clip_path)
                              if f.startswith("CLIP_") and f.endswith(".mp3")
                              and "_FX" not in f)
                if clips:
                    print(f"\n{d}/")
                for c in clips:
                    apply_fx(os.path.join(clip_path, c))
                    total += 1
            print(f"\nProcessed {total} clips.")
        else:
            # Flat folder
            clips = sorted(f for f in os.listdir(target)
                          if f.startswith("CLIP_") and f.endswith(".mp3")
                          and "_FX" not in f)
            for c in clips:
                apply_fx(os.path.join(target, c))
            print(f"\nProcessed {len(clips)} clips.")
    else:
        apply_fx(target)
