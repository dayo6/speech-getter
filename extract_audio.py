import os
import json
import argparse
import sys
from pydub import AudioSegment


def extract_clips_from_dir(clip_dir):
    """Extract audio clips from sample JSONs in a clip directory."""
    files = os.listdir(clip_dir)
    sample_files = sorted(f for f in files if f.startswith("sample_") and f.endswith(".json"))

    if not sample_files:
        return 0

    # Find audio file (prefer audio.mp3, fallback to any mp3)
    audio_path = os.path.join(clip_dir, "audio.mp3")
    if not os.path.exists(audio_path):
        mp3s = [f for f in files if f.endswith(".mp3") and not f.startswith("CLIP_")]
        if mp3s:
            audio_path = os.path.join(clip_dir, mp3s[0])
        else:
            print(f"  No audio file found in {clip_dir}")
            return 0

    audio = AudioSegment.from_mp3(audio_path)
    extracted = 0

    for j_file in sample_files:
        output_name = f"CLIP_{j_file.replace('.json', '.mp3')}"
        output_path = os.path.join(clip_dir, output_name)

        if os.path.exists(output_path):
            continue

        try:
            with open(os.path.join(clip_dir, j_file), "r") as f:
                data = json.load(f)

            start_ms = data.get("start", 0) * 1000
            end_ms = data.get("end", 0) * 1000

            if start_ms == 0 and end_ms == 0:
                print(f"  Skipping {j_file}: no timestamps")
                continue

            clip = audio[start_ms:end_ms]
            # Add silence tail so reverb/delay in FX chain can decay naturally
            tail_ms = 1500
            clip = clip + AudioSegment.silent(duration=tail_ms, frame_rate=clip.frame_rate)
            clip.export(output_path, format="mp3")
            print(f"  Exported: {output_name}")
            extracted += 1

        except Exception as e:
            print(f"  Error processing {j_file}: {e}")

    return extracted


def extract_clips(directory):
    """Walk subfolders and extract clips from each."""
    if not os.path.isdir(directory):
        print(f"Error: '{directory}' does not exist.")
        return

    # Check for per-clip subfolder structure
    clip_dirs = sorted(
        d for d in os.listdir(directory)
        if os.path.isdir(os.path.join(directory, d))
        and any(f.startswith("sample_") and f.endswith(".json") for f in os.listdir(os.path.join(directory, d)))
    )

    if clip_dirs:
        total = 0
        for d in clip_dirs:
            clip_path = os.path.join(directory, d)
            print(f"\n{d}/")
            count = extract_clips_from_dir(clip_path)
            total += count
        print(f"\nExtracted {total} clips from {len(clip_dirs)} folders.")
    else:
        # Fallback: flat directory (old structure)
        count = extract_clips_from_dir(directory)
        print(f"\nExtracted {count} clips.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audio Clip Extractor")
    parser.add_argument("--input", type=str, help="Path to run folder or clip folder", required=True)
    args = parser.parse_args()
    extract_clips(args.input)
