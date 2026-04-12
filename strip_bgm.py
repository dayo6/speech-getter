import sys
import os
from audio_separator.separator import Separator

separator = Separator(output_format="wav")
separator.load_model()  # defaults to mel_band_roformer (SDR 11.4)
print("Model loaded.\n")


def strip_bgm(file_path):
    if not os.path.isfile(file_path):
        print(f"File not found: {file_path}")
        return

    base = os.path.splitext(file_path)[0]
    out_path = f"{base}_vocals.wav"
    output_dir = os.path.dirname(os.path.abspath(file_path))

    print(f"  Separating: {os.path.basename(file_path)}")
    # Set output_dir right before each call — it resets between runs
    separator.output_dir = output_dir
    output_files = separator.separate(os.path.abspath(file_path))

    # Find the vocals file and rename to our convention
    found = False
    for f in output_files:
        # output_files may be just filenames or full paths
        if "Vocals" in f or "vocals" in f:
            # Check multiple possible locations
            candidates = [
                f,
                os.path.join(output_dir, f),
                os.path.join(output_dir, os.path.basename(f)),
                os.path.join(".", os.path.basename(f)),
            ]
            for full in candidates:
                if os.path.exists(full):
                    os.replace(full, out_path)
                    print(f"  Saved: {out_path}")
                    found = True
                    break
            if found:
                break

    if not found:
        print(f"  WARNING: Could not find vocals output file")

    # Clean up instrumental file
    for f in output_files:
        if "Instrumental" in f or "instrumental" in f:
            candidates = [
                f,
                os.path.join(output_dir, f),
                os.path.join(output_dir, os.path.basename(f)),
                os.path.join(".", os.path.basename(f)),
            ]
            for full in candidates:
                if os.path.exists(full):
                    os.remove(full)
                    break

    return out_path


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python strip_bgm.py <file_or_run_folder>")
        raise SystemExit(1)

    target = sys.argv[1]
    if os.path.isdir(target):
        # Walk subfolders looking for CLIP_*.mp3 files to isolate vocals from
        clip_dirs = sorted(
            d for d in os.listdir(target)
            if os.path.isdir(os.path.join(target, d))
        )

        if clip_dirs:
            skipped = 0
            to_process = []
            for d in clip_dirs:
                clip_path = os.path.join(target, d)
                # Find CLIP_*.mp3 files that don't have a matching vocals version
                clips = sorted(
                    f for f in os.listdir(clip_path)
                    if f.startswith("CLIP_") and f.endswith(".mp3")
                    and "_FX" not in f and "_vocals" not in f
                )
                for c in clips:
                    vocals_name = c.replace(".mp3", "_vocals.wav")
                    if os.path.exists(os.path.join(clip_path, vocals_name)):
                        skipped += 1
                    else:
                        to_process.append((d, c))

            if skipped:
                print(f"Skipping {skipped} already processed clips.")

            if not to_process:
                print("Nothing to process.")
            else:
                for i, (d, c) in enumerate(to_process, 1):
                    print(f"\n[{i}/{len(to_process)}] {d}/{c}")
                    strip_bgm(os.path.join(target, d, c))
                print(f"\nDone! Processed {len(to_process)} clips.")
        else:
            # Fallback: treat as flat folder with audio files
            audio_exts = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".opus"}
            files = sorted(
                f for f in os.listdir(target)
                if os.path.splitext(f)[1].lower() in audio_exts
                and "_vocals" not in f
            )
            if not files:
                print(f"No audio files or clip subfolders found in {target}")
                raise SystemExit(1)
            for i, fname in enumerate(files, 1):
                vocals_name = os.path.splitext(fname)[0] + "_vocals.wav"
                if os.path.exists(os.path.join(target, vocals_name)):
                    continue
                print(f"\n[{i}/{len(files)}] {fname}")
                strip_bgm(os.path.join(target, fname))
    else:
        strip_bgm(target)
