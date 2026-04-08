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

    # Separator outputs to current dir by default, we want same dir as input
    output_dir = os.path.dirname(file_path) or "."
    separator.output_dir = output_dir

    print(f"  Separating: {os.path.basename(file_path)}")
    output_files = separator.separate(file_path)

    # Find the vocals file and rename to our convention
    for f in output_files:
        if "Vocals" in f or "vocals" in f:
            full = os.path.join(output_dir, f) if not os.path.isabs(f) else f
            if os.path.exists(full) and full != out_path:
                os.replace(full, out_path)
            print(f"  Saved: {out_path}")
            break

    # Clean up instrumental file
    for f in output_files:
        if "Instrumental" in f or "instrumental" in f:
            full = os.path.join(output_dir, f) if not os.path.isabs(f) else f
            if os.path.exists(full):
                os.remove(full)

    return out_path


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python strip_bgm.py <file_or_run_folder>")
        raise SystemExit(1)

    target = sys.argv[1]
    if os.path.isdir(target):
        # Walk subfolders looking for audio.mp3 in each clip dir
        clip_dirs = sorted(
            d for d in os.listdir(target)
            if os.path.isdir(os.path.join(target, d))
            and os.path.exists(os.path.join(target, d, "audio.mp3"))
        )

        if not clip_dirs:
            # Fallback: treat as flat folder with audio files
            audio_exts = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".opus"}
            files = sorted(
                f for f in os.listdir(target)
                if os.path.splitext(f)[1].lower() in audio_exts
                and not f.endswith("_vocals.wav") and f != "vocals.wav"
            )
            if not files:
                print(f"No audio files or clip subfolders found in {target}")
                raise SystemExit(1)
            for i, fname in enumerate(files, 1):
                base = os.path.splitext(fname)[0]
                if os.path.exists(os.path.join(target, f"{base}_vocals.wav")):
                    continue
                print(f"\n[{i}/{len(files)}] {fname}")
                strip_bgm(os.path.join(target, fname))
        else:
            skipped = 0
            to_process = []
            for d in clip_dirs:
                vocals_path = os.path.join(target, d, "vocals.wav")
                if os.path.exists(vocals_path):
                    skipped += 1
                else:
                    to_process.append(d)

            if skipped:
                print(f"Skipping {skipped} already processed clips.")

            for i, d in enumerate(to_process, 1):
                print(f"\n[{i}/{len(to_process)}] {d}")
                strip_bgm(os.path.join(target, d, "audio.mp3"))

            print(f"\nDone! Processed {len(to_process)} clips.")
    else:
        strip_bgm(target)
