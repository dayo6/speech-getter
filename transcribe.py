import sys
import os
import json
import torch
import whisperx
from dotenv import load_dotenv

load_dotenv()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
COMPUTE_TYPE = "float16" if DEVICE == "cuda" else "int8"
MODEL_SIZE = "large-v3"   # best accuracy; use "base" or "medium" for speed
HF_TOKEN = os.environ.get("HF_TOKEN")

# Load models once
print(f"Loading Whisper {MODEL_SIZE} on {DEVICE}...")
whisper_model = whisperx.load_model(MODEL_SIZE, DEVICE, compute_type=COMPUTE_TYPE)

print("Loading alignment model...")
align_model, align_metadata = whisperx.load_align_model(language_code="en", device=DEVICE)

diarize_model = None
if HF_TOKEN:
    print("Loading diarization model...")
    from whisperx.diarize import DiarizationPipeline, assign_word_speakers
    diarize_model = DiarizationPipeline(token=HF_TOKEN, device=DEVICE)
else:
    print("No HF_TOKEN — diarization disabled.")

print("Models loaded.\n")


def transcribe(file_path):
    if not os.path.isfile(file_path):
        print(f"File not found: {file_path}")
        raise SystemExit(1)

    print(f"  [1/4] Loading audio...")
    audio = whisperx.load_audio(file_path)

    print(f"  [2/4] Transcribing...")
    result = whisper_model.transcribe(audio, batch_size=16, language="en")

    print("  [3/4] Aligning words...")
    try:
        result = whisperx.align(
            result["segments"], align_model, align_metadata, audio, DEVICE,
            return_char_alignments=False
        )
    except ValueError as e:
        print(f"  Alignment skipped: {e}")

    if diarize_model:
        print("  [4/4] Identifying speakers...")
        diarize_segments = diarize_model(audio)
        result = assign_word_speakers(diarize_segments, result)
    else:
        print("  [4/4] Skipping diarization")

    # Build output
    segments = []
    for seg in result["segments"]:
        words = []
        for w in seg.get("words", []):
            words.append({
                "word": w["word"],
                "start": round(w.get("start", 0), 3),
                "end": round(w.get("end", 0), 3),
            })
        entry = {
            "start": round(seg.get("start", 0), 3),
            "end": round(seg.get("end", 0), 3),
            "text": seg.get("text", "").strip(),
            "words": words,
        }
        if "speaker" in seg:
            entry["speaker"] = seg["speaker"]
        segments.append(entry)

    # Build full transcript string
    full_text = " ".join(seg["text"] for seg in segments if seg["text"])

    output = {
        "source_file": os.path.basename(file_path),
        "language": result.get("language", "unknown"),
        "full_transcript": full_text,
        "segments": segments,
    }

    # Save JSON — put transcript.json in the same directory as the input file
    parent_dir = os.path.dirname(file_path)
    out_path = os.path.join(parent_dir, "transcript.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    # Print to terminal
    print(f"\n{'=' * 60}")
    print(f"  Transcript: {os.path.basename(file_path)}")
    print(f"  Language: {output['language']}")
    print(f"{'=' * 60}\n")

    for seg in segments:
        speaker = f" ({seg['speaker']})" if "speaker" in seg else ""
        print(f"[{seg['start']:.2f}s - {seg['end']:.2f}s]{speaker} {seg['text']}")
        for w in seg["words"]:
            print(f"    {w['start']:7.2f}s - {w['end']:7.2f}s  {w['word']}")
        print()

    print(f"Saved to: {out_path}")
    return output


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python transcribe.py <file_or_folder>")
        raise SystemExit(1)

    target = sys.argv[1]
    if os.path.isdir(target):
        # Walk subfolders looking for vocals.wav or audio.mp3 in each clip dir
        clip_dirs = sorted(
            d for d in os.listdir(target)
            if os.path.isdir(os.path.join(target, d))
        )

        if clip_dirs:
            skipped = 0
            to_process = []
            for d in clip_dirs:
                clip_path = os.path.join(target, d)
                transcript_path = os.path.join(clip_path, "transcript.json")
                if os.path.exists(transcript_path):
                    skipped += 1
                    continue
                # Prefer vocals.wav, fall back to audio.mp3
                audio = os.path.join(clip_path, "vocals.wav")
                if not os.path.exists(audio):
                    audio = os.path.join(clip_path, "audio.mp3")
                if os.path.exists(audio):
                    to_process.append((d, audio))

            if skipped:
                print(f"Skipping {skipped} already transcribed clips.")
            if not to_process:
                print("Nothing to transcribe.")
                raise SystemExit(0)

            print(f"Found {len(to_process)} clips to transcribe\n")
            for i, (d, audio) in enumerate(to_process, 1):
                print(f"\n[{i}/{len(to_process)}] {d}")
                transcribe(audio)

            print(f"\nDone! Transcribed {len(to_process)} clips.")
        else:
            # Fallback: flat folder with audio files
            audio_exts = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".opus"}
            files = sorted(
                f for f in os.listdir(target)
                if os.path.splitext(f)[1].lower() in audio_exts
            )
            for fname in files:
                transcribe(os.path.join(target, fname))
    else:
        transcribe(target)
