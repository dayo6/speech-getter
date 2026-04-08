import sys
import os
import json
import torch
import whisperx

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
COMPUTE_TYPE = "float16" if DEVICE == "cuda" else "int8"
MODEL_SIZE = "large-v3"   # best accuracy; use "base" or "medium" for speed
HF_TOKEN = os.environ.get("HF_TOKEN")


def transcribe(file_path):
    if not os.path.isfile(file_path):
        print(f"File not found: {file_path}")
        raise SystemExit(1)

    print(f"[1/4] Loading audio from: {file_path}")
    audio = whisperx.load_audio(file_path)

    print(f"[2/4] Transcribing with Whisper {MODEL_SIZE}...")
    model = whisperx.load_model(MODEL_SIZE, DEVICE, compute_type=COMPUTE_TYPE)
    result = model.transcribe(audio, batch_size=16, language="en")

    print("[3/4] Aligning words...")
    try:
        align_model, metadata = whisperx.load_align_model(
            language_code=result["language"], device=DEVICE
        )
        result = whisperx.align(
            result["segments"], align_model, metadata, audio, DEVICE,
            return_char_alignments=False
        )
    except ValueError as e:
        print(f"  Alignment skipped: {e}")

    # Speaker diarization
    if HF_TOKEN:
        print("[4/4] Identifying speakers...")
        diarize_model = whisperx.DiarizationPipeline(use_auth_token=HF_TOKEN, device=DEVICE)
        diarize_segments = diarize_model(audio)
        result = whisperx.assign_word_speakers(diarize_segments, result)
    else:
        print("[4/4] Skipping diarization (no HF_TOKEN in .env)")

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

    output = {
        "source_file": os.path.basename(file_path),
        "language": result.get("language", "unknown"),
        "segments": segments,
    }

    # Save JSON
    base = os.path.splitext(file_path)[0]
    out_path = f"{base}_transcript.json"
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
        audio_exts = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".opus"}
        files = sorted(
            f for f in os.listdir(target)
            if os.path.splitext(f)[1].lower() in audio_exts
            and not f.endswith("_transcript.json")
        )
        if not files:
            print(f"No audio/video files found in {target}")
            raise SystemExit(1)
        print(f"Found {len(files)} files in {target}\n")
        skipped = 0
        for i, fname in enumerate(files, 1):
            base = os.path.splitext(fname)[0]
            transcript_path = os.path.join(target, f"{base}_transcript.json")
            if os.path.exists(transcript_path):
                skipped += 1
                continue
            print(f"\n[{i}/{len(files)}] {fname}")
            transcribe(os.path.join(target, fname))
        if skipped:
            print(f"\nSkipped {skipped} already transcribed files.")
    else:
        transcribe(target)
