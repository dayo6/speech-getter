import sys
import os
import json
import whisperx

DEVICE = "cuda"
COMPUTE_TYPE = "float16"  # faster on GPU
MODEL_SIZE = "large-v3"   # best accuracy; use "base" or "medium" for speed


def transcribe(file_path):
    if not os.path.isfile(file_path):
        print(f"File not found: {file_path}")
        raise SystemExit(1)

    print(f"[1/3] Loading audio from: {file_path}")
    audio = whisperx.load_audio(file_path)

    print(f"[2/3] Transcribing with Whisper {MODEL_SIZE}...")
    model = whisperx.load_model(MODEL_SIZE, DEVICE, compute_type=COMPUTE_TYPE)
    result = model.transcribe(audio, batch_size=16)

    print("[3/3] Aligning words...")
    align_model, metadata = whisperx.load_align_model(
        language_code=result["language"], device=DEVICE
    )
    result = whisperx.align(
        result["segments"], align_model, metadata, audio, DEVICE,
        return_char_alignments=False
    )

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
        segments.append({
            "start": round(seg.get("start", 0), 3),
            "end": round(seg.get("end", 0), 3),
            "text": seg.get("text", "").strip(),
            "words": words,
        })

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
        print(f"[{seg['start']:.2f}s - {seg['end']:.2f}s] {seg['text']}")
        for w in seg["words"]:
            print(f"    {w['start']:7.2f}s - {w['end']:7.2f}s  {w['word']}")
        print()

    print(f"Saved to: {out_path}")
    return output


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python transcribe.py <video_or_audio_file>")
        raise SystemExit(1)
    transcribe(sys.argv[1])
