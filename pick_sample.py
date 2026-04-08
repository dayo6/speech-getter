import sys
import os
import json
import requests


OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "gemma3:12b"


def format_transcript(transcript):
    """Format transcript into a clear readable format for the LLM."""
    lines = []
    current_speaker = None

    for seg in transcript["segments"]:
        speaker = seg.get("speaker", "Unknown")
        timestamp = f"[{seg['start']:.1f}s - {seg['end']:.1f}s]"

        if speaker != current_speaker:
            current_speaker = speaker
            lines.append(f"\n{speaker}:")

        lines.append(f"  {timestamp} \"{seg['text']}\"")

    return "\n".join(lines)


def pick_sample(transcript_path):
    with open(transcript_path, "r", encoding="utf-8") as f:
        transcript = json.load(f)

    formatted = format_transcript(transcript)
    source_file = transcript.get("source_file", os.path.basename(transcript_path))

    print(f"\n  Transcript preview:\n{formatted[:500]}...\n")

    prompt = f"""You are a sample picker for a dark west coast Chicano type beat producer.

Below is a transcript from: {source_file}
Each line shows the speaker, timestamp, and what they said.

{formatted}

Your job: find the single best ~20 second clip that would hit hardest as a spoken intro over a dark, aggressive hip hop beat.

Rules:
- Pick a continuous section, roughly 15-25 seconds long.
- It should work as a standalone moment — no context needed.
- Prefer lines that are menacing, raw, street, powerful, or iconic.
- If there are multiple speakers, you can include both if the exchange is fire.
- Give exact start and end timestamps from the transcript.

Respond ONLY with valid JSON, no markdown, no code fences:
{{
  "start": <start time in seconds>,
  "end": <end time in seconds>,
  "text": "the transcript text of the selected section",
  "why": "one sentence on why this sample hits hard"
}}"""

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL,
            "messages": [
                {"role": "system", "content": "You are a JSON API. Respond ONLY in valid JSON. No markdown, no explanation."},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
        },
        timeout=120,
    )

    if response.status_code != 200:
        print(f"Error {response.status_code}: {response.text}")
        raise SystemExit(1)

    content = response.json()["message"]["content"].strip()
    if content.startswith("```"):
        content = content.split("\n", 1)[1]
    if content.endswith("```"):
        content = content.rsplit("```", 1)[0]
    content = content.strip()

    pick = json.loads(content)

    # Save result
    base = os.path.splitext(transcript_path)[0].replace("_transcript", "")
    out_path = f"{base}_sample.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(pick, f, indent=2, ensure_ascii=False)

    print(f"{'=' * 60}")
    print(f"  Source: {source_file}")
    print(f"  Sample: {pick['start']:.2f}s - {pick['end']:.2f}s")
    print(f"  Text:   {pick['text']}")
    print(f"  Why:    {pick['why']}")
    print(f"{'=' * 60}")
    print(f"  Saved to: {out_path}\n")

    return pick


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python pick_sample.py <transcript.json or folder>")
        raise SystemExit(1)

    target = sys.argv[1]
    if os.path.isdir(target):
        files = sorted(
            f for f in os.listdir(target)
            if f.endswith("_transcript.json")
        )
        if not files:
            print(f"No transcript files found in {target}")
            raise SystemExit(1)
        print(f"Found {len(files)} transcripts in {target}\n")
        for i, fname in enumerate(files, 1):
            print(f"[{i}/{len(files)}] {fname}")
            pick_sample(os.path.join(target, fname))
    else:
        pick_sample(target)
