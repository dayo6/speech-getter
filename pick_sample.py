import sys
import os
import json
import re
import difflib
import requests
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "qwen/qwq-32b"
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    OPENROUTER_API_KEY = input("Enter your OpenRouter API key: ").strip()


def format_transcript(transcript):
    """Format transcript into a clear readable format for the LLM (no timestamps)."""
    lines = []
    current_speaker = None

    for seg in transcript["segments"]:
        speaker = seg.get("speaker", "Unknown")

        if speaker != current_speaker:
            current_speaker = speaker
            lines.append(f"\n{speaker}:")

        lines.append(f"  \"{seg['text']}\"")

    return "\n".join(lines)


def resolve_timestamps(pick_text, transcript):
    """Fuzzy-match LLM pick text against whisper transcript to get accurate timestamps."""
    # Build word list with timestamps
    all_words = []
    for seg in transcript["segments"]:
        for w in seg.get("words", []):
            if "start" in w and "end" in w:
                all_words.append(w)

    if not all_words:
        return None, None

    # Build full text and track character-to-word-index mapping
    full_text = ""
    char_to_word = []
    for i, w in enumerate(all_words):
        if full_text:
            full_text += " "
            char_to_word.append(i)  # space maps to next word
        for _ in w["word"]:
            char_to_word.append(i)
            full_text += ""
        full_text = full_text[:len(char_to_word)]  # keep in sync

    # Rebuild properly — simple approach
    full_text = ""
    char_to_word = []
    for i, w in enumerate(all_words):
        if full_text:
            full_text += " "
            char_to_word.append(i)
        word_str = w["word"]
        for _ in word_str:
            char_to_word.append(i)
        full_text += word_str

    full_lower = full_text.lower()
    pick_lower = pick_text.strip().lower()

    # Try exact substring match first
    idx = full_lower.find(pick_lower)
    if idx != -1:
        first_word = char_to_word[idx]
        last_word = char_to_word[min(idx + len(pick_lower) - 1, len(char_to_word) - 1)]
        return all_words[first_word]["start"], all_words[last_word]["end"]

    # Fuzzy match — find best matching window
    matcher = difflib.SequenceMatcher(None, full_lower, pick_lower, autojunk=False)
    blocks = matcher.get_matching_blocks()
    if not blocks or blocks[0].size == 0:
        return None, None

    # Find the longest matching block and expand around it
    best = max(blocks, key=lambda b: b.size)
    start_char = best.a
    # Estimate end based on pick length
    end_char = min(start_char + len(pick_lower) - 1, len(char_to_word) - 1)

    first_word = char_to_word[start_char]
    last_word = char_to_word[end_char]
    return all_words[first_word]["start"], all_words[last_word]["end"]


def pick_samples(transcript_path):
    with open(transcript_path, "r", encoding="utf-8") as f:
        transcript = json.load(f)

    formatted = format_transcript(transcript)
    source_file = transcript.get("source_file", os.path.basename(transcript_path))

    # Scale max picks based on clip duration
    segments = transcript.get("segments", [])
    if segments:
        clip_duration = segments[-1]["end"] - segments[0]["start"]
    else:
        clip_duration = 0
    SECONDS_PER_PICK = 300  # 1 pick per 5 minutes
    max_picks = max(2, int(clip_duration / SECONDS_PER_PICK))

    print(f"\n  Transcript preview:\n{formatted[:500]}...")
    print(f"  Clip duration: {clip_duration:.0f}s — picking {max_picks} samples\n")

    prompt = f"""You are a sample picker for a dark west coast Chicano type beat producer.

Below is a transcript. Each line shows the speaker and what they said.

{formatted}

You MUST find exactly {max_picks} samples from it. Spread your picks across the full transcript — don't cluster them all in one section.

What makes a great beat intro sample:

TONE — Raw street-level authenticity, NOT cinematic polish:
- Should sound like a voice note, a studio conversation, or someone talking real shit to their people.
- Loose, conversational grammar. Verbal fillers that add rhythm ("you feel me", "straight up", "on God") are GOOD.
- Unscripted energy — plain-spoken but heavy truths. No movie-script dialogue.

TWO STYLES TO LOOK FOR:
1. "Daily Reminder" (Street Wisdom) — direct real-talk advice about the game, loyalty, haters, survival.
   Short punchy statements followed by a confirmation ("keep winning", "you feel me?", "that's on everything").
2. "Street Anecdote" (Storytelling) — first-person story or philosophy of violence/power/hustle.
   Raw, blunt, ends on a high-energy mic-drop moment.

STRUCTURE:
- 2-5 sentences max. Short, punchy, staccato syntax — no complex conjunctions.
- Prefer words with heavy hard consonants (D, K, T, P) — they hit harder on a beat.
- Start in the middle of a thought — no introductory fluff, no "I think", no "let me tell you".

THEME:
- Ambition, inevitability, betrayal, the cost of power, silence over noise, the hunt.
- NO jokes, puns, cliches about "working hard", or upbeat energy.
- Must work as a standalone moment — zero context needed.

Rules:
- Each sample should be 2-5 sentences long and succinct.
- Each pick MUST work as a standalone moment — if it needs context to make sense, skip it.
- Prefer a single speaker.
- Quote the text EXACTLY as it appears in the transcript — do not paraphrase or reword.
- Rate each sample out of 10.0 based on how hard it would hit as a beat intro.
- You should return {max_picks} or more, if there are enough quotes.
- The samples should NOT overlap — pick different moments.

Respond ONLY with a valid JSON array, no markdown, no code fences:
[
  {{
    "rating": <1.0-10.0>,
    "text": "the transcript text of the selected section",
    "why": "one sentence on why this sample hits hard"
  }}
]"""

    response = requests.post(
        OPENROUTER_URL,
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": MODEL,
            "messages": [
                {"role": "system", "content": "You are a JSON API. Respond ONLY in valid JSON arrays. No markdown, no explanation."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.6,
            "max_tokens": 16000,
        },
        timeout=300,
    )

    if response.status_code != 200:
        print(f"Error {response.status_code}: {response.text}")
        raise SystemExit(1)

    content = response.json()["choices"][0]["message"]["content"].strip()

    # Strip R1 reasoning block
    if "<think>" in content:
        content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)

    # Strip non-ASCII characters (R1 sometimes injects Chinese chars)
    content = re.sub(r'[^\x00-\x7F]+', '', content)

    # Strip markdown code fences
    content = content.strip()
    if content.startswith("```"):
        content = content.split("\n", 1)[1]
    if content.endswith("```"):
        content = content.rsplit("```", 1)[0]
    content = content.strip()

    # Try to extract JSON array
    if not content.startswith("["):
        start = content.find("[")
        if start != -1:
            content = content[start:]
    if not content.endswith("]"):
        end = content.rfind("]")
        if end != -1:
            content = content[:end + 1]

    try:
        picks = json.loads(content)
    except json.JSONDecodeError:
        print(f"  SKIPPED — bad JSON response: {content[:200]}")
        return None

    if not isinstance(picks, list):
        picks = [picks]

    # Resolve timestamps from whisper transcript
    for pick in picks:
        start, end = resolve_timestamps(pick.get("text", ""), transcript)
        if start is not None:
            pick["start"] = round(start, 3)
            pick["end"] = round(end, 3)
        else:
            print(f"  WARNING: Could not match text: \"{pick.get('text', '')[:60]}...\"")

    # Remove picks with no timestamps
    picks = [p for p in picks if "start" in p and "end" in p]

    # Sort by rating (best first)
    picks.sort(key=lambda x: x.get("rating", 0), reverse=True)

    # Save results in same directory as transcript
    clip_dir = os.path.dirname(transcript_path)
    for i, pick in enumerate(picks, 1):
        rating = pick.get("rating", 0)
        out_path = os.path.join(clip_dir, f"sample_{i}_{rating}of10.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(pick, f, indent=2, ensure_ascii=False)

    # Print summary
    print(f"{'=' * 60}")
    print(f"  Source: {source_file}")
    for i, pick in enumerate(picks, 1):
        rating = pick.get("rating", "?")
        duration = pick["end"] - pick["start"]
        print(f"\n  #{i} [{rating}/10] ({duration:.1f}s)")
        print(f"  {pick['start']:.2f}s - {pick['end']:.2f}s")
        print(f"  \"{pick['text']}\"")
        print(f"  Why: {pick['why']}")
    print(f"\n{'=' * 60}")
    print(f"  Saved {len(picks)} samples for {source_file}\n")

    return picks


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python pick_sample.py <transcript.json or folder>")
        raise SystemExit(1)

    target = sys.argv[1]
    if os.path.isdir(target):
        # Walk subfolders looking for transcript.json in each clip dir
        clip_dirs = sorted(
            d for d in os.listdir(target)
            if os.path.isdir(os.path.join(target, d))
            and os.path.exists(os.path.join(target, d, "transcript.json"))
        )

        if not clip_dirs:
            # Fallback: flat folder with *_transcript.json files
            files = sorted(f for f in os.listdir(target) if f.endswith("_transcript.json"))
            if not files:
                print(f"No transcript files found in {target}")
                raise SystemExit(1)
            for i, fname in enumerate(files, 1):
                print(f"[{i}/{len(files)}] {fname}")
                pick_samples(os.path.join(target, fname))
        else:
            skipped = 0
            to_process = []
            for d in clip_dirs:
                clip_path = os.path.join(target, d)
                existing = [f for f in os.listdir(clip_path) if f.startswith("sample_")]
                if existing:
                    skipped += 1
                else:
                    to_process.append(d)

            if skipped:
                print(f"Skipping {skipped} already processed clips.")

            for i, d in enumerate(to_process, 1):
                print(f"[{i}/{len(to_process)}] {d}")
                pick_samples(os.path.join(target, d, "transcript.json"))

            print(f"\nDone! Processed {len(to_process)} clips.")
    else:
        pick_samples(target)
