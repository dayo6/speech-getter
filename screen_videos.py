"""
Pre-screen YouTube videos by fetching auto-captions and asking an LLM
which ones contain good sample material — before downloading any audio.

Usage:
  python screen_videos.py <intros.json> [--output screened.json] [--max-per-query 20]
  python screen_videos.py runs/20260412/intros.json
"""

import sys
import os
import json
import re
import time
import subprocess
import argparse
import requests
from dotenv import load_dotenv

load_dotenv()

YT_DLP = os.path.join(os.path.dirname(sys.executable), "yt-dlp")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "deepseek/deepseek-v3.2"
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    OPENROUTER_API_KEY = input("Enter your OpenRouter API key: ").strip()

# Load download archive to skip already-downloaded videos
ARCHIVE_PATH = os.path.join(os.path.dirname(__file__) or ".", "download_archive.txt")


def load_archive():
    if not os.path.exists(ARCHIVE_PATH):
        return set()
    with open(ARCHIVE_PATH, "r") as f:
        return set(line.strip() for line in f if line.strip())


def search_youtube(query, max_results=20):
    """Search YouTube and return video IDs, titles, durations without downloading."""
    cmd = [
        YT_DLP,
        f"ytsearch{max_results}:{query}",
        "--flat-playlist",
        "--print", "%(id)s\t%(title)s\t%(duration)s",
        "--cookies-from-browser", "firefox",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if result.returncode != 0:
        return []

    videos = []
    for line in result.stdout.strip().split("\n"):
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) >= 3:
            videos.append({
                "id": parts[0],
                "title": parts[1],
                "duration": int(float(parts[2])) if parts[2] and parts[2] != "NA" else 0,
            })
        elif len(parts) >= 2:
            videos.append({"id": parts[0], "title": parts[1], "duration": 0})
    return videos


def fetch_transcript(video_id):
    """Fetch YouTube auto-captions via yt-dlp (uses Firefox cookies). Returns text or None."""
    import tempfile
    url = f"https://www.youtube.com/watch?v={video_id}"
    with tempfile.TemporaryDirectory() as tmpdir:
        cmd = [
            YT_DLP,
            "--write-auto-sub", "--sub-lang", "en",
            "--skip-download",
            "--cookies-from-browser", "firefox",
            "-o", os.path.join(tmpdir, "sub"),
            url,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

        # Find the subtitle file (could be .vtt, .srt, etc.)
        sub_files = [f for f in os.listdir(tmpdir) if f.endswith((".vtt", ".srt", ".json3"))]
        if not sub_files:
            return None

        sub_path = os.path.join(tmpdir, sub_files[0])
        with open(sub_path, "r", encoding="utf-8") as f:
            raw = f.read()

        # Parse VTT/SRT — strip timestamps and formatting, keep just text
        import re
        # Remove VTT header
        raw = re.sub(r'WEBVTT.*?\n\n', '', raw, flags=re.DOTALL)
        # Remove timestamp lines
        raw = re.sub(r'\d{2}:\d{2}:\d{2}[.,]\d{3}\s*-->\s*\d{2}:\d{2}:\d{2}[.,]\d{3}.*\n', '', raw)
        # Remove position/alignment tags
        raw = re.sub(r'<[^>]+>', '', raw)
        # Remove sequence numbers
        raw = re.sub(r'^\d+\s*$', '', raw, flags=re.MULTILINE)
        # Collapse whitespace
        text = " ".join(raw.split()).strip()

        # Remove duplicate phrases (YouTube auto-subs repeat lines)
        words = text.split()
        if len(words) > 10:
            deduped = []
            prev_chunk = ""
            for i in range(0, len(words), 5):
                chunk = " ".join(words[i:i+5])
                if chunk != prev_chunk:
                    deduped.append(chunk)
                prev_chunk = chunk
            text = " ".join(deduped)

        return text if len(text.split()) > 20 else None


def screen_batch(videos_with_transcripts):
    """Ask LLM which videos from a batch have good sample material.
    Returns list of video IDs that passed screening."""

    if not videos_with_transcripts:
        return []

    # Build prompt with all transcripts
    entries = []
    for v in videos_with_transcripts:
        # Truncate long transcripts to ~500 words
        words = v["transcript"].split()
        truncated = " ".join(words[:500])
        entries.append(f'VIDEO_ID: {v["id"]}\nTITLE: {v["title"]}\nTRANSCRIPT: {truncated}\n')

    batch_text = "\n---\n".join(entries)

    prompt = f"""You are screening YouTube videos for a beat producer looking for raw speech samples.

Below are transcripts from {len(videos_with_transcripts)} YouTube videos.
For each one, decide if it contains speech that would work as a dark, aggressive beat intro.

WHAT I'M LOOKING FOR:
- Raw, unfiltered speech about: violence, street life, confrontations, threats, hustle, paranoia
- Chicano/Sureno/Latino vibes are PREFERRED — barrio talk, varrio politics, Mexican-American street culture
- Anti-cop energy — disrespecting police, going off on law enforcement, "fuck 12" attitude
- Dissing rival gangs, mocking opps, talking down on enemies
- Heated moments, arguments, someone going off, real talk with conviction
- NOT music, NOT singing, NOT scripted content, NOT generic commentary
- REJECT videos where cops are shown in a positive/heroic light
- REJECT videos that are pro-law-enforcement or sympathetic to police

For each video, respond YES or NO and give a one-line reason.

{batch_text}

Respond ONLY with a valid JSON array. No markdown, no code fences:
[
  {{"id": "VIDEO_ID", "verdict": "YES or NO", "reason": "why"}}
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
                {"role": "system", "content": "You are a JSON API. Respond ONLY in valid JSON arrays."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.3,
            "max_tokens": 4000,
            "provider": {"ignore": ["Azure"]},
        },
        timeout=120,
    )

    if response.status_code != 200:
        print(f"    LLM error {response.status_code}: {response.text[:200]}")
        return [v["id"] for v in videos_with_transcripts]  # pass all on error

    content = response.json()["choices"][0]["message"]["content"].strip()

    # Parse response
    if "<think>" in content:
        content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
    content = content.replace('\u2018', "'").replace('\u2019', "'")
    content = content.replace('\u201c', '"').replace('\u201d', '"')
    content = re.sub(r'[^\x00-\x7F]+', '', content)
    content = content.strip()
    if content.startswith("```"):
        content = content.split("\n", 1)[1]
    if content.endswith("```"):
        content = content.rsplit("```", 1)[0]
    content = content.strip()
    if not content.startswith("["):
        start = content.find("[")
        if start != -1:
            content = content[start:]
    if not content.endswith("]"):
        end = content.rfind("]")
        if end != -1:
            content = content[:end + 1]

    try:
        verdicts = json.loads(content)
    except json.JSONDecodeError:
        print(f"    Failed to parse LLM response, passing all")
        return [v["id"] for v in videos_with_transcripts]

    passed = []
    for v in verdicts:
        vid_id = v.get("id", "")
        verdict = v.get("verdict", "NO").upper()
        reason = v.get("reason", "")
        if verdict == "YES":
            passed.append(vid_id)
            print(f"      YES: {vid_id} — {reason[:60]}")
        else:
            print(f"      no:  {vid_id} — {reason[:60]}")

    return passed


def main():
    parser = argparse.ArgumentParser(description="Pre-screen YouTube videos for sample material")
    parser.add_argument("intros_json", help="Path to intros.json with search queries")
    parser.add_argument("--output", "-o", default=None, help="Output path for screened.json")
    parser.add_argument("--max-per-query", type=int, default=10, help="Max YouTube results per query")
    parser.add_argument("--batch-size", type=int, default=10, help="Videos per LLM screening batch")
    args = parser.parse_args()

    with open(args.intros_json, "r", encoding="utf-8") as f:
        queries = json.load(f)

    output_path = args.output or os.path.join(os.path.dirname(args.intros_json) or ".", "screened.json")
    archive = load_archive()

    all_candidates = []
    total_searched = 0
    total_with_transcript = 0
    total_passed = 0

    for qi, q in enumerate(queries, 1):
        query = q.get("youtube_search", "")
        if not query:
            continue

        print(f"\n[{qi}/{len(queries)}] Searching: {query}")

        # Step 1: Search YouTube
        videos = search_youtube(query, max_results=args.max_per_query)
        time.sleep(10)  # delay after YouTube search
        # Filter out archived and bad duration
        videos = [v for v in videos if v["id"] not in archive and 30 < v.get("duration", 0) < 7200]
        # Also skip videos already in our candidates
        existing_ids = {c["id"] for c in all_candidates}
        videos = [v for v in videos if v["id"] not in existing_ids]

        print(f"  Found {len(videos)} new videos (after filtering)")
        total_searched += len(videos)

        if not videos:
            continue

        # Step 2: Fetch transcripts
        videos_with_transcripts = []
        for v in videos:
            transcript = fetch_transcript(v["id"])
            if transcript and len(transcript.split()) > 30:
                v["transcript"] = transcript
                videos_with_transcripts.append(v)
            time.sleep(10)  # YouTube rate-limits aggressively

        print(f"  Got transcripts for {len(videos_with_transcripts)}/{len(videos)} videos")
        total_with_transcript += len(videos_with_transcripts)

        if not videos_with_transcripts:
            continue

        # Step 3: LLM screening in batches
        print(f"  Screening with LLM...")
        for batch_start in range(0, len(videos_with_transcripts), args.batch_size):
            batch = videos_with_transcripts[batch_start:batch_start + args.batch_size]
            passed_ids = screen_batch(batch)

            for v in batch:
                if v["id"] in passed_ids:
                    all_candidates.append({
                        "id": v["id"],
                        "title": v["title"],
                        "duration": v["duration"],
                        "url": f"https://www.youtube.com/watch?v={v['id']}",
                        "query": query,
                        "youtube_search": query,
                        "source_title": v["title"][:60],
                        "vibe": "DARK",
                    })
                    total_passed += 1

            time.sleep(1)  # delay between LLM calls

        # Save after each query so progress isn't lost on crash
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(all_candidates, f, indent=2, ensure_ascii=False)
        print(f"  Saved {len(all_candidates)} candidates so far to {output_path}")

    # Final save
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_candidates, f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 60}")
    print(f"  Searched: {total_searched} videos")
    print(f"  Had transcripts: {total_with_transcript}")
    print(f"  Passed screening: {total_passed}")
    print(f"  Saved to: {output_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
