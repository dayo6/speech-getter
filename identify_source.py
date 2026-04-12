"""
Identify the original source of speech samples by transcribing and searching.

Transcribes audio clips with Whisper, extracts distinctive phrases,
searches Google for exact quotes, and saves the likely source.

Usage:
  python identify_source.py <audio_file_or_folder>
  python identify_source.py runs/20260412_014748
  python identify_source.py scraped_intros/
"""

import sys
import os
import json
import time
import whisperx
import torch
from ddgs import DDGS

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
COMPUTE_TYPE = "float16" if DEVICE == "cuda" else "int8"

# Load whisper model once
print(f"Loading Whisper on {DEVICE}...")
model = whisperx.load_model("large-v3", DEVICE, compute_type=COMPUTE_TYPE)
print("Model loaded.\n")


def transcribe_clip(audio_path):
    """Transcribe an audio clip and return the full text."""
    audio = whisperx.load_audio(audio_path)
    result = model.transcribe(audio, batch_size=16, language="en")
    text = " ".join(seg.get("text", "").strip() for seg in result.get("segments", []))
    return text.strip()


STOPWORDS = {"i", "you", "he", "she", "it", "we", "they", "the", "a", "an",
             "is", "was", "are", "were", "be", "been", "do", "did", "have",
             "had", "will", "would", "could", "should", "can", "may", "might",
             "to", "of", "in", "for", "on", "with", "at", "by", "from", "as",
             "into", "that", "this", "but", "and", "or", "not", "no", "so",
             "if", "then", "than", "just", "like", "know", "get", "got",
             "gonna", "gotta", "wanna", "about", "up", "out", "all", "my",
             "your", "his", "her", "our", "their", "me", "him", "them",
             "what", "when", "where", "how", "who", "which", "there", "here",
             "yeah", "oh", "okay", "right", "well", "man", "yo", "uh", "um"}


def score_sentence(sentence):
    """Score a sentence by how distinctive/searchable it is."""
    words = sentence.split()
    if len(words) < 3:
        return 0
    score = len(words)  # longer = better base
    for w in words:
        lower = w.lower().strip(".,!?'\"")
        if lower not in STOPWORDS:
            score += 2  # non-stopword bonus
        if len(w) > 1 and w[0].isupper():  # proper noun
            score += 5
        if any(c.isdigit() for c in w):  # numbers/years
            score += 3
    return score


def extract_phrases(text, num_phrases=8):
    """Extract the most distinctive sentences for searching.
    Splits into sentences, scores by searchability, returns the best ones."""
    import re

    # Split into sentences
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip() and len(s.strip().split()) >= 3]

    if not sentences:
        # Fallback: split by commas or just use the whole text
        sentences = [s.strip() for s in text.split(",") if len(s.strip().split()) >= 3]
    if not sentences:
        return [text] if text else []

    # Score and sort by distinctiveness
    scored = [(score_sentence(s), s) for s in sentences]
    scored.sort(key=lambda x: x[0], reverse=True)

    # Take top sentences, deduplicate
    phrases = []
    seen = set()
    for score, sentence in scored:
        # For QuoDB: trim to 6-8 words (middle of sentence is often most unique)
        words = sentence.split()
        if len(words) > 8:
            # Take the middle chunk — usually the most distinctive part
            mid = len(words) // 2
            short = " ".join(words[max(0, mid - 3):mid + 4])
        else:
            short = sentence

        if short not in seen:
            seen.add(short)
            phrases.append(short)

        # Also add full sentence if different and long enough
        if sentence not in seen and len(words) > 6:
            seen.add(sentence)
            phrases.append(sentence)

        if len(phrases) >= num_phrases:
            break

    return phrases[:num_phrases]


def search_ddg(phrase, num_results=5):
    """Search DuckDuckGo for an exact phrase and return results with titles."""
    query = f'"{phrase}"'
    results = []
    try:
        hits = DDGS().text(query, max_results=num_results)
        for h in hits:
            results.append({
                "source": "ddg",
                "title": h.get("title", ""),
                "url": h.get("href", ""),
                "snippet": h.get("body", ""),
            })
    except Exception as e:
        print(f"    DDG error: {e}")
    return results


def search_youtube(phrase, num_results=5):
    """Search YouTube captions via DuckDuckGo with exact quote matching."""
    results = []
    try:
        # Exact quote in quotes ��� DuckDuckGo indexes YouTube captions
        query = f'site:youtube.com "{phrase}"'
        hits = DDGS().text(query, max_results=num_results)
        for h in hits:
            url = h.get("href", "")
            title = h.get("title", "")
            snippet = h.get("body", "")
            if "youtube.com" in url or "youtu.be" in url:
                results.append({
                    "source": "youtube",
                    "title": title,
                    "url": url,
                    "snippet": snippet,
                })
    except Exception as e:
        print(f"    YouTube search error: {e}")
    return results


def identify(audio_path):
    """Full pipeline: transcribe -> search -> identify source."""
    # Check for cached transcript to avoid re-transcribing
    clip_dir = os.path.dirname(audio_path)
    transcript_cache = os.path.join(clip_dir, "transcript_cache.txt")
    if os.path.exists(transcript_cache):
        with open(transcript_cache, "r", encoding="utf-8") as f:
            text = f.read().strip()
        print(f"  Using cached transcript")
    else:
        print(f"  Transcribing...")
        text = transcribe_clip(audio_path)
        if text:
            with open(transcript_cache, "w", encoding="utf-8") as f:
                f.write(text)
    if not text:
        print(f"  No speech detected")
        return None

    print(f"  Text: \"{text[:100]}{'...' if len(text) > 100 else ''}\"")

    phrases = extract_phrases(text)
    print(f"  Searching {len(phrases)} phrases...")

    all_results = []
    from collections import Counter
    title_counts = Counter()  # count how many phrases match each YouTube video title

    for phrase in phrases:
        print(f"    Searching: \"{phrase}\"")

        # Search YouTube for the original video
        yt_hits = search_youtube(phrase)
        for h in yt_hits:
            h["query"] = phrase
            all_results.append(h)
            title_counts[h["title"]] += 1
            print(f"      YT: {h['title'][:70]}")

        # Also search DDG broadly
        ddg_hits = search_ddg(phrase)
        for h in ddg_hits:
            h["query"] = phrase
            all_results.append(h)

        time.sleep(2)  # Delay between searches

    # The YouTube video title that appears for the most phrases is the likely source
    likely_source = ""
    likely_url = ""
    confidence = 0
    if title_counts:
        likely_source, confidence = title_counts.most_common(1)[0]
        # Find URL for top match
        for r in all_results:
            if r.get("title") == likely_source and r.get("source") == "youtube":
                likely_url = r.get("url", "")
                break
        print(f"\n  Top matches:")
        for title, count in title_counts.most_common(5):
            print(f"    {count}x  {title[:70]}")

    result = {
        "transcript": text,
        "search_queries": phrases,
        "results": all_results,
        "title_matches": dict(title_counts.most_common(10)),
        "likely_source": likely_source,
        "likely_url": likely_url,
        "confidence": confidence,
    }

    return result


def find_audio_files(target):
    """Find audio files to process. Handles run folders, scraped folders, or single files."""
    if os.path.isfile(target):
        return [(os.path.dirname(target) or ".", target)]

    pairs = []  # (output_dir, audio_path)

    # Walk subfolders
    for d in sorted(os.listdir(target)):
        dir_path = os.path.join(target, d)
        if not os.path.isdir(dir_path):
            continue

        # Skip if already identified
        if os.path.exists(os.path.join(dir_path, "source.json")):
            continue

        # Look for speech clips in priority order
        for pattern in ["speech.mp3", "CLIP_*_FX.mp3", "CLIP_*.mp3"]:
            import glob
            matches = sorted(glob.glob(os.path.join(dir_path, pattern)))
            if matches:
                # Use the first (best rated) clip
                pairs.append((dir_path, matches[0]))
                break

    return pairs


def main():
    if len(sys.argv) < 2:
        print("Usage: python identify_source.py <audio_file_or_folder>")
        raise SystemExit(1)

    target = sys.argv[1]
    pairs = find_audio_files(target)

    if not pairs:
        print("No audio files to process (or all already identified).")
        return

    print(f"Found {len(pairs)} clips to identify\n")

    for i, (out_dir, audio_path) in enumerate(pairs, 1):
        name = os.path.basename(out_dir)
        print(f"\n[{i}/{len(pairs)}] {name}")
        print(f"  File: {os.path.basename(audio_path)}")

        result = identify(audio_path)
        if not result:
            continue

        # Save
        out_path = os.path.join(out_dir, "source.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        print(f"  Likely source: {result['likely_source'] or 'unknown'}")
        print(f"  URLs found: {len(result['results'])}")
        print(f"  Saved: source.json")

    print(f"\n{'=' * 60}")
    print(f"  Identified {len(pairs)} clips")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
