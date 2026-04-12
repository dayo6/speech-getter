"""
Scrape speech intros from a YouTube channel's videos.

Downloads the first 60 seconds of each video, detects where speech ends
and music begins, crops to just the speech portion.

Usage:
  python scrape_intros.py <channel_url> [--output <folder>] [--limit <n>]
  python scrape_intros.py https://www.youtube.com/@danielwsp --limit 50
"""

import sys
import os
import json
import subprocess
import argparse
import time
from pydub import AudioSegment

# yt-dlp path
YT_DLP = os.path.join(os.path.dirname(sys.executable), "yt-dlp")


def get_video_urls(channel_url, limit=None):
    """Get all video URLs from a YouTube channel."""
    print(f"Fetching video list from {channel_url}...")
    cmd = [
        YT_DLP,
        "--flat-playlist",
        "--print", "%(id)s %(title)s",
        "--cookies-from-browser", "firefox",
        channel_url,
    ]
    if limit:
        cmd.extend(["--playlist-end", str(limit)])

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error fetching videos: {result.stderr}")
        return []

    videos = []
    for line in result.stdout.strip().split("\n"):
        if not line.strip():
            continue
        parts = line.split(" ", 1)
        vid_id = parts[0]
        title = parts[1] if len(parts) > 1 else vid_id
        videos.append({"id": vid_id, "title": title, "url": f"https://www.youtube.com/watch?v={vid_id}"})

    print(f"Found {len(videos)} videos")
    return videos


def download_first_minute(video_url, output_path):
    """Download only the first 60 seconds of a video as mp3."""
    cmd = [
        YT_DLP,
        "-x", "--audio-format", "mp3",
        "--download-sections", "*0-60",
        "--force-keyframes-at-cuts",
        "-o", output_path,
        "--no-playlist",
        "--cookies-from-browser", "firefox",
        video_url,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0


def detect_speech_region(audio_path):
    """Use inaSpeechSegmenter to find speech vs music segments.
    Returns (speech_start, speech_end) in milliseconds."""
    from inaSpeechSegmenter import Segmenter

    seg = Segmenter(vad_engine="smn", detect_gender=False)
    segments = seg(audio_path)

    # segments is list of (label, start, end) where label is 'speech', 'music', 'noise'
    speech_start = None
    speech_end = None

    for label, start, end in segments:
        if label == "speech":
            if speech_start is None:
                speech_start = start
            speech_end = end
        elif label == "music" and speech_end is not None:
            # Music started after speech — stop here
            break

    if speech_start is None:
        return None, None

    return int(speech_start * 1000), int(speech_end * 1000)


def crop_speech(input_path, output_path, start_ms, end_ms):
    """Crop audio to just the speech region."""
    audio = AudioSegment.from_mp3(input_path)
    # Add small padding (200ms before, 100ms after)
    start_ms = max(0, start_ms - 200)
    end_ms = min(len(audio), end_ms + 100)
    clip = audio[start_ms:end_ms]
    clip.export(output_path, format="mp3")
    return len(clip) / 1000  # duration in seconds


def main():
    parser = argparse.ArgumentParser(description="Scrape speech intros from YouTube channel")
    parser.add_argument("channel_url", help="YouTube channel URL")
    parser.add_argument("--output", "-o", default="scraped_intros", help="Output folder")
    parser.add_argument("--limit", "-n", type=int, default=None, help="Max videos to process")
    parser.add_argument("--min-speech", type=float, default=5.0, help="Min speech duration in seconds")
    parser.add_argument("--max-speech", type=float, default=45.0, help="Max speech duration in seconds")
    parser.add_argument("--delay", type=float, default=3.0, help="Seconds to wait between downloads (default 3)")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # Get video list
    videos = get_video_urls(args.channel_url, limit=args.limit)
    if not videos:
        print("No videos found.")
        return

    # Save video list
    with open(os.path.join(args.output, "videos.json"), "w", encoding="utf-8") as f:
        json.dump(videos, f, indent=2, ensure_ascii=False)

    results = []
    skipped = 0

    for i, v in enumerate(videos, 1):
        safe_title = "".join(c if c.isalnum() or c in " _-" else "_" for c in v["title"])[:60]
        vid_dir = os.path.join(args.output, f"{i:03d}_{safe_title}")

        # Skip if already processed
        speech_path = os.path.join(vid_dir, "speech.mp3")
        if os.path.exists(speech_path):
            skipped += 1
            continue

        os.makedirs(vid_dir, exist_ok=True)
        raw_path = os.path.join(vid_dir, "raw_60s.mp3")

        print(f"\n[{i}/{len(videos)}] {v['title']}")

        # Step 1: Download first 60 seconds
        if not os.path.exists(raw_path):
            print(f"  Downloading first 60s...")
            if not download_first_minute(v["url"], raw_path):
                print(f"  FAILED to download")
                continue
            time.sleep(max(args.delay, 10))  # minimum 10s between YouTube requests

        if not os.path.exists(raw_path):
            # yt-dlp might append format extension
            candidates = [f for f in os.listdir(vid_dir) if f.startswith("raw_60s") and f.endswith(".mp3")]
            if candidates:
                raw_path = os.path.join(vid_dir, candidates[0])
            else:
                print(f"  No audio file found after download")
                continue

        # Step 2: Detect speech region
        print(f"  Detecting speech vs music...")
        try:
            start_ms, end_ms = detect_speech_region(raw_path)
        except Exception as e:
            print(f"  Speech detection failed: {e}")
            continue

        if start_ms is None:
            print(f"  No speech detected — skipping")
            continue

        duration_s = (end_ms - start_ms) / 1000
        print(f"  Speech: {start_ms/1000:.1f}s - {end_ms/1000:.1f}s ({duration_s:.1f}s)")

        # Step 3: Filter by duration
        if duration_s < args.min_speech:
            print(f"  Too short ({duration_s:.1f}s < {args.min_speech}s) — skipping")
            continue
        if duration_s > args.max_speech:
            print(f"  Too long ({duration_s:.1f}s > {args.max_speech}s) — trimming to {args.max_speech}s")
            end_ms = start_ms + int(args.max_speech * 1000)

        # Step 4: Crop to speech
        final_duration = crop_speech(raw_path, speech_path, start_ms, end_ms)
        print(f"  Saved: speech.mp3 ({final_duration:.1f}s)")

        # Save metadata
        meta = {
            "video_id": v["id"],
            "title": v["title"],
            "url": v["url"],
            "speech_start_s": start_ms / 1000,
            "speech_end_s": end_ms / 1000,
            "duration_s": final_duration,
        }
        with open(os.path.join(vid_dir, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        results.append(meta)

    # Summary
    print(f"\n{'=' * 60}")
    print(f"  Processed: {len(results)} speeches")
    print(f"  Skipped: {skipped} already done")
    print(f"  Output: {args.output}/")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
