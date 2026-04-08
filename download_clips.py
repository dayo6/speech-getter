import sys
import os
import json
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import subprocess


def get_top_youtube_url(driver, query):
    driver.get("https://www.youtube.com/results?search_query=" + query.replace(" ", "+"))
    wait = WebDriverWait(driver, 10)
    first_video = wait.until(
        EC.presence_of_element_located((By.CSS_SELECTOR, "ytd-video-renderer a#video-title"))
    )
    return first_video.get_attribute("href")


def download_video(url, output_dir, filename):
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{filename}.%(ext)s")
    yt_dlp = os.path.join(os.path.dirname(sys.executable), "yt-dlp")
    subprocess.run([
        yt_dlp,
        "-x", "--audio-format", "mp3",
        "-o", output_path,
        "--no-playlist",
        url,
    ], check=True)


def main():
    if len(sys.argv) < 2:
        print("Usage: python download_clips.py <intros_XXXXXX.json>")
        raise SystemExit(1)

    json_path = sys.argv[1]
    with open(json_path, "r", encoding="utf-8") as f:
        suggestions = json.load(f)

    # Put clip subfolders in the same directory as the intros JSON
    output_dir = os.path.dirname(json_path) or "."

    # Set up headless Chrome
    opts = Options()
    opts.add_argument("--headless=new")
    opts.add_argument("--disable-gpu")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    driver = webdriver.Chrome(options=opts)

    results = []

    try:
        for i, s in enumerate(suggestions, 1):
            query = s.get("youtube_search", "")
            source = s.get("source_title", f"clip_{i}")
            vibe = s.get("vibe", "?")

            if not query:
                print(f"  {i:2}. SKIP — no youtube_search field")
                continue

            # Each clip gets its own subfolder
            safe_name = f"{i:02d}_{source}"
            safe_name = "".join(c if c.isalnum() or c in " _-" else "_" for c in safe_name)[:80]
            clip_dir = os.path.join(output_dir, safe_name)

            # Skip if already downloaded
            if os.path.exists(os.path.join(clip_dir, "audio.mp3")):
                print(f"  {i:2}. SKIP — already downloaded: {safe_name}")
                results.append({**s, "youtube_url": None, "filename": safe_name, "status": "ok"})
                continue

            print(f"\n  {i:2}. [{vibe}] {source}")
            print(f"      Searching: {query}")

            try:
                url = get_top_youtube_url(driver, query)
                print(f"      Found: {url}")

                print(f"      Downloading...")
                download_video(url, clip_dir, "audio")
                print(f"      Done! → {clip_dir}/")

                # Save clip metadata
                with open(os.path.join(clip_dir, "meta.json"), "w", encoding="utf-8") as f:
                    json.dump({**s, "youtube_url": url}, f, indent=2, ensure_ascii=False)

                results.append({**s, "youtube_url": url, "filename": safe_name, "status": "ok"})

            except Exception as e:
                print(f"      ERROR: {e}")
                results.append({**s, "youtube_url": None, "filename": None, "status": str(e)})

    finally:
        driver.quit()

    # Save results with URLs
    os.makedirs(output_dir, exist_ok=True)
    results_path = os.path.join(output_dir, "results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    ok = sum(1 for r in results if r["status"] == "ok")
    print(f"\n{'=' * 60}")
    print(f"  Downloaded {ok}/{len(suggestions)} clips to {output_dir}/")
    print(f"  Results saved to {results_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
