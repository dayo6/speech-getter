"""
Full pipeline orchestrator:
  1. get_intros.py      — LLM suggests scenes/moments → runs/{timestamp}/intros.json
  2. download_clips.py  — YouTube search + download    → runs/{timestamp}/{clip_name}/audio.mp3
  3. strip_bgm.py       — Vocal isolation               → runs/{timestamp}/{clip_name}/vocals.wav
  4. transcribe.py      — Whisper transcription          → runs/{timestamp}/{clip_name}/transcript.json
  5. pick_sample.py     — LLM picks best quotes          → runs/{timestamp}/{clip_name}/sample_*.json
  6. extract_audio.py   — Slice audio clips              → runs/{timestamp}/{clip_name}/CLIP_*.mp3

Usage:
  python orchestrate.py                          # full pipeline from scratch
  python orchestrate.py --from 3                 # resume from step 3
  python orchestrate.py --run runs/20260408_130716  # use existing run folder
"""

import sys
import os
import subprocess
import shutil
from datetime import datetime
import argparse

PYTHON = sys.executable


def run_step(step_num, description, cmd):
    print(f"\n{'=' * 60}")
    print(f"  STEP {step_num}: {description}")
    print(f"  Running: {' '.join(cmd)}")
    print(f"{'=' * 60}\n")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"\n  STEP {step_num} FAILED (exit code {result.returncode})")
        raise SystemExit(result.returncode)


def find_latest_run():
    """Find the most recent run folder."""
    if not os.path.isdir("runs"):
        return None
    dirs = sorted(os.listdir("runs"), reverse=True)
    for d in dirs:
        run_path = os.path.join("runs", d)
        if os.path.isdir(run_path):
            return run_path
    return None


def main():
    parser = argparse.ArgumentParser(description="Run the full speech-to-sample pipeline")
    parser.add_argument("--from", type=int, default=1, dest="from_step",
                        help="Start from step N (1-6)")
    parser.add_argument("--run", type=str, default=None,
                        help="Use an existing run folder")
    args = parser.parse_args()

    start = args.from_step
    run_dir = args.run

    # Step 1: Get intro suggestions from LLM
    if start <= 1 and not run_dir:
        run_step(1, "Get intro suggestions (LLM)", [PYTHON, "get_intros.py"])

        # Move the generated intros JSON into a new run folder
        import glob
        intros_files = sorted(glob.glob("intros_*.json"), reverse=True)
        intros_files = [f for f in intros_files if os.path.dirname(f) == ""]
        if not intros_files:
            print("No intros_*.json found after step 1.")
            raise SystemExit(1)

        intros_json = intros_files[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join("runs", timestamp)
        os.makedirs(run_dir, exist_ok=True)
        shutil.move(intros_json, os.path.join(run_dir, "intros.json"))
        print(f"\n  Created run: {run_dir}")

    # Find run dir if not set
    if not run_dir:
        run_dir = find_latest_run()
        if not run_dir:
            print("No run folder found. Run from step 1 or specify --run.")
            raise SystemExit(1)

    intros_path = os.path.join(run_dir, "intros.json")
    print(f"\n  Run folder: {run_dir}")

    # Step 2: Download clips from YouTube
    if start <= 2:
        if not os.path.exists(intros_path):
            print(f"  intros.json not found in {run_dir}")
            raise SystemExit(1)
        run_step(2, "Download clips from YouTube", [PYTHON, "download_clips.py", intros_path])

    # Step 3: Strip background music (vocal isolation)
    if start <= 3:
        run_step(3, "Isolate vocals (Mel-Band-Roformer)", [PYTHON, "strip_bgm.py", run_dir])

    # Step 4: Transcribe with Whisper
    if start <= 4:
        run_step(4, "Transcribe audio (Whisper)", [PYTHON, "transcribe.py", run_dir])

    # Step 5: Pick samples (LLM)
    if start <= 5:
        run_step(5, "Pick samples (LLM)", [PYTHON, "pick_sample.py", run_dir])

    # Step 6: Extract audio clips
    if start <= 6:
        run_step(6, "Extract audio clips", [PYTHON, "extract_audio.py", "--input", run_dir])

    print(f"\n{'=' * 60}")
    print(f"  DONE! Check {run_dir}/ for results")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
