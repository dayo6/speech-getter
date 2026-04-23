"""
Full pipeline orchestrator:
  1. get_intros.py      — LLM generates search queries    → runs/{timestamp}/intros.json
  2. screen_videos.py   — Fetch captions, LLM screens     → runs/{timestamp}/screened.json
  3. download_clips.py  — Download only screened videos    → runs/{timestamp}/{clip_name}/audio.mp3
  4. transcribe.py      — Whisper transcription            → runs/{timestamp}/{clip_name}/transcript.json
  5. pick_sample.py     — LLM picks best quotes            → runs/{timestamp}/{clip_name}/sample_*.json
  6. extract_audio.py   — Slice audio clips                → runs/{timestamp}/{clip_name}/CLIP_*.mp3
  7. strip_bgm.py       — Vocal isolation on clips         → runs/{timestamp}/{clip_name}/CLIP_*_vocals.wav
  8. fx_chain.py        — Apply FX + normalize             → runs/{timestamp}/{clip_name}/CLIP_*_FX.mp3

Usage:
  python orchestrate.py                              # auto-detect and resume
  python orchestrate.py --run runs/20260408_130716   # resume a specific run
  python orchestrate.py --from 1                     # force start from step 1
"""

import sys
import os
import subprocess
import glob
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


def detect_resume_step(run_dir):
    """Scan a run folder and figure out which step to resume from.
    Returns the earliest step that still has incomplete work."""

    clip_dirs = [
        d for d in os.listdir(run_dir)
        if os.path.isdir(os.path.join(run_dir, d))
    ]

    if not clip_dirs:
        # No clip folders yet
        if os.path.exists(os.path.join(run_dir, "screened.json")):
            return 3  # have screened, need downloads
        if os.path.exists(os.path.join(run_dir, "intros.json")):
            return 2  # have intros, need screening
        return 1  # need everything

    # Check each clip's progress, find the earliest incomplete step
    # Order: 4=transcribe, 5=pick, 6=extract, 7=vocals, 8=fx
    needs = {4: 0, 5: 0, 6: 0, 7: 0, 8: 0}

    for d in clip_dirs:
        p = os.path.join(run_dir, d)
        files = os.listdir(p)
        has_audio = os.path.exists(os.path.join(p, "audio.mp3"))
        has_transcript = os.path.exists(os.path.join(p, "transcript.json"))
        has_samples = any(f.startswith("sample_") and f.endswith(".json") for f in files)
        has_clips = any(f.startswith("CLIP_") and f.endswith(".mp3") and "_FX" not in f for f in files)
        has_vocals = any(f.endswith("_vocals.wav") for f in files)
        has_fx = any(f.endswith("_FX.mp3") for f in files)

        if has_audio and not has_transcript:
            needs[4] += 1
        if has_transcript and not has_samples:
            needs[5] += 1
        if has_samples and not has_clips:
            needs[6] += 1
        if has_clips and not has_vocals:
            needs[7] += 1
        if has_clips and not has_fx:
            needs[8] += 1

    # Return earliest step that has work to do
    for step in sorted(needs.keys()):
        if needs[step] > 0:
            return step

    return None  # everything done


def main():
    parser = argparse.ArgumentParser(description="Run the full speech-to-sample pipeline")
    parser.add_argument("--from", type=int, default=None, dest="from_step",
                        help="Force start from step N (1-7)")
    parser.add_argument("--run", type=str, default=None,
                        help="Use an existing run folder")
    parser.add_argument("-n", "--num-queries", type=int, default=10,
                        help="Number of search queries (passed to get_intros and screen_videos)")
    args = parser.parse_args()

    from_step = args.from_step
    run_dir = args.run

    # Step 1: Get intro suggestions from LLM
    # Start fresh if no runs exist, or latest run is already complete
    if not run_dir:
        latest = find_latest_run()
        if not latest:
            from_step = 1
        elif from_step is None and detect_resume_step(latest) is None:
            print(f"  Latest run ({latest}) is complete. Starting fresh.")
            from_step = 1

    if from_step is not None and from_step <= 1 and not run_dir:
        run_step(1, "Get intro suggestions (LLM)", [PYTHON, "get_intros.py", "-n", str(args.num_queries)])

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
            print("No run folder found. Use --from 1 to start fresh.")
            raise SystemExit(1)

    intros_path = os.path.join(run_dir, "intros.json")
    print(f"\n  Run folder: {run_dir}")

    # Auto-detect resume point if --from not specified
    if from_step is None:
        from_step = detect_resume_step(run_dir)
        if from_step is None:
            print("\n  All steps complete! Nothing to do.")
            return
        print(f"  Auto-resuming from step {from_step}")

    screened_path = os.path.join(run_dir, "screened.json")

    # Step 2: Screen videos (fetch captions + LLM filter)
    if from_step <= 2:
        if not os.path.exists(intros_path):
            print(f"  intros.json not found in {run_dir}")
            raise SystemExit(1)
        run_step(2, "Screen videos (captions + LLM)", [PYTHON, "screen_videos.py", intros_path, "-o", screened_path, "--max-per-query", str(args.num_queries)])

    # Step 3: Download only screened clips
    if from_step <= 3:
        # Use screened.json if it exists, otherwise fall back to intros.json
        dl_input = screened_path if os.path.exists(screened_path) else intros_path
        run_step(3, "Download screened clips", [PYTHON, "download_clips.py", dl_input])

    # Step 4: Transcribe with Whisper
    if from_step <= 4:
        run_step(4, "Transcribe audio (Whisper)", [PYTHON, "transcribe.py", run_dir])

    # Step 4b: Emotion scoring
    if from_step <= 5:
        run_step("4b", "Emotion scoring (wav2vec2)", [PYTHON, "emotion_score.py", run_dir])

    # Step 5: Pick samples (LLM)
    if from_step <= 5:
        run_step(5, "Pick samples (LLM)", [PYTHON, "pick_sample.py", run_dir])

    # Step 6: Extract audio clips
    if from_step <= 6:
        run_step(6, "Extract audio clips", [PYTHON, "extract_audio.py", "--input", run_dir])

    # Step 7: Isolate vocals on extracted clips
    if from_step <= 7:
        run_step(7, "Isolate vocals (Mel-Band-Roformer)", [PYTHON, "strip_bgm.py", run_dir])

    # Step 8: Apply FX chain
    if from_step <= 8:
        run_step(8, "Apply FX chain", [PYTHON, "fx_chain.py", run_dir])

    print(f"\n{'=' * 60}")
    print(f"  DONE! Check {run_dir}/ for *_FX.mp3 files")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
