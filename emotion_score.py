"""
Score transcript segments for emotional intensity using audeering wav2vec2.
Outputs emotions.json with per-segment arousal/dominance/valence scores
and pre-computed best windows for sample picking.

Usage:
  python emotion_score.py <run_folder>
  python emotion_score.py runs/20260418_071401
"""

import sys
import os
import json
import numpy as np
import torch
import torch.nn as nn
import torchaudio
from transformers import Wav2Vec2Processor
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2Model,
    Wav2Vec2PreTrainedModel,
)
from dotenv import load_dotenv

load_dotenv()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
HF_TOKEN = os.environ.get("HF_TOKEN")
MODEL_NAME = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"
TARGET_SR = 16000
MIN_SEGMENT_SECS = 0.5  # skip segments shorter than this


# -- Custom model classes (from audeering model card) --

class RegressionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = self.dropout(features)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        return self.out_proj(x)


class EmotionModel(Wav2Vec2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.wav2vec2 = Wav2Vec2Model(config)
        self.classifier = RegressionHead(config)
        self.init_weights()

    def forward(self, input_values):
        outputs = self.wav2vec2(input_values)
        hidden_states = torch.mean(outputs[0], dim=1)
        logits = self.classifier(hidden_states)
        return hidden_states, logits


# -- Load model once --

print(f"Loading emotion model ({MODEL_NAME}) on {DEVICE}...")
processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME, token=HF_TOKEN)
model = EmotionModel.from_pretrained(MODEL_NAME, token=HF_TOKEN).to(DEVICE)
model.eval()
print("Emotion model loaded.\n")


def score_segment(audio_tensor, sr):
    """Score a single audio segment. Returns (arousal, dominance, valence)."""
    # Resample to 16kHz if needed
    if sr != TARGET_SR:
        audio_tensor = torchaudio.functional.resample(audio_tensor, sr, TARGET_SR)

    signal = audio_tensor.numpy()
    if signal.ndim == 2:
        signal = signal.mean(axis=0)  # mono

    y = processor(signal, sampling_rate=TARGET_SR)
    y = torch.from_numpy(
        np.array(y["input_values"][0]).reshape(1, -1)
    ).to(DEVICE)

    with torch.no_grad():
        _, logits = model(y)

    vals = logits.detach().cpu().numpy()[0]
    return float(vals[0]), float(vals[1]), float(vals[2])


def compute_intensity(arousal, dominance, valence):
    """High arousal + high dominance + low valence = aggressive/dark."""
    return arousal * 0.45 + dominance * 0.30 + (1.0 - valence) * 0.25


def find_best_windows(segments, min_dur=14, max_dur=23, top_n=10):
    """Find consecutive segment windows in the 14-23s range, ranked by intensity."""
    windows = []
    n = len(segments)

    for i in range(n):
        for j in range(i + 1, n + 1):
            window_segs = segments[i:j]
            dur = window_segs[-1]["end"] - window_segs[0]["start"]
            if dur < min_dur:
                continue
            if dur > max_dur:
                break

            avg_intensity = sum(s["intensity"] for s in window_segs) / len(window_segs)
            avg_arousal = sum(s["arousal"] for s in window_segs) / len(window_segs)
            avg_dominance = sum(s["dominance"] for s in window_segs) / len(window_segs)
            avg_valence = sum(s["valence"] for s in window_segs) / len(window_segs)

            windows.append({
                "start": round(window_segs[0]["start"], 3),
                "end": round(window_segs[-1]["end"], 3),
                "duration": round(dur, 3),
                "avg_intensity": round(avg_intensity, 4),
                "avg_arousal": round(avg_arousal, 4),
                "avg_dominance": round(avg_dominance, 4),
                "avg_valence": round(avg_valence, 4),
                "segment_indices": list(range(i, j)),
            })

    windows.sort(key=lambda w: w["avg_intensity"], reverse=True)
    return windows[:top_n]


def score_clip(clip_dir):
    """Score all transcript segments in a clip directory."""
    transcript_path = os.path.join(clip_dir, "transcript.json")
    audio_path = os.path.join(clip_dir, "audio.mp3")
    emotions_path = os.path.join(clip_dir, "emotions.json")

    if os.path.exists(emotions_path):
        return  # already done

    if not os.path.exists(transcript_path) or not os.path.exists(audio_path):
        return

    with open(transcript_path, "r", encoding="utf-8") as f:
        transcript = json.load(f)

    segments = transcript.get("segments", [])
    if not segments:
        return

    # Load full audio once
    audio, sr = torchaudio.load(audio_path)

    scored_segments = []
    for seg in segments:
        start = seg.get("start", 0)
        end = seg.get("end", 0)
        duration = end - start

        if duration < MIN_SEGMENT_SECS:
            continue

        # Extract segment audio
        start_sample = int(start * sr)
        end_sample = int(end * sr)
        seg_audio = audio[:, start_sample:end_sample]

        if seg_audio.shape[1] < int(TARGET_SR * MIN_SEGMENT_SECS):
            continue

        arousal, dominance, valence = score_segment(seg_audio, sr)
        intensity = compute_intensity(arousal, dominance, valence)

        scored_segments.append({
            "start": round(start, 3),
            "end": round(end, 3),
            "text": seg.get("text", ""),
            "arousal": round(arousal, 4),
            "dominance": round(dominance, 4),
            "valence": round(valence, 4),
            "intensity": round(intensity, 4),
        })

    # Find best windows
    top_windows = find_best_windows(scored_segments)

    # Print summary
    if scored_segments:
        best = max(scored_segments, key=lambda s: s["intensity"])
        avg = sum(s["intensity"] for s in scored_segments) / len(scored_segments)
        print(f"    {len(scored_segments)} segments scored — "
              f"avg intensity: {avg:.3f}, peak: {best['intensity']:.3f}")
        if top_windows:
            w = top_windows[0]
            print(f"    Best window: [{w['start']:.1f}s - {w['end']:.1f}s] "
                  f"intensity={w['avg_intensity']:.3f}")

    output = {
        "model": MODEL_NAME,
        "segments": scored_segments,
        "top_windows": top_windows,
    }

    with open(emotions_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python emotion_score.py <run_folder>")
        raise SystemExit(1)

    target = sys.argv[1]

    if os.path.isdir(target):
        clip_dirs = sorted(
            d for d in os.listdir(target)
            if os.path.isdir(os.path.join(target, d))
        )

        # Filter to dirs that need scoring
        to_process = []
        skipped = 0
        for d in clip_dirs:
            clip_path = os.path.join(target, d)
            if os.path.exists(os.path.join(clip_path, "emotions.json")):
                skipped += 1
            elif (os.path.exists(os.path.join(clip_path, "transcript.json"))
                  and os.path.exists(os.path.join(clip_path, "audio.mp3"))):
                to_process.append((d, clip_path))

        if skipped:
            print(f"Skipping {skipped} already scored clips.")
        if not to_process:
            print("Nothing to score.")
            raise SystemExit(0)

        print(f"Scoring {len(to_process)} clips for emotional intensity\n")
        for i, (name, path) in enumerate(to_process, 1):
            print(f"[{i}/{len(to_process)}] {name[:70]}")
            score_clip(path)

        print(f"\nDone! Scored {len(to_process)} clips.")
    else:
        score_clip(target)
