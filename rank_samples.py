"""
GUI for ranking beat intro samples.
Plays each _FX.mp3 clip, lets you rate 1-10 stars, saves to CSV.

Usage:
  python rank_samples.py <run_folder>
  python rank_samples.py runs/20260410_233714
"""

import sys
import os
import csv
import json
import tkinter as tk
from tkinter import ttk
import subprocess
import threading

# Use pygame for audio if available, otherwise fall back to system player
try:
    import pygame
    pygame.mixer.init()
    HAS_PYGAME = True
except ImportError:
    HAS_PYGAME = False


class SampleRanker:
    def __init__(self, root, run_dir):
        self.root = root
        self.run_dir = run_dir
        self.all_samples = self._find_samples()  # never modified
        self.samples = list(self.all_samples)     # current view (may be filtered)
        self.current_idx = 0
        self.ratings = {}
        self.csv_path = os.path.join(run_dir, "rankings.csv")

        # Load existing ratings and resume from first unrated
        self._load_existing()
        self._resume_position()

        self.root.title(f"Sample Ranker — {os.path.basename(run_dir)}")
        self.root.configure(bg="#1a1a1a")
        self.root.geometry("700x500")
        self.root.resizable(False, False)

        self._build_ui()
        self._show_current()
        self._play()

    def _find_samples(self):
        """Find all _FX.mp3 files across clip folders."""
        samples = []
        for d in sorted(os.listdir(self.run_dir)):
            clip_dir = os.path.join(self.run_dir, d)
            if not os.path.isdir(clip_dir):
                continue
            for f in sorted(os.listdir(clip_dir)):
                if f.endswith("_FX.mp3"):
                    sample_json = f.replace("_FX.mp3", ".json").replace("CLIP_", "")
                    json_path = os.path.join(clip_dir, sample_json)
                    text = ""
                    if os.path.exists(json_path):
                        with open(json_path, "r", encoding="utf-8") as fh:
                            data = json.load(fh)
                            text = data.get("text", "")
                    samples.append({
                        "path": os.path.join(clip_dir, f),
                        "clip": d,
                        "file": f,
                        "text": text,
                    })
        return samples

    def _key(self, sample):
        """Unique key for a sample: clip/file."""
        return f"{sample['clip']}/{sample['file']}"

    def _load_existing(self):
        """Load existing ratings from CSV."""
        if os.path.exists(self.csv_path):
            with open(self.csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    rating = int(row.get("rating", 0))
                    if rating == 0:
                        continue  # skip unrated entries
                    key = f"{row['clip']}/{row['file']}"
                    self.ratings[key] = rating

    def _resume_position(self):
        """Jump to the first unrated sample."""
        if not self.ratings or not self.samples:
            return
        for i, s in enumerate(self.samples):
            if self._key(s) not in self.ratings:
                self.current_idx = i
                return
        # All rated — go to the last one
        self.current_idx = len(self.samples) - 1

    def _save_csv(self):
        """Save all ratings to CSV."""
        with open(self.csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["clip", "file", "rating", "text", "path"])
            writer.writeheader()
            for s in self.all_samples:
                rating = self.ratings.get(self._key(s), 0)
                writer.writerow({
                    "clip": s["clip"],
                    "file": s["file"],
                    "rating": rating,
                    "text": s["text"],
                    "path": s["path"],
                })
        self.status_var.set(f"Saved {len(self.ratings)} ratings to rankings.csv")

    def _build_ui(self):
        style = ttk.Style()
        style.theme_use("clam")

        # Header
        header = tk.Frame(self.root, bg="#1a1a1a")
        header.pack(fill="x", padx=20, pady=(15, 5))

        self.counter_var = tk.StringVar()
        tk.Label(header, textvariable=self.counter_var, font=("Consolas", 11),
                 fg="#888", bg="#1a1a1a").pack(side="left")

        self.progress_var = tk.StringVar()
        tk.Label(header, textvariable=self.progress_var, font=("Consolas", 11),
                 fg="#888", bg="#1a1a1a").pack(side="right")

        # Clip name
        self.clip_var = tk.StringVar()
        tk.Label(self.root, textvariable=self.clip_var, font=("Consolas", 13, "bold"),
                 fg="#fff", bg="#1a1a1a", wraplength=660, anchor="w").pack(
                     fill="x", padx=20, pady=(10, 2))

        # File name
        self.file_var = tk.StringVar()
        tk.Label(self.root, textvariable=self.file_var, font=("Consolas", 10),
                 fg="#666", bg="#1a1a1a", anchor="w").pack(fill="x", padx=20, pady=(0, 5))

        # Text preview
        self.text_frame = tk.Frame(self.root, bg="#252525", highlightbackground="#333",
                                    highlightthickness=1)
        self.text_frame.pack(fill="both", expand=True, padx=20, pady=5)

        self.text_label = tk.Label(self.text_frame, text="", font=("Consolas", 10),
                                    fg="#ccc", bg="#252525", wraplength=640,
                                    justify="left", anchor="nw")
        self.text_label.pack(fill="both", expand=True, padx=10, pady=10)

        # Play button
        btn_frame = tk.Frame(self.root, bg="#1a1a1a")
        btn_frame.pack(fill="x", padx=20, pady=10)

        self.play_btn = tk.Button(btn_frame, text="PLAY", font=("Consolas", 12, "bold"),
                                   bg="#333", fg="#fff", activebackground="#555",
                                   activeforeground="#fff", relief="flat", padx=20, pady=5,
                                   command=self._play)
        self.play_btn.pack(side="left")

        self.stop_btn = tk.Button(btn_frame, text="STOP", font=("Consolas", 12, "bold"),
                                   bg="#333", fg="#fff", activebackground="#555",
                                   activeforeground="#fff", relief="flat", padx=20, pady=5,
                                   command=self._stop)
        self.stop_btn.pack(side="left", padx=(10, 0))

        # Stars
        stars_frame = tk.Frame(self.root, bg="#1a1a1a")
        stars_frame.pack(fill="x", padx=20, pady=5)

        tk.Label(stars_frame, text="Rating:", font=("Consolas", 11),
                 fg="#888", bg="#1a1a1a").pack(side="left")

        self.star_buttons = []
        for i in range(1, 11):
            btn = tk.Button(stars_frame, text=str(i), font=("Consolas", 11, "bold"),
                           width=3, bg="#333", fg="#888", activebackground="#555",
                           relief="flat", command=lambda x=i: self._rate(x))
            btn.pack(side="left", padx=2)
            self.star_buttons.append(btn)

        # Nav buttons
        nav_frame = tk.Frame(self.root, bg="#1a1a1a")
        nav_frame.pack(fill="x", padx=20, pady=(10, 5))

        tk.Button(nav_frame, text="< PREV", font=("Consolas", 11),
                  bg="#333", fg="#fff", activebackground="#555",
                  relief="flat", padx=15, pady=3,
                  command=self._prev).pack(side="left")

        tk.Button(nav_frame, text="SAVE", font=("Consolas", 11, "bold"),
                  bg="#2a5a2a", fg="#fff", activebackground="#3a7a3a",
                  relief="flat", padx=15, pady=3,
                  command=self._save_csv).pack(side="left", padx=10)

        # Best-of filter
        tk.Label(nav_frame, text="  Best:", font=("Consolas", 11),
                 fg="#888", bg="#1a1a1a").pack(side="left", padx=(20, 0))
        self.threshold_var = tk.StringVar(value="7")
        threshold_menu = tk.OptionMenu(nav_frame, self.threshold_var,
                                        "5", "6", "7", "8", "9", "10")
        threshold_menu.config(font=("Consolas", 10), bg="#333", fg="#fff",
                              activebackground="#555", highlightthickness=0)
        threshold_menu.pack(side="left", padx=2)
        tk.Button(nav_frame, text="PLAY", font=("Consolas", 11, "bold"),
                  bg="#2a2a5a", fg="#fff", activebackground="#3a3a7a",
                  relief="flat", padx=8, pady=3,
                  command=self._filter_best).pack(side="left", padx=2)
        tk.Button(nav_frame, text="ALL", font=("Consolas", 10),
                  bg="#333", fg="#888", activebackground="#555",
                  relief="flat", padx=8, pady=3,
                  command=self._show_all).pack(side="left", padx=2)

        tk.Button(nav_frame, text="NEXT >", font=("Consolas", 11),
                  bg="#333", fg="#fff", activebackground="#555",
                  relief="flat", padx=15, pady=3,
                  command=self._next).pack(side="left")

        # Status bar
        self.status_var = tk.StringVar()
        tk.Label(self.root, textvariable=self.status_var, font=("Consolas", 9),
                 fg="#555", bg="#1a1a1a", anchor="w").pack(fill="x", padx=20, pady=(0, 10))

        # Keyboard bindings
        self.root.bind("<Left>", lambda e: self._prev())
        self.root.bind("<Right>", lambda e: self._next())
        self.root.bind("<space>", lambda e: self._play())
        self.root.bind("<Escape>", lambda e: self._stop())
        for i in range(1, 10):
            self.root.bind(str(i), lambda e, x=i: self._rate(x))
        self.root.bind("0", lambda e: self._rate(10))

    def _show_current(self):
        if not self.samples:
            self.clip_var.set("No samples found")
            return

        s = self.samples[self.current_idx]
        self.counter_var.set(f"{self.current_idx + 1} / {len(self.samples)}")
        self.clip_var.set(s["clip"])
        self.file_var.set(s["file"])
        self.text_label.config(text=s["text"] if s["text"] else "(no text)")

        rated = sum(1 for sample in self.samples if self._key(sample) in self.ratings)
        self.progress_var.set(f"{rated}/{len(self.samples)} rated")

        # Highlight current rating
        current_rating = self.ratings.get(self._key(s), 0)
        for i, btn in enumerate(self.star_buttons):
            if i + 1 <= current_rating:
                btn.config(bg="#d4a017", fg="#000")
            else:
                btn.config(bg="#333", fg="#888")

    def _play(self):
        if not self.samples:
            return
        path = self.samples[self.current_idx]["path"]
        if HAS_PYGAME:
            pygame.mixer.music.load(path)
            pygame.mixer.music.play()
        else:
            # Fallback: system player
            if sys.platform == "win32":
                os.startfile(path)

    def _stop(self):
        self._autoplay = False
        if hasattr(self, '_pending_advance') and self._pending_advance:
            self.root.after_cancel(self._pending_advance)
            self._pending_advance = None
        if HAS_PYGAME:
            pygame.mixer.music.stop()

    def _rate(self, rating):
        if not self.samples:
            return
        # Cancel any pending auto-advance
        if hasattr(self, '_pending_advance') and self._pending_advance:
            self.root.after_cancel(self._pending_advance)
            self._pending_advance = None

        s = self.samples[self.current_idx]
        self.ratings[self._key(s)] = rating
        self._save_csv()
        self.status_var.set(f"Rated {s['file']}: {rating}/10 — saved")
        self._show_current()

        # Check if ALL samples (not just filtered) are rated
        rated = sum(1 for sample in self.all_samples if self._key(sample) in self.ratings)
        if rated == len(self.all_samples):
            self._stop()
            from tkinter import messagebox
            messagebox.showinfo("Done!", f"All {len(self.samples)} samples rated!\nRankings saved to rankings.csv")
            return

        # Auto-advance after delay
        self._pending_advance = self.root.after(300, self._next)

    def _filter_best(self):
        """Filter to only show samples at or above the threshold, auto-play through all."""
        self._stop()
        threshold = int(self.threshold_var.get())
        filtered = [s for s in self.all_samples if self.ratings.get(self._key(s), 0) >= threshold]
        if not filtered:
            self.status_var.set(f"No samples rated {threshold}+ found")
            return
        self.samples = filtered
        self.current_idx = 0
        self._autoplay = True
        self._show_current()
        self._play_and_advance()
        self.status_var.set(f"Playing {len(filtered)} samples rated {threshold}+  (press ALL to reset)")

    def _play_and_advance(self):
        """Play current sample, then advance to next when it finishes."""
        if not self._autoplay or not self.samples:
            return
        self._play()
        # Check clip duration via pygame, fallback to 20 seconds
        try:
            if HAS_PYGAME:
                sound = pygame.mixer.Sound(self.samples[self.current_idx]["path"])
                duration_ms = int(sound.get_length() * 1000) + 500  # add 500ms gap
                sound = None  # free it, we play via music not sound
            else:
                duration_ms = 20000
        except Exception:
            duration_ms = 20000
        # Schedule next after this clip finishes
        self._pending_advance = self.root.after(duration_ms, self._autoplay_next)

    def _autoplay_next(self):
        """Advance to next sample in autoplay mode."""
        if not self._autoplay:
            return
        if self.current_idx < len(self.samples) - 1:
            self.current_idx += 1
            self._show_current()
            self._play_and_advance()
        else:
            self._autoplay = False
            self.status_var.set(f"Done previewing {len(self.samples)} best samples — press ALL to go back")

    def _show_all(self):
        """Reset filter and show all samples."""
        self._stop()
        self.samples = list(self.all_samples)
        self.current_idx = 0
        self._show_current()
        self.status_var.set(f"Showing all {len(self.samples)} samples")

    def _prev(self):
        if self.current_idx > 0:
            self._stop()
            self.current_idx -= 1
            self._show_current()
            self._play()

    def _next(self):
        if self.current_idx < len(self.samples) - 1:
            self._stop()
            self.current_idx += 1
            self._show_current()
            self._play()
        else:
            self.status_var.set("Last sample in list. Press PREV to go back.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python rank_samples.py <run_folder>")
        raise SystemExit(1)

    run_dir = sys.argv[1]
    if not os.path.isdir(run_dir):
        print(f"Not a directory: {run_dir}")
        raise SystemExit(1)

    root = tk.Tk()
    app = SampleRanker(root, run_dir)
    root.mainloop()
