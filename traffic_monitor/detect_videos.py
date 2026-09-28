"""Simplest mode: run YOLO11n on EVERY video in the videos/ folder and save
annotated results + a per-video vehicle summary.

Usage:
    1. Copy your video(s) into the videos/ folder
    2. python detect_videos.py
"""
from collections import Counter
from pathlib import Path

from ultralytics import YOLO

VEHICLES = {1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
EXTS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".m4v"}

videos = sorted(p for p in Path("videos").iterdir() if p.suffix.lower() in EXTS)
if not videos:
    raise SystemExit("No videos found. Copy your video file(s) into the 'videos' folder first.")

model = YOLO("yolo11n.pt")

for vid in videos:
    print(f"\n>>> Processing {vid.name}")
    seen = {}  # track_id -> class name (unique vehicles)
    for r in model.track(source=str(vid), stream=True, persist=True, conf=0.35,
                         classes=list(VEHICLES), tracker="bytetrack.yaml",
                         save=True, project="outputs", name=vid.stem, exist_ok=True):
        if r.boxes.id is not None:
            for tid, c in zip(r.boxes.id.int().tolist(), r.boxes.cls.int().tolist()):
                seen[tid] = VEHICLES[c]
    counts = Counter(seen.values())
    print(f"Unique vehicles in {vid.name}: {sum(counts.values())}")
    for name, n in counts.most_common():
        print(f"  {name:12s} {n}")
    print(f"Annotated video saved in outputs/{vid.stem}/")
