"""Traffic Monitoring System - YOLO11n vehicle detection, tracking & counting.

Examples:
    python main.py --source videos/traffic.mp4
    python main.py --source videos/traffic.mp4 --show --line 0.1,0.7,0.9,0.7
    python main.py --source 0                       # webcam
    python main.py --source videos/traffic.mp4 --weights runs/detect/train/weights/best.pt
"""
import argparse
import yaml

from src.pipeline import run


def parse_args():
    ap = argparse.ArgumentParser(description="YOLO11n Traffic Monitoring System")
    ap.add_argument("--source", required=True, help="video path, RTSP URL, or 0 for webcam")
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--output", default="outputs")
    ap.add_argument("--weights", default=None, help="custom .pt weights (overrides config)")
    ap.add_argument("--line", default=None, help="counting line as x1,y1,x2,y2 (relative 0-1)")
    ap.add_argument("--conf", type=float, default=None)
    ap.add_argument("--show", action="store_true", help="display live window")
    ap.add_argument("--no-save-video", action="store_true")
    return ap.parse_args()


def main():
    a = parse_args()
    with open(a.config) as f:
        cfg = yaml.safe_load(f)
    if a.conf is not None:
        cfg["conf"] = a.conf
    line = [float(v) for v in a.line.split(",")] if a.line else None
    source = int(a.source) if a.source.isdigit() else a.source
    run(source, cfg, a.output, a.show, line, a.weights, not a.no_save_video)


if __name__ == "__main__":
    main()
