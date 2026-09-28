"""Core pipeline: read video -> YOLO11 detect + track -> count -> annotate -> save."""
import time
from collections import defaultdict, deque
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from .counter import LineCounter
from .report import save_reports
from .video_utils import make_browser_playable

PALETTE = {
    "car": (255, 128, 0),
    "bus": (0, 200, 255),
    "truck": (0, 0, 255),
    "motorcycle": (0, 220, 0),
    "bicycle": (255, 0, 200),
}


def color_for(name):
    return PALETTE.get(name, (200, 200, 200))


def draw_panel(frame, counter):
    totals = counter.totals()
    lines = ["VEHICLE COUNT (in/out)"] + [
        f"{c}: {v['in']}/{v['out']}" for c, v in totals.items()
    ]
    lines.append(f"TOTAL: {sum(v['total'] for v in totals.values())}")
    h = 24 * len(lines) + 10
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (270, 10 + h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    for i, t in enumerate(lines):
        cv2.putText(frame, t, (18, 32 + 24 * i), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 1, cv2.LINE_AA)


def run(source, cfg, output_dir="outputs", show=False, line=None, weights=None,
        save_video=True, progress_cb=None, max_frames=None, model=None):
    """Process a video.

    progress_cb(frame_idx, total_frames, annotated_frame_bgr) is called every frame
    (used by the Streamlit app). `model` lets callers pass a pre-loaded YOLO model.
    """
    if model is None:
        model = YOLO(weights or cfg["model"])
    else:
        model.predictor = None  # reset tracker state between videos

    class_map = {int(k): v for k, v in cfg["classes"].items()}
    class_ids = list(class_map)

    cap = cv2.VideoCapture(source if isinstance(source, int) else str(source))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video source: {source}")
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if max_frames and total_frames > 0:
        total_frames = min(total_frames, int(max_frames))

    rel = line or cfg["line"]
    p1, p2 = (int(rel[0] * W), int(rel[1] * H)), (int(rel[2] * W), int(rel[3] * H))
    counter = LineCounter(p1, p2)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / "annotated_raw.mp4"
    writer = None
    if save_video:
        writer = cv2.VideoWriter(str(raw_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

    trails = defaultdict(lambda: deque(maxlen=cfg.get("trail_length", 30)))
    frame_idx, t0 = 0, time.time()

    while True:
        if max_frames and frame_idx >= max_frames:
            break
        ok, frame = cap.read()
        if not ok:
            break
        res = model.track(frame, persist=True, conf=cfg["conf"], iou=cfg["iou"],
                          imgsz=cfg["imgsz"], classes=class_ids,
                          tracker=cfg["tracker"], device=cfg.get("device"),
                          verbose=False)[0]

        cv2.line(frame, p1, p2, (0, 255, 255), 2)

        if res.boxes is not None and res.boxes.id is not None:
            boxes = res.boxes.xyxy.cpu().numpy().astype(int)
            ids = res.boxes.id.cpu().numpy().astype(int)
            clss = res.boxes.cls.cpu().numpy().astype(int)
            for (x1, y1, x2, y2), tid, c in zip(boxes, ids, clss):
                name = class_map.get(int(c), str(c))
                col = color_for(name)
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                trails[tid].append((cx, cy))
                direction = counter.update(int(tid), name, (cx, cy), frame_idx, fps)

                cv2.rectangle(frame, (x1, y1), (x2, y2), col, 4 if direction else 2)
                label = f"{name} #{tid}"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), col, -1)
                cv2.putText(frame, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.circle(frame, (cx, cy), 3, col, -1)
                if cfg.get("show_trails", True) and len(trails[tid]) > 1:
                    cv2.polylines(frame, [np.array(trails[tid], np.int32)], False, col, 2)

        draw_panel(frame, counter)
        cur_fps = (frame_idx + 1) / max(time.time() - t0, 1e-6)
        cv2.putText(frame, f"FPS: {cur_fps:.1f}", (W - 130, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

        if writer:
            writer.write(frame)
        if show:
            cv2.imshow("Traffic Monitoring (q to quit)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        frame_idx += 1
        if progress_cb:
            progress_cb(frame_idx, total_frames, frame)
        elif frame_idx % 50 == 0:
            pct = f" ({100 * frame_idx / total_frames:.0f}%)" if total_frames > 0 else ""
            print(f"Processed {frame_idx} frames{pct} @ {cur_fps:.1f} FPS")

    cap.release()
    if writer:
        writer.release()
    try:
        cv2.destroyAllWindows()
    except cv2.error:
        pass  # headless OpenCV build

    video_path = None
    if save_video:
        video_path = out_dir / "annotated.mp4"
        make_browser_playable(raw_path, video_path)

    meta = {"source": str(source), "model": str(weights or cfg["model"]),
            "frames_processed": frame_idx, "video_fps": fps,
            "processing_time_s": round(time.time() - t0, 1)}
    summary = save_reports(counter, out_dir, meta)
    summary["events"] = counter.events
    summary["video_path"] = str(video_path) if video_path else None

    if not progress_cb:
        print("\n=== Summary ===")
        for c, v in summary["per_class"].items():
            print(f"{c:12s} in={v['in']:3d} out={v['out']:3d} total={v['total']:3d}")
        print(f"Grand total: {summary['grand_total']}")
        print(f"Results saved to: {out_dir.resolve()}")
    return summary
