"""Streamlit web app - Traffic Monitoring System (YOLO11n).

Run locally:   streamlit run app.py     ->  http://localhost:8501
"""
import copy
import json
import tempfile
from pathlib import Path

import cv2
import pandas as pd
import streamlit as st
import yaml

st.set_page_config(page_title="Traffic Monitoring System", page_icon="🚦", layout="wide")

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".webm"}
BASE_CFG = yaml.safe_load(Path("configs/config.yaml").read_text())


@st.cache_resource(show_spinner="Loading YOLO11n model...")
def load_model(weights):
    from ultralytics import YOLO
    return YOLO(weights)


def first_frame(path):
    cap = cv2.VideoCapture(str(path))
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


def draw_line_preview(frame, line):
    h, w = frame.shape[:2]
    img = frame.copy()
    cv2.line(img, (int(line[0] * w), int(line[1] * h)),
             (int(line[2] * w), int(line[3] * h)), (0, 255, 255), 3)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# ------------------------------------------------------------------ header
st.title("🚦 Traffic Monitoring System")
st.caption("Detect, track and count vehicles by category in road videos using **YOLO11n**.")

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    conf = st.slider("Confidence threshold", 0.05, 0.95, float(BASE_CFG["conf"]), 0.05)
    imgsz = st.select_slider("Inference size (smaller = faster)", [320, 480, 640, 960], value=640)
    tracker = st.selectbox("Tracker", ["bytetrack.yaml", "botsort.yaml"])
    all_classes = {v: int(k) for k, v in BASE_CFG["classes"].items()}
    chosen = st.multiselect("Vehicle categories", list(all_classes), default=list(all_classes))
    st.subheader("Counting line")
    orient = st.radio("Orientation", ["Horizontal", "Vertical"], horizontal=True)
    pos = st.slider("Position", 0.05, 0.95, 0.60, 0.05)
    max_frames = st.number_input("Max frames to process (0 = whole video)", 0, 100000, 0, 50,
                                 help="Limit frames for quicker results on slow / cloud CPUs.")
    live = st.checkbox("Show live preview while processing", value=True)

line = [0.0, pos, 1.0, pos] if orient == "Horizontal" else [pos, 0.0, pos, 1.0]

# ------------------------------------------------------------------ input
tab_up, tab_local = st.tabs(["📤 Upload video", "📁 Use video from videos/ folder"])
video_path = None
with tab_up:
    up = st.file_uploader("Upload a road/traffic video", type=[e.strip(".") for e in VIDEO_EXTS])
    if up is not None:
        tmp = Path(tempfile.mkdtemp()) / up.name
        tmp.write_bytes(up.getbuffer())
        video_path = tmp
with tab_local:
    local = sorted(p for p in Path("videos").glob("*") if p.suffix.lower() in VIDEO_EXTS)
    if local:
        pick = st.selectbox("Choose a video", ["-"] + [p.name for p in local])
        if pick != "-" and video_path is None:
            video_path = Path("videos") / pick
    else:
        st.info("No videos in the `videos/` folder. Upload one in the other tab.")

# ------------------------------------------------------------------ preview + run
if video_path is not None:
    frame = first_frame(video_path)
    if frame is None:
        st.error("Could not read this video. Try another file.")
    else:
        c1, c2 = st.columns([2, 1])
        with c1:
            st.image(draw_line_preview(frame, line), caption="Preview: yellow line = counting line",
                     use_container_width=True)
        with c2:
            st.markdown("**Ready to analyse**")
            st.write(f"File: `{video_path.name}`")
            st.write(f"Categories: {', '.join(chosen) or '—'}")
            run_btn = st.button("▶️ Run analysis", type="primary", use_container_width=True,
                                disabled=not chosen)

        if run_btn:
            from src.pipeline import run
            cfg = copy.deepcopy(BASE_CFG)
            cfg.update(conf=conf, imgsz=imgsz, tracker=tracker)
            cfg["classes"] = {all_classes[n]: n for n in chosen}

            model = load_model(cfg["model"])
            out_dir = tempfile.mkdtemp(prefix="traffic_")
            bar = st.progress(0.0, text="Starting...")
            live_box = st.empty()

            def on_progress(i, total, annotated):
                if total > 0:
                    bar.progress(min(i / total, 1.0), text=f"Processing frame {i}/{total}")
                else:
                    bar.progress(0.0, text=f"Processing frame {i}")
                if live and i % 5 == 0:
                    live_box.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                                   caption="Live preview", use_container_width=True)

            try:
                summary = run(str(video_path), cfg, out_dir, False, line, None, True,
                              on_progress, int(max_frames) or None, model)
            except Exception as e:
                st.error(f"Processing failed: {e}")
                st.stop()
            bar.progress(1.0, text="Done ✅")
            live_box.empty()

            out = Path(out_dir)
            st.session_state["result"] = {
                "summary": {k: v for k, v in summary.items() if k not in ("events", "video_path")},
                "events": pd.DataFrame(summary["events"]),
                "video": (out / "annotated.mp4").read_bytes(),
            }

# ------------------------------------------------------------------ results
res = st.session_state.get("result")
if res:
    st.divider()
    st.header("📊 Results")
    s = res["summary"]
    per_class = s["per_class"]

    cols = st.columns(len(per_class) + 1 if per_class else 1)
    cols[0].metric("Total vehicles", s["grand_total"])
    for col, (name, v) in zip(cols[1:], per_class.items()):
        col.metric(name.capitalize(), v["total"], help=f"in: {v['in']} | out: {v['out']}")

    left, right = st.columns([3, 2])
    with left:
        st.subheader("🎬 Annotated output video")
        st.video(res["video"], format="video/mp4")
        st.download_button("⬇️ Download video", res["video"], "annotated.mp4", "video/mp4")
    with right:
        st.subheader("Vehicles by category")
        if per_class:
            chart_df = pd.DataFrame(per_class).T[["in", "out"]]
            st.bar_chart(chart_df)
        else:
            st.info("No vehicles crossed the line. Try moving the line or lowering confidence.")
        st.caption(f"Frames processed: {s['frames_processed']} · "
                   f"Time: {s['processing_time_s']}s")

    if not res["events"].empty:
        st.subheader("Crossing events")
        st.dataframe(res["events"], use_container_width=True, height=250)
        d1, d2 = st.columns(2)
        d1.download_button("⬇️ events.csv", res["events"].to_csv(index=False),
                           "events.csv", "text/csv")
        d2.download_button("⬇️ summary.json", json.dumps(s, indent=2),
                           "summary.json", "application/json")
else:
    st.info("Upload a video (or pick one from `videos/`), adjust the settings, and press **Run analysis**.")
