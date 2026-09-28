# 🚦 Traffic Monitoring System (YOLO11n + Streamlit)

A web app that analyses road video, detects and tracks vehicles with **YOLO11n**, classifies them
(car, bus, truck, motorcycle, bicycle), counts them crossing a virtual line, and shows the
annotated output video, charts and tables **in the browser**.

Pipeline: Video -> YOLO11n detection -> ByteTrack tracking -> line-crossing counter -> annotated
H.264 video + charts + CSV/JSON.

## 1. Run on localhost
```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```
Open **http://localhost:8501**, upload a traffic video (or put one in `videos/`),
tune the settings in the sidebar and press **Run analysis**.
YOLO11n weights (`yolo11n.pt`) download automatically on first run.

## 2. Deploy on Streamlit Community Cloud (free)
1. Create a GitHub repo and push this folder (root must contain `app.py` and `requirements.txt`).
   ```bash
   git init && git add . && git commit -m "Traffic monitoring app"
   git branch -M main
   git remote add origin https://github.com/<you>/<repo>.git && git push -u origin main
   ```
2. Go to https://share.streamlit.io -> **Create app** -> select the repo, branch `main`,
   main file path `app.py`.
3. Under **Advanced settings** choose **Python 3.11** (recommended), then **Deploy**.
4. First build takes a few minutes (installing PyTorch/Ultralytics).

Cloud tips: it runs on CPU, so use short clips, inference size 320-480, or set
**Max frames** (e.g. 300) in the sidebar. Upload limit is 200 MB (`.streamlit/config.toml`).
Other hosts (Hugging Face Spaces with the Streamlit SDK, Render, Docker) work too.

## Outputs
Shown in the page and downloadable: annotated video, `events.csv`, `summary.json`.

## Command-line modes (optional)
```bash
python detect_videos.py                          # detect in every video in videos/
python main.py --source videos/traffic.mp4       # full counting, saves to outputs/
```

## Custom training
See `train.py` and `configs/vehicles.yaml` to fine-tune YOLO11n on your own labeled vehicles
(e.g. auto-rickshaws). Put the resulting `best.pt` in the project and set `model:` in
`configs/config.yaml`.

## Structure
```
traffic_monitor/
├── app.py                 # Streamlit web app
├── main.py, detect_videos.py, train.py
├── requirements.txt
├── .streamlit/config.toml
├── configs/               # config.yaml, vehicles.yaml
├── src/                   # pipeline, counter, report, video_utils
├── videos/  outputs/  datasets/
```

## Troubleshooting
- **Video doesn't play in browser:** ffmpeg re-encoding failed; make sure `imageio-ffmpeg`
  installed, or install system ffmpeg. The video can still be downloaded.
- **No vehicles counted:** move the counting line to where traffic crosses it, lower confidence.
- **Slow:** lower inference size, set Max frames, or use a GPU locally.
