"""Fine-tune YOLO11n on a custom vehicle dataset (YOLO format).

Usage:
    python train.py --data configs/vehicles.yaml --epochs 50
"""
import argparse
from ultralytics import YOLO

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="configs/vehicles.yaml")
ap.add_argument("--epochs", type=int, default=50)
ap.add_argument("--imgsz", type=int, default=640)
ap.add_argument("--batch", type=int, default=16)
ap.add_argument("--device", default=None)
a = ap.parse_args()

model = YOLO("yolo11n.pt")
model.train(data=a.data, epochs=a.epochs, imgsz=a.imgsz, batch=a.batch, device=a.device)
metrics = model.val()
print("mAP50-95:", metrics.box.map)
print("Best weights: runs/detect/train/weights/best.pt")
