"""
Resume YOLO26s-obb training on RunPod A100.

ONLY works if previous training was stopped via Ctrl+C (not auto-completed).
last.pt must contain optimizer state for true resume.

Usage:
  cd /workspace/YOLO26s-DOTAv2.0/src
  python resume_runpod.py
"""

from ultralytics import YOLO

RUN_PATH = "/workspace/YOLO26s-DOTAv2.0/runs/yolo26s-obb-dotav2-a100"

model = YOLO(f"{RUN_PATH}/weights/last.pt")
results = model.train(resume=True)
