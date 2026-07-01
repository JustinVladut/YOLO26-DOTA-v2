"""
GPU Inference Script — YOLO26s-OBB on aerial test images

Saves ALL annotated images to one flat folder.
Logs full per-image and per-detection metrics to CSV.

Edit the CONFIG paths below, then run:
  python inference_gpu.py
"""

import time
import csv
import cv2
from pathlib import Path
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

MODEL_PATH  = "best.pt"                    # trained weights
TEST_IMAGES = "test_images"                # folder of images to run on
OUTPUT_DIR  = "results/gpu"                # CSVs + annotated images

IMGSZ  = 1024
CONF   = 0.25
IOU    = 0.45
DEVICE = 0

# ---------------------------------------------------------------------------
# SETUP
# ---------------------------------------------------------------------------

output_path   = Path(OUTPUT_DIR)
annotated_dir = output_path / "annotated"
annotated_dir.mkdir(parents=True, exist_ok=True)

img_files = sorted(
    list(Path(TEST_IMAGES).glob("*.png")) +
    list(Path(TEST_IMAGES).glob("*.jpg")) +
    list(Path(TEST_IMAGES).glob("*.jpeg")) +
    list(Path(TEST_IMAGES).glob("*.tif"))
)

print(f"Model:       {MODEL_PATH}")
print(f"Test images: {len(img_files)} files")
print(f"Output:      {annotated_dir}")
print(f"Device:      GPU  |  imgsz: {IMGSZ}px  |  conf: {CONF}  |  iou: {IOU}")
print("=" * 60)

if not img_files:
    print("ERROR: No images found. Check TEST_IMAGES path.")
    exit(1)

# ---------------------------------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------------------------------

model = YOLO(MODEL_PATH)
print(f"\nModel loaded: {len(model.names)} classes")
print(f"Classes: {list(model.names.values())}\n")

# Warmup
print("Warming up GPU...")
model.predict(source=str(img_files[0]), imgsz=IMGSZ, conf=CONF,
              device=DEVICE, verbose=False)
print("Warmup done.\n")

# ---------------------------------------------------------------------------
# INFERENCE LOOP
# ---------------------------------------------------------------------------

summary_rows   = []   # one row per image
detection_rows = []   # one row per detection

for i, img_path in enumerate(img_files, 1):
    print(f"[{i}/{len(img_files)}] {img_path.name}")

    # Get image dimensions
    img_orig = cv2.imread(str(img_path))
    img_h, img_w = img_orig.shape[:2] if img_orig is not None else (0, 0)

    start = time.perf_counter()
    results = model.predict(
        source=str(img_path),
        imgsz=IMGSZ,
        conf=CONF,
        iou=IOU,
        device=DEVICE,
        verbose=False,
        save=False,
    )
    latency_ms = (time.perf_counter() - start) * 1000

    result = results[0]

    # Save annotated image flat
    annotated_img = result.plot()
    cv2.imwrite(str(annotated_dir / img_path.name), annotated_img)

    # Parse detections
    num_det = 0
    class_counts = {}

    if result.obb is not None and len(result.obb.cls) > 0:
        cls_array  = result.obb.cls.cpu().numpy()
        conf_array = result.obb.conf.cpu().numpy()
        xyxyxyxy   = result.obb.xyxyxyxy.cpu().numpy()  # [N, 4, 2] polygon points

        num_det = len(cls_array)

        for j in range(num_det):
            cls_idx   = int(cls_array[j])
            cls_name  = model.names[cls_idx]
            conf_val  = float(conf_array[j])

            # Compute approximate box area from polygon
            pts = xyxyxyxy[j]  # [4, 2]
            xs, ys = pts[:, 0], pts[:, 1]
            # Shoelace formula for polygon area
            n = len(xs)
            area = abs(sum(xs[k]*ys[(k+1)%n] - xs[(k+1)%n]*ys[k] for k in range(n))) / 2

            class_counts[cls_name] = class_counts.get(cls_name, 0) + 1

            detection_rows.append({
                "image":      img_path.name,
                "class_idx":  cls_idx,
                "class_name": cls_name,
                "confidence": round(conf_val, 4),
                "box_area_px": round(float(area), 1),
            })

    print(f"  Latency: {latency_ms:.1f}ms  |  Detections: {num_det}")
    if class_counts:
        for cls, count in sorted(class_counts.items(), key=lambda x: -x[1]):
            print(f"    {cls}: {count}")

    summary_rows.append({
        "image":         img_path.name,
        "width_px":      img_w,
        "height_px":     img_h,
        "latency_ms":    round(latency_ms, 2),
        "detections":    num_det,
        "classes_found": str(class_counts),
    })

# ---------------------------------------------------------------------------
# SAVE CSVs
# ---------------------------------------------------------------------------

# Summary CSV — one row per image
summary_csv = output_path / "inference_summary.csv"
with open(summary_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["image","width_px","height_px","latency_ms","detections","classes_found"])
    writer.writeheader()
    writer.writerows(summary_rows)

# Detections CSV — one row per detection
detections_csv = output_path / "inference_detections.csv"
with open(detections_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["image","class_idx","class_name","confidence","box_area_px"])
    writer.writeheader()
    writer.writerows(detection_rows)

# ---------------------------------------------------------------------------
# PRINT SUMMARY
# ---------------------------------------------------------------------------

latencies = [r["latency_ms"] for r in summary_rows]
avg_lat   = sum(latencies) / len(latencies)

print("\n" + "=" * 60)
print("INFERENCE COMPLETE")
print("=" * 60)
print(f"Images processed : {len(summary_rows)}")
print(f"Total detections : {sum(r['detections'] for r in summary_rows)}")
print(f"Avg latency      : {avg_lat:.1f} ms/image")
print(f"Min latency      : {min(latencies):.1f} ms")
print(f"Max latency      : {max(latencies):.1f} ms")
print(f"Throughput       : {1000/avg_lat:.2f} images/sec")
print(f"\nAnnotated images : {annotated_dir}")
print(f"Summary CSV      : {summary_csv}")
print(f"Detections CSV   : {detections_csv}")