# Methodology & Reproducibility

Full experimental setup behind the YOLO26s-OBB × DOTA v2.0 project. Every number in the README is reproducible from the CSVs in [`../results/raw/`](../results/raw/).

## 1. Dataset

[DOTA v2.0](https://captain-whu.github.io/DOTA/dataset.html): 11,268 source images, 1,793,658 instances, 18 classes. Source images range from ~800×800 to ~20,000×20,000 px.

Following the standard DOTA workflow, the large images are **split into 1024×1024 tiles for training** (Ultralytics provides `split_dota.py` for this; the project used a DOTA v2.0 already prepared in this tiled, YOLO-OBB format).

| Split | Tiles | Tile size |
|-------|------:|-----------|
| Training | 17,521 | 1024×1024 |
| Validation | 5,305 | 1024×1024 |

Labels are in YOLO-OBB 4-point format: `class x1 y1 x2 y2 x3 y3 x4 y4` (normalised).

**At inference time, full-size images are used directly** (no tiling), so test images span a wide range of resolutions.

## 2. Training

Exact configuration (from [`../results/raw/train_args.yaml`](../results/raw/train_args.yaml)):

| Parameter | Value |
|-----------|-------|
| Model | YOLO26s-OBB (9.8M params, 55.1 GFLOPs) |
| Task | Oriented bounding box (OBB) |
| Input size | 1024 × 1024 |
| Batch | 32 |
| Epochs | 150 target (patience 50); ran to epoch 134 |
| **Best checkpoint** | **epoch 43** |
| Optimizer | `auto` (Ultralytics-selected) |
| LR schedule | cosine, lr0 0.01, 3 warmup epochs |
| Precision | AMP (mixed precision) |
| Augmentation | mosaic 1.0 (closed last 10 ep), mixup 0.15, flips (0.5/0.5), rotation ±10° |
| Seed | 42 |
| Hardware | NVIDIA A100 (RunPod) |

## 3. Evaluation

Validation metrics at the best epoch (from [`../results/raw/training_log.csv`](../results/raw/training_log.csv)):

| Metric | Value |
|--------|-------|
| mAP@50 | 0.666 |
| mAP@50-95 | 0.503 |
| Precision | 0.776 |
| Recall | 0.604 |
| Peak mean F1 | 0.65 @ conf 0.458 |

Per-class AP@50 ranges from tennis-court (0.941) down to helipad (0.035), tracking each class's training-set frequency.

## 4. Inference

Full-size DOTA v1.5 test images (469) were run through the trained model with `inference_gpu.py`, producing 31,990 detections logged per-image and per-detection to CSV. Mean detection confidence: 0.708.

## 5. Edge-Deployment Study (proxy)

**Important:** these are proxy measurements on a **desktop (AMD Ryzen 9 5900HX / RTX 3060)**, not on physical edge hardware. The model was exported to ONNX and benchmarked over the 469 test images under three runtimes:

| Runtime | Provider | Stands in for | FPS | Avg latency | Peak RAM |
|---------|----------|---------------|----:|------------:|---------:|
| ONNX GPU | CUDAExecutionProvider | Jetson-class accelerator | 30.7 | 32.5 ms | 1,000 MB |
| PyTorch GPU | CUDA (FP16) | desktop baseline | 7.9 | 126.5 ms | — |
| ONNX CPU | CPUExecutionProvider (4 threads) | Pi-class device | 3.5 | 283.6 ms | 722 MB |

Source: [`../results/raw/deployment_summary.csv`](../results/raw/deployment_summary.csv).

**Interpretation caveats:**
- An RTX 3060 is more powerful than a Jetson Orin Nano's GPU, so the 30.7 FPS figure is an **optimistic upper bound**, not on-device performance.
- The reliable, hardware-independent finding is that the **ONNX-GPU path is ~4× faster than PyTorch** on the same hardware, and that the model is **memory-light** (≤1 GB).
- On-device validation is future work.

## 6. Software Stack

| Component | Version |
|-----------|---------|
| OS | Windows (inference) / Linux (training, RunPod) |
| Python | 3.11 |
| Framework | Ultralytics 8.4.x |
| Runtime | PyTorch (CUDA) · ONNX Runtime |

## 7. Reproducibility

- `results/raw/training_log.csv` — per-epoch metrics for all 134 epochs
- `results/raw/train_args.yaml` — the exact training arguments
- `results/raw/deployment_summary.csv` — the edge-study summary
- `results/raw/inference_summary.csv` — per-image inference results

The chart scripts in `src/` regenerate every figure from these files.
