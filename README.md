# License-Plate-Recognition
An end-to-end computer vision system that detects vehicle license plates with YOLOv8 and automatically anonymizes them for privacy-compliant image/video processing — built with production deployment in mind, not just a training notebook.

## 📌 Overview

This project implements a full computer vision pipeline that goes beyond "train a model and stop." It covers **data validation, training, evaluation, privacy-preserving inference, quality verification, edge deployment, cloud integration, and continuous learning** — the components that separate a research notebook from a system that could actually run in production.

**Core use case:** detect license plates in images and blur them automatically, enabling organizations to process and share visual data (dashcam footage, street imagery, parking systems, etc.) while staying privacy-compliant.

## ✨ Key Features

| Stage | What it does |
|---|---|
| 🔍 **Dataset Auditing** | Validates image/label counts per split, detects and removes orphaned files (images without labels or vice versa) |
| 🎯 **Object Detection** | Fine-tunes a YOLOv8 model on a custom license plate dataset |
| 📊 **Model Evaluation** | Computes mAP@50, mAP@50-95, precision, and recall on a held-out test set |
| 🔒 **Privacy Anonymization** | Detects and Gaussian-blurs license plates in single images or batches, with confidence-threshold control |
| ✅ **Blur Verification** | Uses OCR (Tesseract) to programmatically confirm plates are unreadable post-blur — a quantifiable privacy guarantee, not just a visual check |
| 🖼️ **Before/After Visualization** | Side-by-side comparison plots of original vs. anonymized output |
| 📦 **Edge Deployment** | Exports the trained model to **ONNX** for lightweight, cross-platform inference |
| ☁️ **Cloud Integration** | Batch-processes images directly from a **Google Cloud Storage** bucket to another, end-to-end |
| 🔁 **Continuous Learning Loop** | Flags low-confidence detections for human review, enabling active-learning-style dataset improvement |
| ⚡ **Performance Benchmarking** | Measures inference latency, FPS, and throughput for capacity planning |

## 🏗️ Pipeline Architecture

```
Raw Dataset
    │
    ▼
Data Audit & Cleaning  →  removes mismatched image/label pairs
    │
    ▼
YOLOv8 Training  →  custom-configured (SGD, cosine LR, AMP, early stopping)
    │
    ▼
Evaluation  →  mAP@50 / mAP@50-95 / Precision / Recall
    │
    ▼
Inference  →  detect plates → Gaussian blur → verify with OCR
    │
    ├──▶ ONNX Export (edge deployment)
    ├──▶ GCS Batch Processing (cloud pipeline)
    └──▶ Low-Confidence Flagging (continuous learning)
    │
    ▼
Latency & Throughput Benchmarking
```

## 🛠️ Tech Stack

- **Detection:** YOLOv8 (Ultralytics)
- **Image Processing:** OpenCV, PIL
- **OCR Verification:** Tesseract (pytesseract)
- **Deployment:** ONNX export for edge/cross-platform inference
- **Cloud:** Google Cloud Storage (`google-cloud-storage`)
- **Tooling:** NumPy, Matplotlib, tqdm, YAML

## 📈 Results

Model: **YOLOv8n**, trained for 5 epochs on the custom dataset below, evaluated on a held-out 386-image test set (Tesla T4 GPU).

### Dataset

| Split | Images | Labels |
|---|---|---|
| Train | 25,470 | 25,470 |
| Val | 1,073 | 1,073 |
| Test | 386 | 386 |

*(0 mismatched image/label pairs found after the audit step.)*

### Detection Performance (test set)

| Metric | Value |
|---|---|
| mAP@50 | **0.877** |
| mAP@50-95 | **0.619** |
| Precision | **0.919** |
| Recall | **0.832** |

### Inference Speed (Tesla T4, 640×640, benchmarked over 100 runs)

| Metric | Value |
|---|---|
| Mean latency | **8.05 ms** |
| Median latency | 7.69 ms |
| Min / Max latency | 7.30 ms / 16.87 ms |
| Throughput | **124.3 FPS** |
| Scalability | **~447,000 images/hour** |

### Anonymization Quality

| Metric | Value |
|---|---|
| Test images processed | 386 |
| License plates detected | 402 |
| Failed detections | 0 |
| OCR-verified blur success rate | **100%** (plates unreadable post-blur) |

### Model Export

| Format | Size |
|---|---|
| PyTorch (best.pt) | 5.9 MB |
| ONNX (opset 20, simplified) | 11.7 MB |

## 🚀 Getting Started

### Installation
```bash
pip install ultralytics pytesseract opencv-python google-cloud-storage
```

### Train
```python
from ultralytics import YOLO

model = YOLO('yolov8n.pt')
model.train(data='dataset.yaml', epochs=5, imgsz=640, device=0)  # bump epochs for production training
```

### Run Anonymization
```python
from anonymize import detect_and_blur_license_plates

anonymized_img, detections = detect_and_blur_license_plates(
    image_path="input.jpg",
    model=model,
    confidence_threshold=0.5
)
```

### Batch Process a Directory
```python
from anonymize import batch_anonymize_images

stats = batch_anonymize_images(
    input_dir="images/raw",
    output_dir="images/anonymized",
    model=model
)
```

## 📁 Project Structure
```
├── License_Plate_Detection.ipynb   # Full pipeline: audit → train → evaluate → deploy
├── dataset.yaml                    # YOLO dataset config
└── README.md
```

## 🎯 Why This Project Matters

Most license plate detection projects stop at "the model works on a test image." This one treats detection as one component of a **privacy-preserving data pipeline**: it validates its own inputs, quantifies whether anonymization actually succeeded (via OCR, not eyeballing), exports for real-world deployment, integrates with cloud storage, and builds in a feedback loop for improving the model over time.

## 📬 Contact

Feel free to reach out if you'd like to discuss this project, computer vision, or opportunities!

---
*Built with YOLOv8 and a focus on production-readiness.*
