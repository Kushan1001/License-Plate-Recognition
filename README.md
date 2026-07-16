# License Plate Detection & Anonymization

A YOLOv8-based pipeline that detects license plates in images and blurs them out, with the blur quality checked using OCR instead of just eyeballing it. Also handles dataset cleanup, model export to ONNX, and flagging low-confidence detections for review

## What it does

- Audits the dataset before training — checks that every image has a matching label file and removes anything that doesn't
- Trains a YOLOv8n model to detect license plates
- Evaluates on a held-out test set (mAP, precision, recall)
- Runs inference and blurs any detected plates with Gaussian blur
- Verifies the blur actually worked by running OCR on the plate region before and after — if Tesseract can still read text after blurring, that's a failure
- Exports the trained model to ONNX for deployment outside of a Python/PyTorch environment
- Flags detections below a confidence threshold so they can be reviewed and added back into training later
- Benchmarks inference latency and throughput

## Dataset

| Split | Images | Labels |
|---|---|---|
| Train | 25,470 | 25,470 |
| Val | 1,073 | 1,073 |
| Test | 386 | 386 |

No mismatched image/label pairs after cleaning.

## Results

Trained YOLOv8n for 5 epochs on a Tesla T4, evaluated on the 386-image test set.

Detection (test set):
- mAP@50: 0.877
- mAP@50-95: 0.619
- Precision: 0.919
- Recall: 0.832

Inference speed (640x640, T4, averaged over 100 runs):
- Mean latency: 8.05 ms
- Median: 7.69 ms
- Throughput: ~124 FPS (~447k images/hour)

Anonymization:
- 402 plates detected across 386 test images, 0 failed
- 100% of blurred plates came back unreadable when checked with OCR

Exported model: 5.9 MB (.pt) / 11.7 MB (.onnx). Note: the ONNX export is produced but not separately benchmarked in this notebook — no inference was run against it to compare speed or accuracy vs. the PyTorch model.

Note: 5 epochs is low — this was run as a proof of concept for the full pipeline rather than a push for best possible accuracy. With more epochs and a larger backbone (e.g. yolov8s/m) these numbers would likely improve.

## Pipeline

```
raw dataset
  -> audit & remove mismatched files
  -> train YOLOv8n (SGD, cosine LR, AMP)
  -> evaluate on test set
  -> inference: detect -> blur -> verify with OCR
  -> export to ONNX / GCS batch function (untested) / flag low-confidence detections
  -> benchmark latency & throughput
```

## Stack

Python, Ultralytics YOLOv8, OpenCV, Tesseract (pytesseract), ONNX, google-cloud-storage

## Usage

Install:
```bash
pip install ultralytics pytesseract opencv-python google-cloud-storage
```

Train:
```python
from ultralytics import YOLO

model = YOLO('yolov8n.pt')
model.train(data='dataset.yaml', epochs=5, imgsz=640, device=0)
```

Blur plates in one image:
```python
anonymized_img, detections = detect_and_blur_license_plates(
    image_path="input.jpg",
    model=model,
    confidence_threshold=0.5
)
```

Batch process a folder:
```python
stats = batch_anonymize_images(
    input_dir="images/raw",
    output_dir="images/anonymized",
    model=model
)
```

## Files

- `License_Plate_Detection.ipynb` — the full pipeline, from dataset audit through benchmarking
- `dataset.yaml` — YOLO dataset config
