# 🏷️ Industrial Label Quality Inspection (RECO)

An automated computer vision system designed for industrial manufacturing and packaging environments (such as PepsiCo operations) to perform real-time quality control on product labels using custom YOLOv8 instance segmentation and OpenCV.

## 🚀 Overview

Ensuring label integrity on production lines is critical. Defective labels, unreadable barcodes, or misaligned placements can cause supply chain issues and rejection downstream. This system processes high-speed video feeds to monitor and classify labels in real-time, distinguishing between compliant ("good") and non-compliant ("bad") items based on strict visual and positional criteria.

### Classification Logic:
*   🟢 **Good Label (`etiqueta buena`)**: The barcode is fully readable, intact, and correctly positioned inside the target validation zone.
*   🔴 **Bad Label (`etiqueta mala`)**: Triggered when the barcode is illegible, torn, poorly printed, or positioned outside the designated bounding box area.

---

## 🛠️ Tech Stack

*   **Python 3.14**
*   **Ultralytics YOLOv8** (Custom Instance Segmentation)
*   **OpenCV (`cv2`)** (Video processing, masking, and real-time rendering)
*   **Cvzone** (Visual UI overlays and text bounding)
*   **NumPy** (Data and coordinate manipulation)

---

## 📁 Project Structure

```text
etiquetas/
│
├── yolov8-custom-segment-main/
│   ├── main1.py             # Main execution script for video processing
│   ├── yolo_segmentation.py # YOLOv8 custom wrapper class for segmentation masks
│   ├── best.pt              # Trained YOLOv8 custom weights model
│   ├── coco1.txt            # Class labels configuration file
│   └── etiquetas.mp4        # Sample input video feed for testing
│
└── README.md
