# Triton Vision Infer

**Triton Vision Infer** is a lightweight Python library for running high-performance computer vision inference on NVIDIA Triton Inference Server.  
It provides simple utilities to send images, handle model outputs, and post-process results — with built-in support for YOLO object detection.

---

## Features
- Easy integration with NVIDIA Triton Inference Server
- Pre-processing and post-processing utilities for vision models
- YOLO output parsing with or without TensorRT NMS plugin
- Supports batch and single-image inference
- Extendable for classification, segmentation, and other vision tasks

---

## Requirements
- Python 3.8+
- NVIDIA Triton Inference Server
- Basic Python environment with `tritonclient` and `numpy`

---

## Use Cases
- Real-time object detection
- Video analytics pipelines
- Edge and cloud-based inference
- Integration of AI vision models into production systems

---

## Contributing
Contributions are welcome!  
Please open an issue or submit a pull request to discuss changes.

---

## License
MIT License — see [LICENSE](LICENSE) for details.

---

## Architecture Overview

```
 ┌───────────────────┐
 │   Application     │
 │ (Your AI Service) │
 └───────▲───────────┘
         │ Calls Library API
         │
 ┌───────┴────────────────────────┐
 │  Triton Vision Infer Library   │
 │ ┌────────────────────────────┐ │
 │ │  Preprocessing Module      │ │
 │ │  - Image loading           │ │
 │ │  - Resize & normalize      │ │
 │ │  - Batch creation          │ │
 │ └──────────────▲─────────────┘ │
 │                │               │
 │ ┌──────────────┴─────────────┐ │
 │ │  Triton Client Module      │ │
 │ │  - gRPC/HTTP calls         │ │
 │ │  - Input/Output handling   │ │
 │ └──────────────▲─────────────┘ │
 │                │               │
 │ ┌──────────────┴─────────────┐ │
 │ │  Postprocessing Module     │ │
 │ │  - YOLO NMS parsing        │ │
 │ │  - Bounding box conversion │ │
 │ │  - Confidence filtering    │ │
 │ └────────────────────────────┘ │
 └────────────────────────────────┘
         │
         │ gRPC/HTTP
         ▼
 ┌────────────────────────────────┐
 │ NVIDIA Triton Inference Server │
 │ - Hosts TensorRT / ONNX models │
 │ - Runs inference               │
 └────────────────────────────────┘
```
