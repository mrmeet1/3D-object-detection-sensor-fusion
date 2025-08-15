# LiDAR-Camera Fusion for 3D Object Detection Evaluation

This project implements a LiDAR–camera fusion pipeline for 3D object detection.

It:
- Loads KITTI-format dataset (images, LiDAR point clouds, calibration, and labels)
- Segments objects in images using YOLO
- Projects LiDAR points to image space to associate points with detected objects
- Clusters LiDAR points for each object
- Fits oriented bounding boxes using PCA
- Computes IoU between predicted and ground truth boxes
- Evaluates precision, recall, F1-score, mean distance error
- Generates:
  - Annotated images with predicted and ground truth 3D boxes
  - Distance error plots
---

## Requirements

Install dependencies:
```bash
pip install numpy opencv-python ultralytics open3d scikit-learn matplotlib tqdm
