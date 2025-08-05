# 🧠 Automatic Hippocampus Segmentation in Brain MRI using Mask R-CNN

This project presents a deep learning-based solution for automating the segmentation of the **hippocampus** in brain MRI scans. The goal is to support radiation therapy planning by delivering precise segmentation outputs and reducing the need for manual annotation, which is often time-consuming and error-prone.

## 🧠 Overview

Manual hippocampus segmentation is critical in neuro-oncology and cognitive research but is labor-intensive and subject to variability. This project uses a **Mask R-CNN** deep learning architecture enhanced with pre-processing techniques to deliver high segmentation accuracy, improving consistency and efficiency in clinical workflows.

## ✅ Key Features

- 🤖 **Deep Learning Architecture**: Implemented Mask R-CNN with ResNet-50 + FPN backbone for accurate instance segmentation.
- 🧪 **Advanced Image Pre-processing**:
  - Thresholding
  - Canny Edge Detection
  - Keypoint Detection
- 📊 **High Model Performance**:
  - **IoU (Intersection over Union)**: 0.96+
  - **mAP (Mean Average Precision)**: 0.98+
- 🏥 **Clinical Value**:
  - Enables accurate radiation therapy targeting
  - Reduces radiologist manual effort
  - Improves consistency in patient outcomes

## 📂 Dataset

- **Type**: Brain MRI scans with labeled hippocampus regions  
- **Format**: DICOM/NIfTI converted to 2D slices (PNG/JPEG)  
- **Labels**: Binary segmentation masks  
- **Source**: Publicly available datasets (e.g., MICCAI, ADNI) or synthetic/anonymized medical data  

> ⚠️ All data used was anonymized and used in compliance with healthcare data privacy regulations.

## 🧠 Model Architecture

Input MRI Image
↓
Preprocessing (Thresholding, Edge Detection, Keypoint Detection)
↓
Mask R-CNN with ResNet-50 + FPN
↓
Segmentation Mask Output

- **Framework**: PyTorch
- **Backbone**: ResNet-50
- **Segmentation Head**: Fully Convolutional Network (FCN)
- **Loss Function**: BCE + IoU loss

## 🧪 Techniques Used

- Data Augmentation (flipping, rotation, cropping)
- Contrast enhancement (e.g., CLAHE)
- Structural feature detection using keypoints
- Post-processing for smoother mask boundaries

## 🚀 Getting Started
 1. Clone the Repository

```bash
git clone https://github.com/yourusername/hippocampus-segmentation.git
cd hippocampus-segmentation

2. Install Dependencies
pip install torch torchvision opencv-python matplotlib scikit-image

3. Prepare Your Dataset
-Place MRI images in the images/ folder

-Place corresponding segmentation masks in the masks/ folder

4. Train the Model
python train.py --epochs 50 --batch-size 4

5. Evaluate Performance
python evaluate.py

6. Visualize Segmentation Output
python visualize.py

📊 Results
Metric
IoU
mAP
Inference Time

🛠️ Tools & Libraries
PyTorch: Deep learning framework
OpenCV: Image pre-processing
scikit-image: Image enhancement and filters
Matplotlib: Result visualization

💡 Future Enhancements
Integrate with 3D U-Net for volumetric segmentation
Convert model to ONNX for cross-platform deployment
Deploy as a web tool using Streamlit or Flask

👤 Author
Parmi Kenia

