<div align="center">

# MSE-Net
### A Lightweight Multi-scale Detection Network for Corn Leaf Disease Detection in Complex Field Environments

[![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4.1-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Ultralytics](https://img.shields.io/badge/Ultralytics-8.4.143-111F68)](https://github.com/ultralytics/ultralytics)
[![Base Model](https://img.shields.io/badge/Base-YOLOv11n-orange)](https://github.com/ultralytics/ultralytics)
[![Parameters](https://img.shields.io/badge/Parameters-2.25M-brightgreen)](#model-complexity)
[![Weights](https://img.shields.io/badge/Weights-4.8%20MB-blueviolet)](#model-complexity)
[![GFLOPs](https://img.shields.io/badge/GFLOPs-5.9-red)](#model-complexity)
<img width="692" height="436" alt="image" src="https://github.com/user-attachments/assets/e4021580-d84a-4c1a-9e1d-44fef4b49f99" />

**MSE-Net = YOLOv11n + MAFPN + SimAM + ECA**

A lightweight object detector designed to localize and recognize nine categories of corn diseases, pest damage, and healthy leaves in complex field images.

</div>

---

## 📋 Table of Contents

- [Overview](#overview)
- [Core Challenges Addressed](#core-challenges-addressed)
- [What's New](#whats-new)
- [Architecture](#architecture)
  - [Backbone Optimization](#1-backbone-optimization)
  - [MAFPN](#2-mafpn)
  - [SimAM](#3-simam)
  - [ECA](#4-eca)
- [Dataset](#dataset)
- [Experimental Setup](#experimental-setup)
- [Main Results](#main-results)
  - [Comparison with Detection Models](#1-comparison-with-detection-models)
  - [Ablation Study](#2-ablation-study)
  - [Model Complexity](#3-model-complexity)
  - [Deployment Evaluation](#4-deployment-evaluation)
- [Quick Start](#quick-start)
- [Training](#training)
- [Inference and Validation](#inference-and-validation)
- [Visualization](#visualization)
- [Project Structure](#project-structure)
- [Citation](#citation)
- [Acknowledgements](#acknowledgements)
- [License](#license)
- [Contact](#contact)

---

## 📌Overview

Accurate detection of corn leaf diseases in complex field environments is important for precision crop protection and intelligent agricultural management. However, early disease lesions are often small, weakly textured, and poorly bounded. Similar visual symptoms, leaf occlusion, illumination variation, and complex backgrounds further increase the difficulty of reliable detection.

To address these challenges while considering the computational constraints of edge devices, we propose **MSE-Net**, a lightweight multi-scale object detection network based on **YOLOv11n**. The proposed model restructures the backbone and neck and introduces two attention modules before the detection head. Specifically, the shallow backbone channels are compressed, the original feature-fusion pathway is replaced by a multi-scale adaptive feature pyramid network (MAFPN), and SimAM and ECA are sequentially applied to the three detection-scale features.

The resulting model achieves a precision of **88.37 ± 0.86%**, a recall of **79.03 ± 0.40%**, an mAP@0.5 of **84.40 ± 0.26%**, and an mAP@0.5:0.95 of **71.87 ± 0.12%** in the reported multi-seed experiments. It contains approximately **2.25 M parameters**, requires **5.9 GFLOPs**, and has a weight size of approximately **4.8 MB**.

## 🎯Core Challenges Addressed

| Challenge in field images | Corresponding design |
|---|---|
| 🔍Small lesions and weak local texture | Multi-level feature interaction in MAFPN helps preserve fine-grained information. |
| 📐Large variation in lesion scale | AVG branches, cross-level connections, and RepHMS-based feature processing improve multi-scale fusion. |
| 🌿Background texture and leaf interference | SimAM recalibrates feature responses before detection. |
| ⚡Unequal contribution of feature channels | ECA models local cross-channel interaction using a lightweight 1D convolution. |
| 🏗️Limited computational resources on edge platforms | Backbone channel compression and an efficient feature-fusion/attention design reduce model complexity. |

## 🆕What's New

- **✅Backbone channel compression:** The output channels of the second and fourth C3k2 modules are reduced from 256 to 128 and from 512 to 256, respectively. The remaining backbone module types and connection order are retained, including SPPF and C2PSA.
- **✅MAFPN in the neck:** The original upsampling/downsampling and concatenation pathway is reorganized with AVG branches, cross-level connections, and RepHMS blocks to improve interaction among P3, P4, and P5 features.
- **✅Sequential SimAM–ECA attention:** SimAM and ECA are placed sequentially on the P3, P4, and P5 features before they enter the Detect head.
- **✅Lightweight detection:** The integrated architecture reduces the model size and computational cost while improving precision and mAP@0.5:0.95 relative to the YOLOv11n ablation baseline.

---

## Architecture

MSE-Net is developed from YOLOv11n. Its modifications cover the backbone, neck, and detection head.

### Overall Architecture
<img width="692" height="436" alt="image" src="https://github.com/user-attachments/assets/cac92185-b2c5-4243-b9d5-797b6b742c40" />

<!-- Insert the actual overall architecture figure from Figures/.
     Replace the filename below with the exact filename in the repository. -->
<!-- ![Overall architecture of MSE-Net](Figures/REPLACE_WITH_ACTUAL_ARCHITECTURE_FILENAME.png) -->

### Structural modifications

| Network component | Modification | Purpose |
|---|---|---|
| Backbone | Reduce channels in the second and fourth C3k2 modules from 256/512 to 128/256 | Reduce shallow-stage computation and model complexity |
| Neck | Replace the original feature-fusion pathway with MAFPN | Improve multi-scale feature interaction |
| Neck | Introduce AVG branches, cross-level connections, and RepHMS blocks | Fuse features from different levels and refine fused representations |
| Detection head input | Apply SimAM followed by ECA to P3, P4, and P5 | Recalibrate feature responses and channel importance |
| Detection head | Retain the YOLOv11n Detect head after attention processing | Predict disease classes and bounding boxes |

### 1. Backbone Optimization

In the backbone, the output channels of the second and fourth C3k2 modules are reduced from 256 to 128 and from 512 to 256, respectively. The module types and connection order of the other backbone stages remain unchanged, and SPPF and C2PSA are retained. This adjustment reduces the channel scale in the shallow feature-extraction stages to control computational cost.

### 2. MAFPN

The original YOLOv11n neck, which is mainly composed of upsampling, downsampling, concatenation, and C3k2 operations, is reconstructed as MAFPN.

During the construction of P5, P4, and P3, AVG branches are introduced to align and aggregate features from different scales. Cross-level connections allow shallow lesion edges and texture details to participate in the construction of higher-level representations, while deeper semantic information is propagated back to lower-level features. RepHMS blocks replace multiple C3k2 blocks along the fusion pathway and use scale-specific configurations to further process the concatenated features.

The MAFPN design combines AVG, Concat, and RepHMS operations to improve information exchange among P3, P4, and P5 features, particularly for lesions with different sizes and weak textures.

### 3. SimAM

SimAM is inserted into each of the three detection-scale feature paths (P3, P4, and P5), before the Detect head. It estimates the importance of individual feature responses using an energy-based formulation and converts the resulting values into attention weights through a sigmoid function. The weights are applied to the input features element-wise.

SimAM does not introduce additional learnable parameters. In MSE-Net, it is used to enhance informative lesion-related responses and reduce interference from less relevant feature regions.

### 4. ECA

ECA follows SimAM in each detection-scale path. It uses global average pooling and a one-dimensional convolution to model local cross-channel interactions without dimensionality reduction. In this study, the 1D convolution kernel size is **3**.

The generated channel weights recalibrate the feature channels before the features are passed to the Detect head. Thus, SimAM and ECA sequentially refine the detection features from response-level and channel-level perspectives.

<!-- Insert the actual module illustration (MAFPN, SimAM, and ECA) from Figures/.
     Replace the filename with the exact repository filename. -->
<!-- ![Detailed structures of MAFPN, SimAM, and ECA](Figures/REPLACE_WITH_ACTUAL_MODULE_FIGURE_FILENAME.png) -->

---
<img width="685" height="399" alt="image" src="https://github.com/user-attachments/assets/5db55fe9-258d-420b-9b3f-ac84a81bd613" />

## 📊Dataset

### Dataset source and curation

The dataset was collected from **three publicly available corn-disease image subsets on Roboflow**. The source images cover different corn growth stages, including jointing, the large bell stage, and tasseling/silking.

Because the public images do not consistently provide acquisition-device, geographic, date, or disease-stage metadata, the source subsets differ in resolution, imaging devices, backgrounds, symptom stages, geographic origins, and environmental conditions. Images were visually screened. Images with unclear symptoms, low resolution, severe compression, irrelevant backgrounds, duplicate content, or uncertain labels were excluded.

Labels were reviewed according to characteristic disease symptoms, including lesion shape, color, boundary, distribution pattern, and affected plant part. The final dataset contains **3,600 images** across nine categories, with images and annotations stored in YOLO format.
<img width="691" height="306" alt="image" src="https://github.com/user-attachments/assets/9413e612-68f9-4916-a6a4-6aef8bc8a488" />

### Disease categories

| ID | Category |
|---:|---|
| 0 | Common Rust |
| 1 | Gray Leaf Spot |
| 2 | Corn Smut |
| 3 | Southern Rust |
| 4 | Philippine Downy Mildew |
| 5 | Maize Stripe Virus Disease |
| 6 | Corn Borer Damage |
| 7 | Corn Borer Larvae |
| 8 | Healthy |

### Dataset split

The original dataset was divided into three subsets at a ratio of **7:2:1**. To reduce potential information leakage, images from the same web source or with highly similar visual characteristics were not assigned across different subsets. Class and image source were considered during the split.

| Class | Set 1 (70%) | Set 2 (20%) | Set 3 (10%) | Total |
|---|---:|---:|---:|---:|
| Common Rust | 280 | 80 | 40 | 400 |
| Gray Leaf Spot | 280 | 80 | 40 | 400 |
| Corn Smut | 280 | 80 | 40 | 400 |
| Southern Rust | 280 | 80 | 40 | 400 |
| Philippine Downy Mildew | 280 | 80 | 40 | 400 |
| Maize Stripe Virus Disease | 280 | 80 | 40 | 400 |
| Corn Borer Damage | 280 | 80 | 40 | 400 |
| Corn Borer Larvae | 280 | 80 | 40 | 400 |
| Healthy | 280 | 80 | 40 | 400 |
| **Total** | **2,520** | **720** | **360** | **3,600** |

### Data augmentation

Data augmentation was applied **only to Set 1**, after the dataset split. Set 2 and Set 3 were kept unchanged. Spatial transformations included horizontal flipping, translation (within ±10% of image width/height), and rotation (within ±15°). Pixel-level transformations included contrast adjustment (0.7–1.4), brightness shift (−15 to +15 gray levels), Gaussian noise (standard deviation 3–15 gray levels), and motion blur using 3×3 or 5×5 linear kernels.

The augmented Set 1 was doubled in size. The resulting dataset contained **6,120 images in total**: 5,040 training images, 720 validation images, and 360 test images.

| Subset | Images used | Role |
|---|---:|---|
| Augmented Set 1 | 5,040 | Training |
| Set 2 | 720 | Validation |
| Set 3 | 360 | Test |
| **Total** | **6,120** | |

---

## Experimental Setup

The models were trained and evaluated on a Linux server equipped with an **NVIDIA RTX 4090D 24 GB GPU**.

| Item | Setting |
|---|---|
| Python | 3.10 |
| PyTorch | 2.4.1+cu121 |
| Ultralytics | 8.4.143 |
| Input image size | 640 × 640 |
| Epochs | 250 |
| Patience | 60 |
| Batch size | 16 |
| Optimizer | SGD |
| Workers | 0 |
| Random seeds | 0, 42, 123 |
| Cache | RAM |
| AMP | Enabled |

All compared models used the same dataset split and main training settings. Multi-seed results are reported as **mean ± standard deviation**.

---

## Main Results

### 1. Comparison with Detection Models

The proposed MSE-Net was compared with CNN-based YOLO detectors and Transformer-based detectors under the same experimental protocol.

| Model | Precision (%) | Recall (%) | mAP@0.5 (%) | mAP@0.5:0.95 (%) | Weights (MB) | Parameters | GFLOPs |
|---|---:|---:|---:|---:|---:|---:|---:|
| YOLOv5n | 86.10 ± 2.00 | 79.43 ± 0.29 | 83.73 ± 0.48 | 69.90 ± 0.59 | 8.22 | 2,504,699 | 7.1 |
| YOLOv6n | 88.23 ± 0.39 | 78.53 ± 1.39 | 83.80 ± 0.42 | 71.40 ± 0.22 | 8.32 | 4,234,635 | 11.7 |
| YOLOv8n | 87.50 ± 0.94 | 80.46 ± 1.30 | 84.97 ± 0.45 | 71.73 ± 0.52 | 5.99 | 3,307,403 | 8.1 |
| YOLOv9s | 87.83 ± 0.52 | 80.80 ± 0.62 | 85.37 ± 0.68 | 73.90 ± 0.54 | 14.56 | 7,170,571 | 26.7 |
| YOLOv9m | 88.30 ± 1.31 | 82.67 ± 0.63 | 85.37 ± 0.42 | 74.23 ± 0.37 | 38.98 | 20,019,499 | 76.5 |
| YOLOv10n | 86.30 ± 0.99 | 80.87 ± 1.30 | 84.67 ± 0.58 | 71.73 ± 0.45 | 5.52 | 2,266,923 | 6.6 |
| YOLOv10m | 87.13 ± 1.32 | 81.00 ± 1.36 | 85.37 ± 0.48 | 73.23 ± 0.45 | 31.97 | 15,318,379 | 59.0 |
| YOLOv10l | 87.00 ± 1.48 | 80.20 ± 1.22 | 84.60 ± 0.29 | 73.07 ± 0.17 | 49.82 | 24,316,267 | 120.2 |
| YOLOv11n | 86.57 ± 0.09 | 80.40 ± 0.22 | 84.63 ± 0.17 | 71.70 ± 0.22 | 5.25 | 2,583,907 | 6.4 |
| YOLOv12n | 87.93 ± 0.79 | 79.27 ± 0.76 | 84.10 ± 0.41 | 70.67 ± 0.90 | 5.29 | 2,558,483 | 7.3 |
| RT-DETR-L | 88.33 ± 0.83 | 79.80 ± 1.27 | 84.13 ± 0.19 | 73.47 ± 0.26 | 63.2 | 32,824,571 | 109.9 |
| RT-DETR-ResNet50 | 87.50 ± 0.45 | 79.77 ± 0.56 | 82.97 ± 0.26 | 73.33 ± 0.26 | 82.2 | 41,953,179 | 127.6 |
| RT-DETR-ResNet101 | 86.70 ± 1.59 | 79.73 ± 0.80 | 82.77 ± 0.69 | 73.43 ± 0.31 | 118.6 | 60,919,195 | 188.2 |
| RT-DETR-X | 86.47 ± 2.19 | 80.63 ± 0.90 | 83.80 ± 0.59 | 73.53 ± 0.77 | 129.2 | 65,485,931 | 224.7 |
| LW-DETR-Tiny | 84.07 ± 0.39 | 95.30 ± 0.03 | 84.43 ± 0.54 | 74.20 ± 0.57 | 135.1 | 11,760,564 | 11.2 |
| LW-DETR-Small | 82.43 ± 0.63 | 95.83 ± 0.34 | 82.43 ± 0.63 | 73.13 ± 0.62 | 163.8 | 14,265,940 | 16.6 |
| D-FINE-N | 70.73 ± 0.78 | 60.30 ± 1.68 | 67.93 ± 2.78 | 57.97 ± 2.35 | 58.1 | 3,726,779 | 7.1 |
| **MSE-Net** | **88.37 ± 0.86** | **79.03 ± 0.40** | **84.40 ± 0.26** | **71.87 ± 0.12** | **4.8** | **2,246,216** | **5.9** |

MSE-Net achieves the highest precision among the compared models while maintaining a compact architecture. Compared with YOLOv11n, its precision increases by **1.80 percentage points**, while mAP@0.5:0.95 increases by **0.17 percentage points**. Its weight size, parameter count, and computational complexity are reduced by approximately **8.6%**, **13.1%**, and **7.8%**, respectively. Recall and mAP@0.5 are slightly lower than those of the YOLOv11n baseline, reflecting a trade-off between precision, recall, and model compactness.

### 2. Ablation Study

The contributions and interactions of MAFPN, SimAM, and ECA were evaluated using eight configurations and three random seeds.

| No. | MAFPN | SimAM | ECA | Precision (%) | Recall (%) | mAP@0.5 (%) | mAP@0.5:0.95 (%) | Weights (MB) | Parameters | GFLOPs |
|---:|:---:|:---:|:---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | × | × | × | 86.57 ± 0.09 | 80.40 ± 0.22 | 84.63 ± 0.17 | 71.70 ± 0.22 | 5.25 | 2,583,907 | 6.4 |
| 2 | × | ✓ | × | 85.20 ± 2.31 | 79.37 ± 2.01 | 83.47 ± 0.65 | 70.80 ± 0.82 | 5.25 | 2,583,907 | 6.4 |
| 3 | × | × | ✓ | 87.90 ± 2.00 | 79.17 ± 0.51 | 84.57 ± 0.40 | 71.30 ± 0.79 | 5.03 | 2,464,760 | 5.4 |
| 4 | ✓ | × | × | 85.87 ± 2.41 | 79.10 ± 2.29 | 83.77 ± 0.75 | 71.27 ± 1.27 | 4.75 | 2,246,207 | 5.9 |
| 5 | × | ✓ | ✓ | 86.67 ± 2.80 | 78.60 ± 1.47 | 83.77 ± 0.59 | 68.63 ± 0.45 | 5.19 | 2,547,960 | 5.5 |
| 6 | ✓ | ✓ | × | 88.93 ± 2.50 | 79.77 ± 1.42 | 84.47 ± 0.83 | 71.67 ± 0.64 | 4.75 | 2,246,207 | 5.9 |
| 7 | ✓ | × | ✓ | 88.77 ± 0.15 | 79.17 ± 0.55 | 83.83 ± 0.68 | 71.00 ± 0.20 | 4.76 | 2,246,216 | 5.9 |
| **8 (MSE-Net)** | **✓** | **✓** | **✓** | **88.37 ± 0.86** | **79.03 ± 0.40** | **84.40 ± 0.26** | **71.87 ± 0.12** | **4.76** | **2,246,216** | **5.9** |

The complete model obtains the highest mAP@0.5:0.95 among the ablation configurations and exhibits a small standard deviation for this metric. The results also show that the effects of the three modules are not simply additive: the individual and paired configurations have different performance and stability characteristics. The final combination was selected for its overall balance among precision, localization performance, and model complexity.

### 3. Model Complexity

| Metric | MSE-Net |
|---|---:|
| Weight size | Approximately 4.8 MB |
| Parameters | 2,246,216 (approximately 2.25 M) |
| GFLOPs | 5.9 |

### 4. Deployment Evaluation

Deployment was evaluated on a Windows computer equipped with an **AMD Ryzen 7 5800H CPU and Radeon Graphics**, and on an **NVIDIA Jetson Orin Nano Engineering Reference Developer Kit Super**. PyTorch and ONNX were evaluated on both platforms; TensorRT was additionally evaluated on Jetson Orin Nano.

#### Windows CPU platform

| Model | Format | Precision (%) | Recall (%) | mAP@0.5 (%) | mAP@0.5:0.95 (%) | Latency (ms) |
|---|---|---:|---:|---:|---:|---:|
| YOLOv11n | PyTorch | 84.5 | 81.4 | 84.3 | 70.8 | 110.2 |
| YOLOv11n | ONNX | 89.3 | 78.1 | 83.9 | 69.9 | 74.0 |
| MSE-Net | PyTorch | 89.9 | 79.4 | 84.9 | 71.1 | 164.7 |
| MSE-Net | ONNX | 88.9 | 80.8 | 84.5 | 70.5 | 78.0 |

#### NVIDIA Jetson Orin Nano platform

| Model | Format | Precision (%) | Recall (%) | mAP@0.5 (%) | mAP@0.5:0.95 (%) | Latency (ms) |
|---|---|---:|---:|---:|---:|---:|
| YOLOv11n | PyTorch | 84.4 | 81.4 | 84.3 | 70.8 | 29.0 |
| YOLOv11n | ONNX | 89.3 | 78.1 | 83.9 | 69.9 | 25.6 |
| YOLOv11n | TensorRT | 89.3 | 78.0 | 83.9 | 69.9 | 12.0 |
| MSE-Net | PyTorch | 89.9 | 79.4 | 84.9 | 71.1 | 53.3 |
| MSE-Net | ONNX | 88.9 | 80.8 | 84.5 | 70.5 | 36.8 |
| MSE-Net | TensorRT | 88.9 | 80.8 | 84.5 | 70.5 | 25.5 |

On Jetson Orin Nano, MSE-Net reaches **36.8 ms** in ONNX format (approximately **27.2 FPS**) and **25.5 ms** in TensorRT format (approximately **39.2 FPS**). Under the same Jetson ONNX deployment setting, MSE-Net improves recall, mAP@0.5, and mAP@0.5:0.95 by 2.7, 0.6, and 0.6 percentage points, respectively, compared with YOLOv11n.

---

## 🚀Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/XinyueChe311/MSE-Net-maize-leaf-disease-detection.git
cd MSE-Net-maize-leaf-disease-detection
```

### 2. Create the Python environment

```bash
conda create -n msenet python=3.10 -y
conda activate msenet
```

### 3. Install dependencies

The experiments reported in the paper used Python 3.10, PyTorch 2.4.1+cu121, and Ultralytics 8.4.143. Install the matching dependencies in the source-code environment.

```bash
# Change to the repository's actual source-code directory first.
# Install the dependencies using the dependency file provided there.
```

> The exact source directory and dependency manifest should be verified against the checked-out repository version before using an installation command. Do not assume a root-level `requirements.txt` or `pyproject.toml` exists.

---

## Training

### Dataset configuration

Prepare the dataset in YOLO format and create a dataset YAML file. The following is an example configuration; replace the paths with the actual local dataset paths.

```yaml
path: /path/to/dataset
train: images/train
val: images/val
test: images/test

nc: 9
names:
  0: Common Rust
  1: Gray Leaf Spot
  2: Corn Smut
  3: Southern Rust
  4: Philippine Downy Mildew
  5: Maize Stripe Virus Disease
  6: Corn Borer Damage
  7: Corn Borer Larvae
  8: Healthy
```

### Training configuration used in the paper

| Argument | Value |
|---|---:|
| Model | MSE-Net configuration |
| Image size | 640 |
| Epochs | 250 |
| Patience | 60 |
| Batch size | 16 |
| Optimizer | SGD |
| Workers | 0 |
| Seeds | 0, 42, 123 |
| Cache | RAM |
| AMP | True |

### Training command template

```bash
yolo detect train \
    model=/path/to/MSE-Net.yaml \
    data=/path/to/data.yaml \
    imgsz=640 \
    epochs=250 \
    patience=60 \
    batch=16 \
    optimizer=SGD \
    workers=0 \
    device=0 \
    seed=0 \
    cache=ram \
    amp=True
```

Run the experiment with seeds 0, 42, and 123 to reproduce the multi-seed evaluation protocol. The model YAML path must point to the actual MSE-Net configuration in the checked-out source tree.

---

## 🔍Inference and Validation

### Prediction

```bash
yolo detect predict \
    model=/path/to/best.pt \
    source=/path/to/images \
    imgsz=640
```

### Validation

```bash
yolo detect val \
    model=/path/to/best.pt \
    data=/path/to/data.yaml \
    imgsz=640 \
    split=test
```

Use the test subset (Set 3) for final test evaluation. Keep the dataset configuration, image size, and evaluation settings consistent with the reported experimental protocol.

---

## 🔥Visualization

The following figures are intended to document the model structure and experimental findings. Insert the actual images from the repository's `Figures/` directory and update each relative path to its exact filename.

### Detection Results Comparison

<img width="692" height="576" alt="image" src="https://github.com/user-attachments/assets/980d8d5c-7c93-4ff6-b24a-4b76638730f6" />

We compare MSE-Net with representative object detectors, including YOLOv5n, YOLOv8n, YOLOv10n, YOLOv11n, YOLOv12n, and RT-DETR variants.

MSE-Net demonstrates advantages in the following challenging scenarios:
- 🔬 Small-scale lesions — improved detection of small and weakly contrasted disease regions, particularly scattered lesions associated with Southern Rust.
- 📦 Complex and dense disease regions — improved detection of Corn Borer Damage and Corn Borer Larvae, which are challenging targets for lightweight object detectors.
- 🎯 Irregular lesion patterns — improved localization of disease regions under complex field backgrounds, helping reduce missed and incorrect detections.
In representative test images, MSE-Net can identify Corn Borer Damage and Corn Borer Larvae, which are difficult to distinguish in some lightweight baseline models. For healthy leaf samples, the model also produces the Healthy prediction without additional disease detections.

Although some small or weakly contrasted lesions remain challenging, MSE-Net achieves competitive detection performance while maintaining a compact model architecture.

### Grad-CAM Heatmap Analysis

<img width="692" height="490" alt="image" src="https://github.com/user-attachments/assets/54cdb94e-2cd3-4772-b20a-308797fdc57a" />

Based on Grad-CAM, we further investigate the feature activation patterns of different module configurations across the nine target categories.

The visualization results reveal the following characteristics:
Configuration	Attention Pattern
YOLOv11n (Baseline)	Relatively dispersed activation regions, with responses influenced by leaf vein textures and background structures.
+ MAFPN	Enhances multi-scale feature fusion and helps preserve disease details at different spatial scales.
+ SimAM	Recalibrates spatial features and increases the responses of informative spatial regions.
+ ECA	Enhances channel-wise feature selection and emphasizes disease-related feature channels.
+ SimAM + ECA	Combines spatial and channel-wise feature recalibration.
+ MAFPN + SimAM	Integrates multi-scale feature fusion and spatial feature recalibration.
+ MAFPN + ECA	Combines multi-scale feature fusion with channel-wise feature selection.
MSE-Net (Full Model)	Combines all three modules, producing more concentrated activation regions in some samples and improving the correspondence between activated regions and disease areas.
The combined effects of MAFPN, SimAM, and ECA improve feature representation from complementary perspectives.
---

## Project Structure

The directory names below should match the version actually committed to the GitHub repository. The currently documented top-level resources include:

```text
MSE-Net-maize-leaf-disease-detection/
├── Figures/                  # Architecture and experimental figures
├── datasets/                 # Dataset resources
├── yolov11-main-new/             # Source-code directory (verify against current repository)
├── .gitattributes            # Git LFS tracking rules
├── .gitignore
└── README.md
```

---

## 📝Citation

If you use this work, please cite the corresponding paper. Replace the fields below with the final author and publication information before the paper is published.

```bibtex
@article{MSE-Net,
  title   = {A Lightweight Multi-scale Detection Network for Corn Leaf Disease Detection in Complex Field Environments},
  author={Xinyue,Che；Zhangpeng,Luo}
  journal={},
  year={},
  url={https://github.com/XinyueChe311/MSE-Net-maize-leaf-disease-detection}
}
```

---

## 🙏Acknowledgements

This project is based on the open-source Ultralytics YOLO framework. We acknowledge the developers and contributors of Ultralytics and the providers of the public Roboflow datasets used to construct the corn disease image dataset.

- [Ultralytics](https://github.com/ultralytics/ultralytics)
- [Roboflow](https://roboflow.com/)

Please consult the original dataset pages for their individual licenses and attribution requirements.

---

## 📜License

Add the applicable `LICENSE` file to the repository and specify the corresponding license here. Until a license is included, users should not assume that the code, trained weights, or dataset may be reused or redistributed.

---

## 📧Contact

For questions, bug reports, or discussions, please open an issue in this repository:

[GitHub Issues](https://github.com/XinyueChe311/MSE-Net-maize-leaf-disease-detection/issues)

<div align="center">

**MSE-Net: Lightweight corn leaf disease detection for complex field environments.**

</div>
