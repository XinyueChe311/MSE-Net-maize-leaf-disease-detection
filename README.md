# MSE-Net: A Lightweight and Accurate Object Detection Network for Corn Leaf Disease Detection in Complex Field Environments


<img width="692" height="436" alt="image" src="https://github.com/user-attachments/assets/0d63081f-26d0-4b67-b303-2fcfdacc65de" />


> MSE-Net is a lightweight object detection network developed for corn leaf disease detection under complex field conditions. Built upon YOLOv11n, MSE-Net integrates Multi-scale Adaptive Feature Pyramid Network (MAFPN), Simple Attention Module (SimAM), and Efficient Channel Attention (ECA) to enhance multi-scale feature representation and disease-related feature extraction. With a compact architecture, MSE-Net provides an effective solution for accurate corn disease detection and edge-device deployment.

---

## 📋 Table of Contents

- [Overview](#-overview)
- [What's New](#-whats-new)
- [Architecture](#-architecture)
- [Dataset](#-dataset)
- [Main Results](#-main-results)
- [Quick Start](#-quick-start)
- [Training](#-training)
- [Inference & Validation](#-inference--validation)
- [Visualization](#-visualization)
- [Project Structure](#-project-structure)
- [Citation](#-citation)
- [Acknowledgements](#-acknowledgements)
- [License](#-license)
- [Contact](#-contact)

---

## 📌 Overview

Corn is one of the most important cereal crops worldwide. However, various diseases and pests can significantly affect corn growth, yield, and quality. Accurate and timely identification of corn leaf diseases is essential for effective field management and precision agriculture.

In complex field environments, early-stage disease symptoms often exhibit small lesion areas, weak texture features, and indistinct boundaries. Furthermore, leaf occlusion, illumination variations, and complex background interference make accurate disease detection particularly challenging.

To address these issues, we propose MSE-Net, a lightweight object detection network based on YOLOv11n. By incorporating MAFPN, SimAM, and ECA, the proposed network enhances multi-scale feature fusion and attention-based feature representation while maintaining a compact model architecture.

### 🎯 Core Challenges Addressed

| Challenge | Solution in MSE-Net |
|---|---|
| Small-scale disease lesions | MAFPN enhances multi-scale feature representation and fusion |
| Complex disease morphology | Multi-scale feature fusion improves the representation of disease-related patterns |
| Background interference | SimAM enhances spatial feature responses without introducing additional learnable parameters |
| Unequal channel importance | ECA adaptively recalibrates channel-wise feature responses |
| Edge-device deployment | Lightweight architecture with low parameter count and computational cost |

---

## 🆕 What's New

- **MAFPN (Multi-scale Adaptive Feature Pyramid Network)** — Introduces multi-scale feature fusion to improve the representation of disease lesions with different sizes and morphological characteristics.
- **SimAM (Simple Attention Module)** — Enhances informative spatial features through a parameter-free attention mechanism, helping the network focus on disease-related regions.
- **ECA (Efficient Channel Attention)** — Models local cross-channel interactions to improve channel-wise feature discrimination with low computational overhead.
- **MSE-Net** — Integrates MAFPN, SimAM, and ECA into the YOLOv11n framework, balancing detection performance and model complexity for practical deployment.

---

## 🏗️ Architecture

MSE-Net is developed based on YOLOv11n, with three targeted improvements to strengthen feature extraction and representation.

### Overall Architecture

<img width="692" height="436" alt="image" src="https://github.com/user-attachments/assets/505355cf-cb40-4683-8981-88129b815077" />


### Module-level Modifications

| Component | Module | Main Function |
|---|---|---|
| Baseline | YOLOv11n | Basic object detection framework |
| Feature fusion | MAFPN | Multi-scale feature aggregation and fusion |
| Attention mechanism | SimAM | Spatial feature enhancement |
| Attention mechanism | ECA | Efficient channel-wise feature recalibration |
| Proposed model | MSE-Net | Integrated lightweight disease detection network |

### MAFPN Module

MAFPN is introduced to improve multi-scale feature representation and information fusion. It facilitates the integration of features from different network stages, helping the detector capture disease lesions with diverse scales and morphological characteristics.

### SimAM Module

SimAM is a parameter-free attention module that evaluates the importance of individual neurons based on their energy functions. It enhances informative features and suppresses less relevant responses, encouraging the network to focus on disease-related regions.

### ECA Module

ECA employs efficient local cross-channel interactions to capture channel dependencies without dimensionality reduction. It improves the representation of informative channels while maintaining low computational complexity.


<img width="692" height="391" alt="image" src="https://github.com/user-attachments/assets/7c295c33-2479-42ea-827b-360728bc2754" />

---

## 📊 Dataset

### Dataset Overview

A corn leaf disease object detection dataset was constructed by integrating publicly available images and field-collected photographs. The images were carefully screened and annotated to provide samples under diverse visual conditions.

The dataset includes nine categories covering common corn diseases, pest damage, insect larvae, and healthy leaves.

### Disease Categories

| No. | Disease Category | Chinese Name |
|---|---|---|
| 1 | Common Rust | 普通锈病 |
| 2 | Gray Leaf Spot | 灰斑病 |
| 3 | Corn Smut | 玉米黑穗病 |
| 4 | Southern Rust | 南方锈病 |
| 5 | Philippine Downy Mildew | 菲律宾玉米霜霉病 |
| 6 | Maize Stripe Virus | 玉米条纹病毒病 |
| 7 | Corn Borer Damage | 玉米螟危害 |
| 8 | Corn Borer Larva | 玉米螟幼虫 |
| 9 | Healthy | 健康 |

### Dataset Versions

Two dataset versions are provided:

| Dataset | Description | Number of Images |
|---|---|---:|
| dataset_final_v3_1_fixed | Original curated dataset | 3,600 |
| dataset_final_v3_1_fixed_aug2x | Augmented dataset | 6,120 |

### Data Split

The original curated dataset contains 3,600 images, divided into training, validation, and test sets.

| Subset | Images | Proportion |
|---|---:|---:|
| Training | 2,520 | 70% |
| Validation | 720 | 20% |
| Test | 360 | 10% |
| Total | 3,600 | 100% |

The annotations follow the standard YOLO format (`.txt`). Each annotation file contains the class index and normalized bounding-box coordinates.

### Dataset Construction


<img width="691" height="306" alt="image" src="https://github.com/user-attachments/assets/69814948-8d9d-4c85-ad22-e0a14e115fe9" />


> The dataset is intended for research on automated corn leaf disease detection under complex field conditions.

---

## 📈 Main Results

### Model Performance

MSE-Net is evaluated against representative YOLO-series detectors and RT-DETR models to assess its detection performance.


<img width="1142" height="735" alt="image" src="https://github.com/user-attachments/assets/6f71e766-dba4-411c-824c-ccf90e00b45c" />


### Model Complexity

MSE-Net maintains a compact architecture while incorporating additional feature fusion and attention mechanisms.

| Metric | MSE-Net |
|---|---:|
| Model Size | 4.8 MB |
| Parameters | 2.25 M |
| FLOPs | 5.9 GFLOPs |

### Edge-device Deployment

MSE-Net is further evaluated on the NVIDIA Jetson Orin Nano edge platform to investigate its deployment feasibility.

<img width="1076" height="526" alt="image" src="https://github.com/user-attachments/assets/a1e544df-762b-4f2f-b862-d9d357bc3c80" />

The deployment results indicate the potential of MSE-Net for resource-constrained edge computing applications.

---

## 🚀 Quick Start

### Clone the Repository

```bash
git clone https://github.com/XinyueChe311/MSE-Net-maize-leaf-disease-detection.git
cd MSE-Net-maize-leaf-disease-detection
```

### Environment Setup

We recommend using Python 3.10.

```bash
conda create -n msenet python=3.10 -y
conda activate msenet
```

Install the required dependencies:

```bash
pip install -e .
```

Make sure that the installed PyTorch and CUDA versions are compatible with your hardware.

### System Requirements

| Item | Configuration |
|---|---|
| Python | 3.10 |
| Deep Learning Framework | PyTorch |
| Detection Framework | Ultralytics YOLOv11 |
| GPU | NVIDIA GPU recommended |
| Operating System | Linux |

---

## 🏋️ Training

### Dataset Preparation

Organize the dataset in the following YOLO format:

```text
datasets/
├── dataset_final_v3_1_fixed/
│   ├── images/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   ├── labels/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── data.yaml
│
└── dataset_final_v3_1_fixed_aug2x/
    ├── images/
    ├── labels/
    └── data.yaml
```

### Model Configuration

The MSE-Net model configuration is provided in:

```text
ultralytics/cfg/models/11/yolo11n_MAFPN_SimAM_ECA.yaml
```

### Start Training

The following example demonstrates how to initialize and train MSE-Net using the Ultralytics API:

from linecache import cache
import warnings

from huggingface_hub.cli.inference_endpoints import resume

from ultralytics import YOLO, RTDETR

warnings.filterwarnings("ignore")
if __name__ == '__main__':
    # 1. 初始化模型
    model = RTDETR(r'/root/autodl-tmp/ultralytics/ultralytics/cfg/models/rt-detr/rtdetr-x.yaml')

    # 2. 训练模型
    results = model.train(
        data=r'/root/autodl-tmp/dataset_final_v3_1_fixed/data.yaml ',
        epochs=250,
        batch=16,
        imgsz=640,
        device=0,
        patience=60,
        workers=0, 
        cache='ram', 
        amp=True,  
        pretrained=False,
        project="runs",
        name="rtdetr-x_seed123",
        optimizer='SGD',
        resume=False,
        seed=123,
    )
    # 3. 模型验证
    metrics = model.val()

> Note: Adjust the training parameters according to your hardware and experimental configuration. For reproducibility, use the corresponding configuration files and random seeds provided in the repository.

---

## 🔍 Inference & Validation

### Single Image Inference

```python
from ultralytics import YOLO

model = YOLO("weights/seed0/best.pt")

results = model.predict(
    source="path/to/image.jpg",
    imgsz=640,
    conf=0.25,
    save=True
)
```

### Folder Inference

```python
results = model.predict(
    source="path/to/images",
    imgsz=640,
    conf=0.25,
    save=True
)
```

### Model Validation

```python
from ultralytics import YOLO

model = YOLO("weights/seed0/best.pt")

metrics = model.val(
    data="datasets/dataset_final_v3_1_fixed/data.yaml",
    imgsz=640,
    batch=16,
    device=0
)
```

---

## 🔥 Visualization
### Detection  Results Comparison

<img width="692" height="576" alt="image" src="https://github.com/user-attachments/assets/51833117-7a44-4702-802c-0f9353535dd1" />

### Grad-CAM Heatmap Analysis


<img width="692" height="490" alt="image" src="https://github.com/user-attachments/assets/99b87da0-4dc8-4f51-b839-0cd54a5e3043" />


Grad-CAM is used to analyze feature response distributions and investigate the effects of different module configurations on feature extraction.


---

## 📁 Project Structure

```text
MSE-Net-maize-leaf-disease-detection/
│
├── Figures/
│   ├── data_augmentation/
│   ├── gradcam/
│   ├── nine_class_detections/
│   ├── separate_curves/
│   └── ...
│
├── datasets/
│   ├── dataset_final_v3_1_fixed/
│   └── dataset_final_v3_1_fixed_aug2x/
│
├── ultralytics/
│   └── ...
│
├── configs/
│   └── ...
│
├── weights/
│   ├── seed0/
│   ├── seed42/
│   └── seed123/
│
├── results/
│   ├── seed0/
│   ├── seed42/
│   └── seed123/
│
├── docs/
├── examples/
├── tests/
│
├── pyproject.toml
├── LICENSE
├── README.md
└── README.zh-CN.md
```

---

## 📝 Citation

If you find this work useful in your research, please consider citing:

```bibtex
@article{MSE-Net,
  title={MSE-Net: A Lightweight and Accurate Object Detection Network for Corn Leaf Disease Detection in Complex Field Environments},
  author={Che, Xinyue；Luo，Zhanpeng},
  journal={},
  year={2026},
  note={Research project}
}
```

Please update the citation information when the corresponding paper is formally published.

---

## 🙏 Acknowledgements

- **Ultralytics** — For providing the YOLO framework and related tools.
- **YOLOv11** — For serving as the baseline object detection architecture.
- **Roboflow** — For providing publicly available image annotation and dataset resources.
- All contributors who supported dataset collection, annotation, and experimental evaluation.

---

## 📜 License

This project is distributed under the license specified in the `LICENSE` file.

---

## 📧 Contact

For questions, suggestions, or collaboration, please open an issue in this repository.

- GitHub: [@XinyueChe311](https://github.com/XinyueChe311)
- Repository: [MSE-Net-maize-leaf-disease-detection](https://github.com/XinyueChe311/MSE-Net-maize-leaf-disease-detection)

---

⭐ If you find this project useful, please consider giving it a star! ⭐
