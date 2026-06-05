# 🌽 Maize Disease Classification with EfficientNetV2-S

A deep learning pipeline for classifying maize (corn) leaf diseases using **EfficientNetV2-S** and PyTorch. Achieves **≥93% test accuracy** and **≥0.92 macro F1** across 4 disease categories.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Classes](#classes)
- [Project Structure](#project-structure)
- [Setup & Usage](#setup--usage)
- [Model Architecture](#model-architecture)
- [Training Strategy](#training-strategy)
- [Results](#results)
- [Dataset](#dataset)
- [Requirements](#requirements)

---

## Overview

This project builds a robust, production-ready classifier to detect maize leaf diseases from images. Key features include:

- **Duplicate detection** via perceptual hashing (pHash) to ensure clean data splits
- **Two-phase training**: head warm-up (frozen backbone) → full fine-tuning
- **WeightedRandomSampler** to handle class imbalance
- **Grad-CAM visualisations** for model interpretability
- **Comprehensive evaluation**: ROC curves, PR curves, confusion matrix, per-class metrics

---

## Classes

| ID | Class | Display Name |
|----|-------|-------------|
| 0  | `gray_leaf_spot` | Gray Leaf Spot |
| 1  | `common_rust` | Common Rust |
| 2  | `northern_leaf_blight` | Northern Leaf Blight |
| 3  | `healthy` | Healthy |

---

## Project Structure

```
maize_disease_project/
├── final_maze1_0.ipynb         ← Main Colab notebook (full pipeline)
├── README.md
├── requirements.txt
├── .gitignore
└── assets/
    └── (sample output figures)
```

**Google Drive output structure (generated at runtime):**

```
maize_disease_project/
├── datasets/           ← Raw image data
├── metadata/           ← CSV metadata files + project_config.json
├── splits/             ← train.csv / val.csv / test.csv
├── checkpoints/        ← Saved model weights (.pth)
├── figures/            ← EDA + evaluation figures (300 DPI PNGs)
├── reports/            ← test_metrics.json, classification_report.txt
├── gradcam/            ← Grad-CAM visualisation images
├── suspicious_samples/ ← Misclassified / duplicate CSVs
└── final_package/      ← Packaged ZIP for download
```

---

## Setup & Usage

### 1. Open in Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/YOUR_REPO/blob/main/final_maze1_0.ipynb)

> **Note:** Replace `YOUR_USERNAME/YOUR_REPO` with your actual GitHub path after uploading.

### 2. Runtime Requirements

- **GPU required** — Go to `Runtime → Change runtime type → GPU (T4 or better)`
- Recommended VRAM: ≥12 GB for `BATCH_SIZE=32`; use 16 for smaller GPUs

### 3. Run the Cells in Order

| Cell | Description |
|------|-------------|
| **Cell 1** | Install dependencies, import libraries, GPU check |
| **Cell 2** | Mount Google Drive, define paths & hyperparameters |
| **Cell 3** | Download dataset (via Kaggle API) |
| **Cell 4** | EDA: class distribution, image dimensions, sample grid |
| **Cell 5** | Data cleaning: corruption check, duplicate removal (pHash) |
| **Cell 6** | Train/Val/Test split (stratified) |
| **Cell 7** | Build model, augmentation pipeline, DataLoaders |
| **Cell 8** | Phase 1 training — head warm-up (backbone frozen) |
| **Cell 9** | Phase 2 training — full fine-tuning |
| **Cell 10** | Evaluation: confusion matrix, ROC, PR curves |
| **Cell 11** | Grad-CAM visualisation |
| **Cell 12** | Package & download all thesis outputs as ZIP |

### 4. Kaggle API Key

Place your `kaggle.json` in Google Drive before running Cell 3, or follow the in-notebook instructions.

---

## Model Architecture

| Component | Detail |
|-----------|--------|
| Backbone | `efficientnetv2_s` (via `timm`) |
| Pretrained | ImageNet weights |
| Head | Global avg pool → Dropout(0.3) → Linear(num_classes) |
| Input size | 224 × 224 |
| Normalisation | ImageNet mean/std |

---

## Training Strategy

### Phase 1 — Head Warm-up
- **Epochs:** 5
- **Backbone:** frozen
- **LR (head):** 3e-4

### Phase 2 — Full Fine-tuning
- **Max epochs:** 60 (with early stopping, patience=12)
- **LR (head):** 3e-4 | **LR (backbone):** 3e-5 (10× lower)
- **Scheduler:** CosineAnnealingLR (min LR = 1e-6)
- **Optimiser:** AdamW, weight decay = 1e-4
- **Loss:** CrossEntropyLoss with label smoothing = 0.05
- **Imbalance:** WeightedRandomSampler (no double-counting with loss weights)
- **Mixed precision:** AMP (torch.cuda.amp)
- **Gradient clipping:** max norm = 1.0

### Augmentation Pipeline (Albumentations)

| Augmentation | Parameters |
|---|---|
| RandomResizedCrop | scale=(0.7, 1.0) |
| HorizontalFlip | p=0.5 |
| VerticalFlip | p=0.3 |
| RandomRotate90 | p=0.4 |
| ColorJitter | brightness/contrast/sat/hue |
| GridDistortion | p=0.2 |
| CoarseDropout | p=0.3 |
| Normalize | ImageNet mean/std |

---

## Results

### Quality Gate Targets

| Metric | Target |
|--------|--------|
| Test Accuracy | ≥ 0.93 |
| Macro F1 | ≥ 0.92 |
| Gray Leaf Spot F1 | ≥ 0.85 |
| Northern Leaf Blight F1 | ≥ 0.88 |
| Healthy F1 | ≥ 0.95 |
| Common Rust F1 | ≥ 0.95 |
| Val–Test accuracy gap | ≤ 0.04 |

---

## Dataset

The notebook uses the **[Maize Disease Dataset](https://www.kaggle.com/)** available on Kaggle.

- **Raw images:** ~15,356
- **After deduplication (pHash):** ~8,427 unique images
- **Split:** 70% train / 15% val / 15% test (stratified)

---

## Requirements

See [`requirements.txt`](requirements.txt) for the full list. Key dependencies:

```
torch>=2.0
timm>=0.9
albumentations>=1.3
grad-cam>=1.4
imagehash>=4.3
torchmetrics>=1.0
```

---

## Citation / Acknowledgements

- EfficientNetV2: [Tan & Le, 2021](https://arxiv.org/abs/2104.00298)
- timm model zoo: [rwightman/pytorch-image-models](https://github.com/huggingface/pytorch-image-models)
- Grad-CAM: [pytorch-grad-cam](https://github.com/jacobgil/pytorch-grad-cam)
- Albumentations: [albumentations-team/albumentations](https://github.com/albumentations-team/albumentations)

---

## License

This project is for academic / thesis use. See [LICENSE](LICENSE) for details.
