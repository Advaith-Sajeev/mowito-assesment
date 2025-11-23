# Text Scratch Detection - Mowito Assessment

Deep learning models for detecting and segmenting scratches on text images.

**GitHub:** https://github.com/Advaith-Sajeev/mowito-assesment  
**Models (Google Drive):** [Download All Models](https://drive.google.com/drive/folders/1Rr81oR3aud6hXNRk950853TZat9u3arW?usp=drive_link)

---

## 🏆 Best Models

### 1. Best Classification + Segmentation: UNeXt Multi-task ⭐

**Performance:** Precision: 1.000 | Recall: 0.971 | F1: 0.985  
**Bonus:** Provides segmentation masks for scratch localization

Perfect precision with excellent recall. Jointly trained for classification and segmentation, providing interpretable results with precise scratch masks.

### 2. Best Classification Only: EfficientNetB0 ⭐

**Performance:** Precision: 1.000 | Recall: 0.990 | F1: 0.995  
**Advantage:** Lightweight (16 MB), fastest inference

Perfect precision (no false alarms) with near-perfect recall. Ideal for production deployment when segmentation is not required.

---

## 📋 Table of Contents

1. [Quick Start](#-quick-start)
2. [Dataset Setup](#-dataset-setup)
3. [Model Validation](#-model-validation)
4. [Results Summary](#-results-summary)
5. [GAN Models (Google Drive)](#-gan-models-google-drive)

---

## 🚀 Quick Start

### 1. Installation

```bash
# Clone repository
git clone https://github.com/Advaith-Sajeev/mowito-assesment.git
cd mowito-assesment

# Install dependencies
pip install -r requirements.txt
```

### 2. Prepare Your Data

See [Dataset Setup](#-dataset-setup) section below.

### 3. Run Validation

**Option 1: UNeXt Multi-task (Classification + Segmentation) ⭐**
```bash
python src/unext_multitask/val.py \
    --model_path models/unext_multitask/scratch_UNext_multitask_subset_2.pth \
    --data_root data
```

**Option 2: EfficientNetB0 (Classification Only) ⭐**
```bash
python src/classification/validate_efficientnet.py \
    --model_path models/classification/efficientnet_b0.pth \
    --data_root data
```

---

## 📂 Dataset Setup

### Required Directory Structure

Your dataset must follow this structure:

```
data/
├── train/
│   ├── good/     # Clear text images (training)
│   └── bad/      # Scratched text images (training)
├── val/
│   ├── good/     # Clear text images (validation)
│   └── bad/      # Scratched text images (validation)
└── masks/
    ├── train/
    │   ├── good/ # Zero masks for good images
    │   └── bad/  # Scratch masks for bad images
    └── val/
        ├── good/ # Zero masks for good images
        └── bad/  # Scratch masks for bad images
```

### Automatic Dataset Creation

If you have raw data in this format:
```
raw_data/
├── good/   # All good images
├── bad/    # All bad images
└── masks/  # Masks for bad images
```

Run the dataset preparation script:

```bash
# Basic usage (uses defaults)
python src/utils/create_dataset.py

# Custom source and output directories
python src/utils/create_dataset.py \
    --source /path/to/raw_data \
    --output data

# Custom validation ratio (20% instead of default 10%)
python src/utils/create_dataset.py \
    --source raw_data \
    --output data \
    --val-ratio 0.20
```

**Available Options:**
- `--source`: Source directory containing `good/`, `bad/`, and `masks/` folders (default: `anomaly_detection_test_data_synt`)
- `--output`: Output directory for processed dataset (default: `data_synth`)
- `--val-ratio`: Validation split ratio, e.g., 0.10 for 10% (default: `0.10`)

**What it does:**
- Balances good/bad classes
- Creates 90/10 train/val split (or custom ratio with `--val-ratio`)
- Generates zero masks for good images
- Copies masks for bad images
- Outputs to specified directory in required structure

---

## ✅ Model Validation

### 1. UNeXt Multi-task (Best Classification + Segmentation) ⭐

**Location:** `models/unext_multitask/scratch_UNext_multitask_subset_2.pth` (5.6 MB - in repo)

**Validation Script:**
```bash
python src/unext_multitask/val.py \
    --model_path models/unext_multitask/scratch_UNext_multitask_subset_2.pth \
    --data_root data
```

**Options:**
- `--model_path`: Direct path to model `.pth` file (required)
- `--data_root`: Path to data directory containing `train/` and `val/` (default: `../../data`)
- `--output_root`: Where to save validation outputs (default: `outputs`)

**Outputs:**
- Classification predictions (good/bad)
- Segmentation masks (scratch localization)
- Evaluation metrics (precision, recall, IoU)
- Overlaid visualizations

**Performance:** Precision: 1.000 | Recall: 0.971 | F1: 0.985

**Why it's best for cls+seg:**
- Perfect precision (no false positives)
- Joint training: classification & segmentation help each other
- Provides interpretable masks showing exactly where scratches are
- Lightweight (5.6 MB) yet powerful

**📊 View Results:**
- [Confusion Matrix](results/UNext_multitask/confusion_matrix_val.png)
- [Validation Report (HTML)](results/UNext_multitask/val_report.html)
- [Validation Panels (102 images)](results/UNext_multitask/val_panels/)

---

### 2. EfficientNetB0 (Best Classification Only) ⭐

**Location:** `models/classification/efficientnet_b0.pth` (16 MB - in repo)

**Validation Script:**
```bash
python src/classification/validate_efficientnet.py \
    --model_path models/classification/efficientnet_b0.pth \
    --data_root data
```

**Expected Output:**
- Validation accuracy, precision, recall per class
- Confusion matrix
- Sample predictions saved to `val_outputs_efficientnet/`

**Performance:** Precision: 1.000 | Recall: 0.990 | F1: 0.995

**Why it's best for classification only:**
- Perfect precision (zero false alarms)
- Excellent recall (99% catch rate)
- Fastest inference speed
- Lightweight (16 MB) and production-ready
- Best F1 score among classification models

**📊 View Results:**
- [Confusion Matrix](results/efficientnet/confusion_matrix_val_efficientnet.png)
- [Misclassified Example](results/efficientnet/misclassified_val/false_negative_good/00003_09_08_2024_18_21_00.844774_classifier_input.png) (1 false negative)

---

### 3. ResNet50 (Alternative - Best Recall)

**Location:** `models/classification/resnet50.pth` (90 MB - in repo)

**Validation Script:**
```bash
python src/classification/validate_resnet50.py \
    --model_path models/classification/resnet50.pth \
    --data_root data
```

**Performance:** Precision: 0.953 | Recall: 1.000 (catches ALL scratches) | F1: 0.976

**📊 View Results:**
- [Confusion Matrix](results/resnet50/confusion_matrix_val.png)
- [Misclassified Examples](results/resnet50/misclassified_val/false_positive_bad/) (5 false positives)

---

## 📊 Results Summary

### Best Models by Category

| Category | Model | Precision | Recall | F1 | Size | Location |
|----------|-------|-----------|--------|-------|------|----------|
| **Cls + Seg** | **UNeXt Multi-task** ⭐ | **1.000** | 0.971 | 0.985 | 5.6 MB | ✅ In repo |
| **Cls Only** | **EfficientNetB0** ⭐ | **1.000** | **0.990** | **0.995** | 16 MB | ✅ In repo |

### All Models

| Model | Precision | Recall | F1 | Size | Location |
|-------|-----------|--------|-------|------|----------|
| **UNeXt Multi-task** | **1.000** | 0.971 | 0.985 | 5.6 MB | ✅ In repo |
| **EfficientNetB0** | **1.000** | **0.990** | **0.995** | 16 MB | ✅ In repo |
| ResNet50 | 0.953 | **1.000** | 0.976 | 90 MB | ✅ In repo |
| UNeXt + GradCAM | ~1.000 | 0.971 | 0.985 | 5.6 MB | 📁 Google Drive |

**All metrics:** Validation set (10% of data, balanced classes)

---

## 📥 GAN Models (Google Drive)

Synthetic data generation models are hosted on Google Drive due to file size.

**Download:** [Google Drive Folder](https://drive.google.com/drive/folders/1Rr81oR3aud6hXNRk950853TZat9u3arW?usp=drive_link)

### Available Models

1. **UNeXt-GradCAM** (5.6 MB)
   - Classification model with GradCAM segmentation
   - Download and place in: `models/unext_gradcam/scratch_UNext_cls.pth`

2. **Pix2Pix GAN** (171 MB total)
   - Generator: 160 MB (generates scratch patterns)
   - Discriminator: 11 MB
   - Download and place in: `models/synthetic_data/pix2pix_gan/`

3. **Vanilla GAN** (if available)
   - Download and place in: `models/synthetic_data/vanilla_gan/`

### Using GAN Models

#### UNeXt-GradCAM (Classification + Post-hoc Segmentation)

```bash
# After downloading from Google Drive
python src/unext_gradcam/gradcam_infer.py \
    --name scratch_UNext_cls \
    --phase val \
    --cam_thresh 0.4 \
    --scratch_size_threshold 0.0
```

**Outputs:** Classification + GradCAM-based segmentation masks

#### Pix2Pix GAN (Synthetic Data Generation)

```bash
# Generate synthetic scratched images
python src/synthetic_data/pix2pix_gan/generate_synthetic_pix2pix.py \
    --generator_path models/synthetic_data/pix2pix_gan/generator_best.pth \
    --input_dir data/train/good \
    --output_dir synthetic_outputs
```

---

## 🗂️ Repository Structure

```
mowito-assesment/
├── models/                         # Trained weights
│   ├── classification/             # ✅ In repo
│   │   ├── efficientnet_b0.pth    (16 MB) ⭐ BEST
│   │   └── resnet50.pth           (90 MB)
│   └── unext_multitask/           # ✅ In repo
│       └── scratch_UNext_multitask_subset_2.pth (5.6 MB)
│
├── src/
│   ├── classification/             # ResNet50 & EfficientNetB0
│   │   ├── train_resnet50.py
│   │   ├── train_efficientnet.py
│   │   ├── validate_resnet50.py
│   │   └── validate_efficientnet.py
│   ├── unext_multitask/           # Multi-task learning
│   ├── unext_gradcam/             # GradCAM segmentation
│   ├── synthetic_data/            # GAN implementations
│   └── utils/
│       └── create_dataset.py      # Data preparation script
│
├── results/                        # Training logs
└── requirements.txt               # Dependencies
```

---

## 💻 Development

### Training Models

Refer to source code in `src/` for training scripts:
- ResNet50: `src/classification/train_resnet50.py`
- EfficientNetB0: `src/classification/train_efficientnet.py`
- UNeXt Multi-task: `src/unext_multitask/train.py`
- UNeXt GradCAM: `src/unext_gradcam/train.py`

### Synthetic Data Generation

Three methods implemented:
1. **OpenCV-based:** `src/synthetic_data/opencv_method/make_synthetic_bad.py`
2. **Vanilla GAN:** `src/synthetic_data/vanilla_gan/`
3. **Pix2Pix GAN:** `src/synthetic_data/pix2pix_gan/`

---

## 📄 Citation

If using this code, please cite:

```
Mowito Assessment Project - Text Scratch Detection
GitHub: https://github.com/Advaith-Sajeev/mowito-assesment
```

UNeXt architecture:
```
@article{valanarasu2022unext,
  title={UNeXt: MLP-based Rapid Medical Image Segmentation Network},
  author={Valanarasu, Jeya Maria Jose and Patel, Vishal M},
  journal={arXiv preprint arXiv:2203.04967},
  year={2022}
}
```

---

## 📧 Contact

For questions or issues: Open an issue on GitHub or contact repository owner.

**Google Drive Models:** [Download Here](https://drive.google.com/drive/folders/1Rr81oR3aud6hXNRk950853TZat9u3arW?usp=drive_link)
