# Text Scratch Detection - Mowito Assessment

Deep learning models for detecting and segmenting scratches on text images.

**GitHub:** https://github.com/Advaith-Sajeev/mowito-assesment  
**Models (Google Drive):** [Download All Models](https://drive.google.com/drive/folders/1Rr81oR3aud6hXNRk950853TZat9u3arW?usp=drive_link)

---

## 🏆 Best Model: EfficientNetB0

**Performance:** Precision: 1.000 | Recall: 0.990 | F1: 0.995 ⭐

The best overall model for production use. Perfect precision (no false alarms) with excellent recall.

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

```bash
# Validate EfficientNetB0 (Best Model)
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
python src/utils/create_dataset.py
```

**What it does:**
- Balances good/bad classes
- Creates 90/10 train/val split
- Generates zero masks for good images
- Copies masks for bad images
- Outputs to `data/` directory in required structure

**Configuration:** Edit `create_dataset.py` lines 10-13 to change:
- Source data location (default: `anomaly_detection_test_data_synt`)
- Output location (default: `data_synth`)
- Validation ratio (default: 0.10)

---

## ✅ Model Validation

### 1. EfficientNetB0 (Best Overall) ⭐

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

---

### 2. ResNet50 (Best Recall)

**Location:** `models/classification/resnet50.pth` (90 MB - in repo)

**Validation Script:**
```bash
python src/classification/validate_resnet50.py \
    --model_path models/classification/resnet50.pth \
    --data_root data
```

**Performance:** Precision: 0.953 | Recall: 1.000 (catches ALL scratches) | F1: 0.976

---

### 3. UNeXt Multi-task (Classification + Segmentation)

**Location:** `models/unext_multitask/scratch_UNext_multitask_subset_2.pth` (5.6 MB - in repo)

**Validation Script:**
```bash
cd src/unext_multitask
python val.py --name scratch_UNext_multitask_subset_2
```

**Or run inference:**
```bash
python src/unext_multitask/inference.py \
    --model_path models/unext_multitask/scratch_UNext_multitask_subset_2.pth \
    --input_dir data/val \
    --output_dir outputs/unext_multitask
```

**Outputs:**
- Classification predictions
- Segmentation masks
- Overlaid visualizations

**Performance:** Precision: 1.000 | Recall: 0.971 | F1: 0.985

---

## 📊 Results Summary

| Model | Precision | Recall | F1 | Size | Location |
|-------|-----------|--------|-------|------|----------|
| **EfficientNetB0** | **1.000** | **0.990** | **0.995** | 16 MB | ✅ In repo |
| ResNet50 | 0.953 | **1.000** | 0.976 | 90 MB | ✅ In repo |
| UNeXt Multi-task | **1.000** | 0.971 | 0.985 | 5.6 MB | ✅ In repo |
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
