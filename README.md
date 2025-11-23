# Scratch Detection on Text Images

**Mowito Assessment Project** - Text scratch detection using deep learning with classification and segmentation approaches.

## 🎯 Problem Statement

Detect and segment scratches on text images using binary classification (good/bad) with optional mask generation.

## 🏆 Results Summary

| Model | Precision | Recall | F1 Score | Size | Status |
|-------|-----------|--------|----------|------|--------|
| **EfficientNetB0** | 1.000 | 0.990 | **0.995** | 16 MB | ⭐ Best Overall |
| **ResNet50** | 0.953 | 1.000 | 0.976 | 90 MB | 🎯 Best Recall |
| **UNeXt Multi-task** | 1.000 | 0.971 | 0.985 | 5.6 MB | ⚖️ Joint Cls+Seg |

*All metrics on validation set (10% of data)*

## 📁 Repository Structure

```
mowito-assesment/
├── models/                  # Trained model weights (112 MB)
│   ├── classification/      # Transfer learning models
│   └── unext_multitask/    # Multi-task model
│
├── src/                    # Source code
│   ├── classification/     # ResNet50 & EfficientNetB0
│   ├── unext_gradcam/     # UNeXt + GradCAM
│   ├── unext_multitask/   # UNeXt joint training
│   ├── synthetic_data/     # 3 augmentation methods
│   └── utils/             # Data preparation
│
├── results/               # Training logs & configs
├── docs/                  # Documentation
└── scripts/              # Helper scripts
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone repository
git clone https://github.com/yourusername/mowito-assesment.git
cd mowito-assesment

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Download Dataset

Place your dataset in the following structure:
```
data/
├── train/
│   ├── good/
│   └── bad/
├── val/
│   ├── good/
│   └── bad/
└── masks/
    ├── train/
    └── val/
```

See `docs/DATASET.md` for more details.

### 3. Run Inference

**EfficientNetB0 (Recommended)**
```bash
python src/classification/validate_efficientnet.py \
    --model_path models/classification/efficientnet_b0.pth \
    --data_root data/val
```

**UNeXt Multi-task (with segmentation)**
```bash
python src/unext_multitask/inference.py \
    --model_path models/unext_multitask/scratch_UNext_multitask_subset_2.pth \
    --input_dir data/val
```

## 🎓 Approaches Implemented

### 1. Transfer Learning Classification
- **ResNet50**: Perfect recall (1.000), catches all bad images
- **EfficientNetB0**: Best F1 (0.995), perfect precision

### 2. UNeXt + GradCAM
- Classification with post-hoc segmentation
- User-controllable scratch size threshold

### 3. UNeXt Multi-task
- Joint classification + segmentation training
- Direct mask supervision
- Perfect precision with good recall

### 4. Synthetic Data Generation
- **OpenCV method**: Text-aware scratch placement
- **Vanilla GAN**: Learns scratch patterns
- **Pix2Pix**: Conditional scratch generation

## 📊 Training

See individual approach READMEs:
- [Classification Training](src/classification/README.md)
- [UNeXt GradCAM](src/unext_gradcam/README.md)
- [UNeXt Multi-task](src/unext_multitask/README.md)

## 📈 Evaluation Metrics

- **Recall (Bad)**: Minimize false negatives (crucial requirement)
- **Precision (Bad)**: Minimize false positives
- **F1 Score**: Harmonic mean of precision and recall
- **IoU (Segmentation)**: Intersection over Union for masks

## 🎁 Bonus Features

✅ Segmentation masks (2 methods: GradCAM & direct)  
✅ User-controllable scratch size threshold  
✅ Advanced synthetic data generation (3 methods)  
✅ Extensible to other surfaces (architecture-ready)

## 📄 License

This project was created as part of the Mowito assessment.

## 🙏 Acknowledgments

- UNeXt architecture: [Paper](https://arxiv.org/abs/2203.04967)
- Transfer learning: ImageNet pretrained models
- Dataset: Mowito assessment dataset

## 📧 Contact

For questions about this implementation, please contact the repository owner.
