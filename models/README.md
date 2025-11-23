# Trained Model Weights

This directory contains the 3 best performing models for text scratch detection.

---

## 1. UNeXt Multi-task (5.6 MB) ⭐

**Path:** `unext_multitask/scratch_UNext_multitask_subset_2.pth`

**Architecture:** UNeXt (MLP-based) with joint classification + segmentation heads

**Performance:**
- Precision: **1.000** (perfect - no false alarms)
- Recall: 0.971
- F1 Score: 0.985
- **Bonus:** Provides segmentation masks showing scratch locations

**Best for:** Applications requiring interpretable results with visual scratch localization

---

## 2. EfficientNetB0 (16 MB) ⭐

**Path:** `classification/efficientnet_b0.pth`

**Architecture:** EfficientNetB0 with pretrained ImageNet weights

**Performance:**
- Precision: **1.000** (perfect - no false alarms)
- Recall: 0.990
- F1 Score: **0.995** (best overall)

**Best for:** Production deployment - lightweight, fast, and most balanced performance

---

## 3. ResNet50 (90 MB)

**Path:** `classification/resnet50.pth`

**Architecture:** ResNet50 with pretrained ImageNet weights

**Performance:**
- Precision: 0.953
- Recall: **1.000** (perfect - catches ALL scratches)
- F1 Score: 0.976

**Best for:** Critical applications where missing scratches is unacceptable

---

**Total Size:** 112 MB
