# Trained Model Weights

This directory contains the top 3 performing models from the scratch detection project.

## Models Included

### 1. ResNet50 (90 MB)
- **Path**: `classification/resnet50.pth`
- **Architecture**: ResNet50 with pretrained ImageNet weights
- **Performance**: 
  - Precision: 0.953
  - Recall: **1.000** (perfect - catches all scratches)
  - F1: 0.976
- **Best for**: Critical applications where missing scratches is unacceptable
- **Training**: 50 epochs, Adam optimizer (lr=1e-4)

### 2. EfficientNetB0 (16 MB) ⭐ RECOMMENDED
- **Path**: `classification/efficientnet_b0.pth`
- **Architecture**: EfficientNetB0 with pretrained ImageNet weights
- **Performance**: 
  - Precision: **1.000** (perfect - no false alarms)
  - Recall: 0.990
  - F1: **0.995** (best overall)
- **Best for**: Production deployment (smallest, fastest, most balanced)
- **Training**: 50 epochs, Adam optimizer (lr=1e-4)

### 3. UNeXt Multi-task (5.6 MB)
- **Path**: `unext_multitask/scratch_UNext_multitask_subset_2.pth`
- **Architecture**: UNeXt (MLP-based) with classification + segmentation heads
- **Performance**: 
  - Precision: **1.000**
  - Recall: 0.971
  - F1: 0.985
  - **Bonus**: Provides segmentation masks
- **Best for**: Applications requiring interpretable results with masks
- **Training**: 2000 epochs, joint classification + segmentation loss

## Loading Models

### Classification Models (ResNet50, EfficientNetB0)

```python
import torch
from torchvision import models

# ResNet50
model = models.resnet50(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load('models/classification/resnet50.pth'))

# EfficientNetB0
from torchvision.models import efficientnet_b0
model = efficientnet_b0(pretrained=False)
model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 2)
model.load_state_dict(torch.load('models/classification/efficientnet_b0.pth'))
```

### UNeXt Multi-task

```python
import sys
sys.path.append('src/unext_multitask')
from archs import UNext

model = UNext(num_classes=1, input_channels=3, deep_supervision=False)
model.load_state_dict(torch.load('models/unext_multitask/scratch_UNext_multitask_subset_2.pth'))
```

## Model Selection Guide

| Use Case | Recommended Model | Why |
|----------|------------------|-----|
| Production API | EfficientNetB0 | Best balance, smallest size, fastest |
| Quality Control | ResNet50 | Never misses a scratch (perfect recall) |
| Analysis/Research | UNeXt Multi-task | Provides visual explanations via masks |
| Mobile/Edge | EfficientNetB0 | Only 16MB, efficient inference |

## Download Individual Models

If repository clone is too large, download models individually:

```bash
# Download only the model you need
wget https://github.com/yourusername/mowito-assesment/raw/main/models/classification/efficientnet_b0.pth
```

## Total Size: 112 MB
