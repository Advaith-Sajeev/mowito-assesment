# Pix2Pix GAN Models

## Model Information

These are the best performing Pix2Pix GAN models for synthetic scratch generation.

### Files

**Note:** The generator model (160 MB) exceeds GitHub's 100 MB file limit and cannot be uploaded directly.

#### Discriminator (Included)
- **File:** `discriminator_best.pth`
- **Size:** 11 MB
- **Purpose:** Discriminates between real and synthetic scratch patterns

#### Generator (Too Large for GitHub)
- **File:** `generator_best.pth` 
- **Size:** 160 MB
- **Purpose:** Generates synthetic scratch patterns on text images
- **Status:** ⚠️ Not included in repository (exceeds GitHub limit)

---

## Alternative: Download Options

### Option 1: Use Git LFS (Recommended)

If you have Git LFS installed:
```bash
# Install Git LFS
brew install git-lfs  # macOS
# or: sudo apt-get install git-lfs  # Linux

# Initialize Git LFS
git lfs install

# Track large files
git lfs track "models/SyntheticDataP2P/generator_best.pth"

# Add and commit
git add .gitattributes
git add models/SyntheticDataP2P/generator_best.pth
git commit -m "Add Pix2Pix generator via Git LFS"
git push
```

### Option 2: External Hosting

Upload the generator to:
- **Google Drive** (recommended for assessment submission)
- **Dropbox**
- **OneDrive**
- **Hugging Face Model Hub**

Then share the download link in the README or issues.

---

## Using the Models

### Loading Discriminator
```python
import torch
from src.synthetic_data.pix2pix_gan.pix2pix_models import PatchDiscriminator

# Load discriminator
discriminator = PatchDiscriminator(in_channels=3)
discriminator.load_state_dict(
    torch.load('models/SyntheticDataP2P/discriminator_best.pth')
)
discriminator.eval()
```

### Loading Generator (if you have the file)
```python
from src.synthetic_data.pix2pix_gan.pix2pix_models import Pix2PixGenerator

# Load generator
generator = Pix2PixGenerator(in_channels=2, out_channels=1)
generator.load_state_dict(
    torch.load('path/to/generator_best.pth')
)
generator.eval()
```

---

## Training Details

- **Architecture:** Pix2Pix (conditional GAN)
- **Training Data:** Bad image patches with scratch masks
- **Input:** Clean image + mask (2 channels)
- **Output:** Scratch pattern (1 channel)
- **Loss:** Adversarial + L1 reconstruction (λ=100)
- **Optimizer:** Adam (lr=2e-4, β1=0.5)

---

## Model Performance

These models were trained to generate realistic scratch patterns for data augmentation. See training logs in `results/` directory.

---

## Contact

For access to the full generator model (160 MB), please:
1. Open an issue in this repository
2. Contact the repository owner
3. Request via email with justification
