# scratch_gan/scratch_patch_dataset.py

from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def list_images(folder: Path):
    return sorted(
        [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS]
    )


class ScratchPatchDataset(Dataset):
    """
    Dataset of scratch mask patches.

    Expects:
      - bad_dir: folder containing BAD images (we use only names)
      - mask_dir: folder containing corresponding masks (same filenames)
    Only the masks are used; images are just for filename matching.
    """

    def __init__(self, bad_dir, mask_dir, patch_size=64):
        self.bad_dir = Path(bad_dir)
        self.mask_dir = Path(mask_dir)
        self.patch_size = patch_size

        self.bad_imgs = list_images(self.bad_dir)
        # map filename -> mask path
        self.mask_paths = {p.name: p for p in list_images(self.mask_dir)}

        self.items = []
        for p in self.bad_imgs:
            if p.name in self.mask_paths:
                self.items.append(p)

        print(f"ScratchPatchDataset: {len(self.items)} images with masks found.")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        img_path = self.items[idx]
        mask_path = self.mask_paths[img_path.name]

        # We only care about the mask shape
        mask = Image.open(mask_path).convert("L")
        mask_np = np.array(mask)  # H,W

        # tight bounding box around scratch
        ys, xs = np.where(mask_np > 0)
        if len(xs) == 0 or len(ys) == 0:
            # empty scratch; return empty patch
            patch_mask = np.zeros((self.patch_size, self.patch_size), np.float32)
        else:
            y_min, y_max = ys.min(), ys.max() + 1
            x_min, x_max = xs.min(), xs.max() + 1

            patch_mask = mask_np[y_min:y_max, x_min:x_max]

            patch_mask = Image.fromarray(patch_mask)
            patch_mask = patch_mask.resize(
                (self.patch_size, self.patch_size),
                Image.NEAREST
            )

            patch_mask = np.array(patch_mask).astype(np.float32)

        # binarize and normalize to [0,1]
        patch_mask = (patch_mask > 0).astype(np.float32)
        # final shape: (1, H, W)
        patch_mask = np.expand_dims(patch_mask, axis=0)

        return patch_mask
