# scratch_pix2pix_dataset.py

from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def list_images(folder: Path):
    return sorted(
        [p for p in folder.iterdir()
         if p.is_file() and p.suffix.lower() in IMG_EXTS]
    )


class ScratchPix2PixDataset(Dataset):
    """
    For each bad image + mask, we:
      - crop the tight bbox around the mask
      - resize to (patch_size, patch_size)
      - build:
          clean_approx  (inpainted under the mask)
          mask_patch    (binary, 0/1)
          bad_patch     (original intensity)

    Returns:
      cond:   (2, H, W)  = [ clean_approx_norm, mask ]
      target: (1, H, W)  = bad_patch_norm
    where intensities are in [-1, 1], mask in {0,1}.
    """

    def __init__(self, bad_dir, mask_dir, patch_size=128):
        self.bad_dir = Path(bad_dir)
        self.mask_dir = Path(mask_dir)
        self.patch_size = patch_size

        bad_imgs = list_images(self.bad_dir)
        mask_paths = {p.name: p for p in list_images(self.mask_dir)}

        self.items = []
        for img_path in bad_imgs:
            if img_path.name in mask_paths:
                self.items.append((img_path, mask_paths[img_path.name]))

        print(f"ScratchPix2PixDataset: {len(self.items)} bad images with masks.")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        img_path, mask_path = self.items[idx]

        # Read as grayscale
        img_gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)  # H,W
        mask_raw = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        if img_gray is None or mask_raw is None:
            # fall back to zeros if something is wrong
            H = W = self.patch_size
            cond = np.zeros((2, H, W), dtype=np.float32)
            target = np.zeros((1, H, W), dtype=np.float32)
            return torch.from_numpy(cond), torch.from_numpy(target)

        # Binarize mask
        mask_bin = (mask_raw > 0).astype(np.uint8)

        ys, xs = np.where(mask_bin > 0)
        if len(xs) == 0 or len(ys) == 0:
            # empty mask -> return 0 patch
            H = W = self.patch_size
            cond = np.zeros((2, H, W), dtype=np.float32)
            target = np.zeros((1, H, W), dtype=np.float32)
            return torch.from_numpy(cond), torch.from_numpy(target)

        # Tight bbox around scratch
        y_min, y_max = ys.min(), ys.max() + 1
        x_min, x_max = xs.min(), xs.max() + 1

        img_crop = img_gray[y_min:y_max, x_min:x_max]    # h,w
        mask_crop = mask_bin[y_min:y_max, x_min:x_max]   # h,w

        # Resize both to patch_size x patch_size
        img_crop = cv2.resize(img_crop, (self.patch_size, self.patch_size),
                              interpolation=cv2.INTER_LINEAR)
        mask_crop = cv2.resize(mask_crop, (self.patch_size, self.patch_size),
                               interpolation=cv2.INTER_NEAREST)

        # Build "clean" approximation via inpainting
        mask_for_inpaint = (mask_crop > 0).astype(np.uint8) * 255  # 0/255
        clean_approx = cv2.inpaint(
            img_crop,
            mask_for_inpaint,
            inpaintRadius=3,
            flags=cv2.INPAINT_TELEA
        )

        # Normalize intensities to [-1,1]
        bad_norm = (img_crop.astype(np.float32) / 255.0) * 2.0 - 1.0
        clean_norm = (clean_approx.astype(np.float32) / 255.0) * 2.0 - 1.0
        mask_float = (mask_crop > 0).astype(np.float32)

        cond = np.stack([clean_norm, mask_float], axis=0)  # (2,H,W)
        target = bad_norm[None, ...]                       # (1,H,W)

        cond_t = torch.from_numpy(cond)
        target_t = torch.from_numpy(target)
        return cond_t, target_t
