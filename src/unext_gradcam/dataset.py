import os
from typing import List, Tuple, Optional

import cv2
import numpy as np
import torch
import torch.utils.data


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def _is_image_file(fname: str) -> bool:
    return os.path.splitext(fname.lower())[1] in IMG_EXTS


def _list_images_in_folder(folder: str) -> List[str]:
    if not os.path.isdir(folder):
        return []
    return sorted(
        [
            f
            for f in os.listdir(folder)
            if os.path.isfile(os.path.join(folder, f)) and _is_image_file(f)
        ]
    )


class Dataset(torch.utils.data.Dataset):
    """
    Dataset that returns (image_tensor, mask_tensor, label, meta)

    Expected directory layout:

      data/
        train/
          good/
            img1.jpg
            img2.jpg
            ...
          bad/
            imgA.jpg
            imgB.jpg
            ...
        val/
          good/
          bad/
        masks/
          train/
            good/
              img1.png
              img2.png
            bad/
              imgA.png
              imgB.png
          val/
            good/
            bad/

    If a mask file does not exist, a zero-mask is created on the fly.
    """

    def __init__(
        self,
        img_root: str,
        mask_root: str,
        phase: str = "train",
        class_names: Tuple[str, str] = ("good", "bad"),
        transform: Optional[object] = None,
        mask_ext: str = ".png",
        image_size: Tuple[int, int] = (224, 224),
    ):
        """
        Args:
            img_root: root that contains 'train'/'val' image folders.
            mask_root: root that contains 'masks/train'/'masks/val' folders.
            phase: 'train' or 'val'.
            class_names: ('good_folder_name', 'bad_folder_name').
            transform: optional Albumentations-like transform (expects numpy HWC image & mask).
            mask_ext: mask file extension (default '.png').
            image_size: (H, W) to resize to when transform is None.
        """
        super().__init__()
        self.img_root = img_root
        self.mask_root = mask_root
        self.phase = phase
        self.transform = transform
        self.mask_ext = mask_ext
        self.class_names = class_names
        self.image_size = image_size  # (H, W)

        self.samples = []  # each: dict(img_path, mask_path or None, label, img_id)

        for label_idx, cname in enumerate(self.class_names):
            img_dir = os.path.join(self.img_root, phase, cname)
            mask_dir = os.path.join(self.mask_root, phase, cname)
            filenames = _list_images_in_folder(img_dir)
            for fname in filenames:
                img_path = os.path.join(img_dir, fname)
                base = os.path.splitext(fname)[0]
                mask_fname = base + self.mask_ext
                mask_path = os.path.join(mask_dir, mask_fname)
                if not os.path.exists(mask_path):
                    mask_path = None
                self.samples.append(
                    {
                        "img_id": base,
                        "img_path": img_path,
                        "mask_path": mask_path,
                        "label": int(label_idx),  # 0 = good, 1 = bad
                    }
                )

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No images found under {self.img_root}/{self.phase}. "
                f"Check that data/{phase}/good and data/{phase}/bad exist and contain images."
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        img_path = s["img_path"]
        mask_path = s["mask_path"]
        label = s["label"]
        img_id = s["img_id"]

        # ---- load image (BGR) and convert to RGB ----
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            raise RuntimeError(f"Failed to read image: {img_path}")
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # ---- load or create mask (single-channel) ----
        if mask_path is not None and os.path.exists(mask_path):
            mask_gray = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask_gray is None:
                mask_gray = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
        else:
            mask_gray = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)

        # H x W x 1 for Albumentations
        mask = mask_gray[..., None]

        # ---- apply transform (if provided) ----
        if self.transform is not None:
            # transform should handle resizing & normalization
            augmented = self.transform(image=img, mask=mask)
            img = augmented["image"]
            mask = augmented["mask"]
        else:
            # No transform: we manually resize to fixed image_size (H, W)
            h, w = self.image_size
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask_gray, (w, h), interpolation=cv2.INTER_NEAREST)
            mask = mask[..., None]  # back to H x W x 1

        # ---- convert to torch tensors ----
        # Now img & mask should be numpy arrays in HWC format
        if isinstance(img, torch.Tensor):
            # If at some point you plug a ToTensor transform, you can handle it here.
            img_t = img.float()
        else:
            img = img.astype(np.float32)
            img = img.transpose(2, 0, 1)  # HWC -> CHW
            img_t = torch.from_numpy(img)

        if isinstance(mask, torch.Tensor):
            mask_t = mask.float()
            if mask_t.ndim == 2:
                mask_t = mask_t.unsqueeze(0)
            elif mask_t.ndim == 3 and mask_t.shape[0] != 1 and mask_t.shape[-1] == 1:
                mask_t = mask_t.permute(2, 0, 1)
            if mask_t.max() > 1.5:
                mask_t = mask_t / 255.0
        else:
            mask = mask.astype(np.float32)
            if mask.max() > 1.5:
                mask = mask / 255.0
            mask = mask.transpose(2, 0, 1)  # (1, H, W)
            mask_t = torch.from_numpy(mask)

        label_t = torch.tensor(label, dtype=torch.long)
        meta = {"img_id": img_id, "img_path": img_path}

        return img_t, mask_t, label_t, meta


def make_dataloader(
    img_root: str,
    mask_root: str,
    phase: str = "train",
    batch_size: int = 8,
    shuffle: bool = True,
    transform=None,
    num_workers: int = 4,
    image_size: Tuple[int, int] = (224, 224),
):
    ds = Dataset(
        img_root=img_root,
        mask_root=mask_root,
        phase=phase,
        transform=transform,
        image_size=image_size,
    )
    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )
    return loader


if __name__ == "__main__":
    # quick sanity check
    data_root = "data"  # your root
    masks_root = os.path.join(data_root, "masks")

    train_loader = make_dataloader(
        img_root=data_root,
        mask_root=masks_root,
        phase="train",
        batch_size=4,
        shuffle=True,
        transform=None,   # no albumentations here, we rely on internal resize
        num_workers=0,    # easier for debugging
        image_size=(224, 224),
    )

    for imgs, masks, labels, meta in train_loader:
        print("Imgs:", imgs.shape)    # (B, 3, 224, 224)
        print("Masks:", masks.shape)  # (B, 1, 224, 224)
        print("Labels:", labels.shape)
        print("Meta[0]:", meta["img_id"][0], meta["img_path"][0])
        break
