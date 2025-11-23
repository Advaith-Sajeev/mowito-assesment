# scratch_gan/generate_synthetic_from_gan.py
import os
import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import torch

from gan_models import ScratchGenerator

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
random.seed(42)

def list_images(folder: Path):
    return sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS])

def find_text_region(gray_np):
    """
    Very simple heuristic:
    - binarize
    - find horizontal bands with many white/black pixels
    Returns y0,y1,x0,x1 for a candidate text stripe or None.
    """
    H, W = gray_np.shape
    # adaptive threshold
    thr = cv2.adaptiveThreshold(gray_np, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                cv2.THRESH_BINARY_INV, 15, 10)
    # sum per row
    row_sums = thr.sum(axis=1) / 255
    # pick rows where text density > some threshold
    mask_rows = row_sums > (0.15 * W)
    ys = np.where(mask_rows)[0]
    if len(ys) == 0:
        return None

    y_min, y_max = ys.min(), ys.max()
    # small padding
    y_min = max(0, y_min - 5)
    y_max = min(H - 1, y_max + 5)

    # restrict horizontally to center if you want
    x_min, x_max = int(0.05 * W), int(0.95 * W)
    return y_min, y_max, x_min, x_max

def paste_scratch_on_text(good_img_np, intensity_np, mask_np):
    """
    good_img_np: H,W,3 uint8
    intensity_np: ph,pw float32 in [-1,1] (from GAN)
    mask_np: ph,pw float32 in [0,1]
    """
    H, W, _ = good_img_np.shape
    gray = cv2.cvtColor(good_img_np, cv2.COLOR_BGR2GRAY)
    region = find_text_region(gray)
    if region is None:
        # fallback: whole image
        y_min, y_max, x_min, x_max = 0, H - 1, 0, W - 1
    else:
        y_min, y_max, x_min, x_max = region

    # choose random top-left inside this region
    ph, pw = mask_np.shape
    max_y0 = max(y_min, y_max - ph)
    max_x0 = max(x_min, x_max - pw)
    if max_y0 < y_min or max_x0 < x_min:
        # scratch bigger than band; fallback center
        y0 = max(0, (H - ph) // 2)
        x0 = max(0, (W - pw) // 2)
    else:
        y0 = random.randint(y_min, max_y0)
        x0 = random.randint(x_min, max_x0)

    aug = good_img_np.copy()
    mask_bool = mask_np > 0.5

    # convert intensity [-1,1] -> [0,255] but push scratches to white
    scratch_val = ((intensity_np + 1.0) * 0.5) * 255.0
    scratch_val = np.clip(scratch_val, 0, 255).astype(np.uint8)
    # force high brightness where mask=1
    scratch_val[mask_bool] = 255

    region = aug[y0:y0+ph, x0:x0+pw, :]

    # apply scratch as white over existing text
    for c in range(3):
        channel = region[..., c]
        channel[mask_bool] = scratch_val[mask_bool]
        region[..., c] = channel

    aug[y0:y0+ph, x0:x0+pw, :] = region

    # build full mask
    full_mask = np.zeros((H, W), dtype=np.uint8)
    full_mask[y0:y0+ph, x0:x0+pw][mask_bool] = 255

    return aug, full_mask

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    good_dir = Path("anomaly_detection_test_data/good")
    weights_path = Path("scratch_gan/weights/G_latest.pth")

    out_root = Path("synthetic_bad")
    out_imgs = out_root / "images"
    out_masks = out_root / "masks"
    out_panels = out_root / "panels"
    for d in [out_imgs, out_masks, out_panels]:
        d.mkdir(parents=True, exist_ok=True)

    good_imgs = list_images(good_dir)
    print(f"Found {len(good_imgs)} GOOD images.")

    # load generator
    z_dim = 128
    G = ScratchGenerator(z_dim=z_dim).to(device)
    G.load_state_dict(torch.load(weights_path, map_location=device))
    G.eval()

    # how many synthetic you want:
    total_to_create = len(good_imgs)   # or whatever

    created = 0

    for img_path in good_imgs:
        if created >= total_to_create:
            break

        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue
        H, W, _ = img_bgr.shape

        # sample scratch size by resizing generator output
        z = torch.randn(1, z_dim, device=device)
        with torch.no_grad():
            intensity, mask_logits = G(z)
            intensity = intensity[0, 0].cpu().numpy()      # (h,w)
            mask = torch.sigmoid(mask_logits)[0, 0].cpu().numpy()

        # optionally enforce minimum scratch size
        ph, pw = intensity.shape
        scale = random.uniform(0.5, 1.2)
        new_h = max(8, int(ph * scale))
        new_w = max(8, int(pw * scale))

        intensity_rs = cv2.resize(intensity, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        mask_rs = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

        aug_img_np, aug_mask_np = paste_scratch_on_text(img_bgr, intensity_rs, mask_rs)

        # panel: [original | synthetic | synthetic+mask overlay]
        orig_vis = img_bgr.copy()
        synth_vis = aug_img_np.copy()
        overlay = synth_vis.copy()
        # draw mask in red
        red_layer = np.zeros_like(overlay)
        red_layer[..., 2] = 255
        m_float = (aug_mask_np > 0).astype(np.float32)[..., None]
        overlay = (overlay * (1 - 0.4 * m_float) + red_layer * (0.4 * m_float)).astype(np.uint8)

        panel = np.concatenate([orig_vis, synth_vis, overlay], axis=1)

        stem = img_path.stem
        out_img_path = out_imgs / f"{stem}_synth.png"
        out_mask_path = out_masks / f"{stem}_synth.png"
        out_panel_path = out_panels / f"{stem}_synth_panel.png"

        cv2.imwrite(str(out_img_path), aug_img_np)
        cv2.imwrite(str(out_mask_path), aug_mask_np)
        cv2.imwrite(str(out_panel_path), panel)

        created += 1
        if created % 50 == 0:
            print(f"Created {created} synthetic samples...")

    print(f"Done. Created {created} synthetic bad images.")

if __name__ == "__main__":
    main()
