import os
import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

# -----------------------------
# CONFIG
# -----------------------------
SRC_ROOT = Path("anomaly_detection_test_data")  # contains: good/, bad/, masks/
GOOD_DIR = SRC_ROOT / "good"
BAD_DIR = SRC_ROOT / "bad"
MASK_DIR = SRC_ROOT / "masks"

# Where to put synthetic outputs
SYNTH_ROOT = Path("synthetic_bad")
SYNTH_IMG_DIR = SYNTH_ROOT / "images"   # synthetic bad images
SYNTH_MASK_DIR = SYNTH_ROOT / "masks"   # synthetic bad masks
SYNTH_PANEL_DIR = SYNTH_ROOT / "panels" # visualization panels

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
random.seed(42)

# scratch size requirements (to avoid tiny specks)
MIN_PATCH_PIXELS = 50         # minimum non-zero pixels in extracted patch mask
MIN_IMG_FRAC = 0.002          # min fraction of image area for final synthetic mask (~0.2%)

# how many random candidate positions to try when placing mask on text
PLACEMENT_TRIALS = 50


def list_images(folder: Path):
    """List image files in a folder (non-recursive)."""
    return sorted(
        [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS]
    )


def load_image_rgb(path: Path):
    return Image.open(path).convert("RGB")


def load_mask_gray(path: Path):
    return Image.open(path).convert("L")


def get_scratch_mask(mask_img: Image.Image):
    """
    From a bad mask image, extract the tight scratch patch
    as a numpy array (HxW).
    Returns patch_mask or None if empty/too small.
    """
    mask_np = np.array(mask_img)  # H,W

    ys, xs = np.where(mask_np > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None

    y_min, y_max = ys.min(), ys.max() + 1
    x_min, x_max = xs.min(), xs.max() + 1

    patch_mask = mask_np[y_min:y_max, x_min:x_max]

    # filter tiny patches
    if (patch_mask > 0).sum() < MIN_PATCH_PIXELS:
        return None

    return patch_mask


def random_resize_mask(patch_mask, target_h, target_w):
    """
    Randomly rescale the patch mask so it fits nicely inside the target image.
    Returns resized patch_mask_np (uint8).
    """
    ph, pw = patch_mask.shape[:2]

    # base scale so the patch <= 1/2 of the target dim
    max_scale_h = (target_h * 0.5) / ph
    max_scale_w = (target_w * 0.5) / pw
    max_scale = min(max_scale_h, max_scale_w, 1.5)
    max_scale = max(max_scale, 0.3)   # avoid degenerate

    # random scale in [0.5, max_scale]
    scale = random.uniform(0.5, max_scale)

    new_h = max(4, int(ph * scale))
    new_w = max(4, int(pw * scale))

    p_mask_pil = Image.fromarray(patch_mask)
    p_mask_resized = np.array(p_mask_pil.resize((new_w, new_h), Image.NEAREST))

    return p_mask_resized


def compute_text_score_map(good_img: Image.Image):
    """
    Build a 'text-likelihood' map:
      - edges via Canny
      - emphasize darker pixels (likely text)
    Returns text_score: float32 array in [0,1].
    """
    g_np = np.array(good_img.convert("RGB"))  # H,W,3
    gray = cv2.cvtColor(g_np, cv2.COLOR_RGB2GRAY)

    # edges
    edges = cv2.Canny(gray, 50, 150)  # 0 or 255
    edges = edges.astype(np.float32) / 255.0

    # darker-than-background mask (simple threshold)
    dark = (gray < 200).astype(np.float32)

    text_score = edges * dark

    # optional: blur a bit to smooth
    text_score = cv2.GaussianBlur(text_score, (3, 3), 0)

    # normalize to [0,1] (avoid divide by zero)
    max_val = text_score.max()
    if max_val > 0:
        text_score = text_score / max_val

    return text_score.astype(np.float32)


def choose_position_for_patch(patch_mask, text_score):
    """
    Choose a top-left position (y0, x0) to place patch_mask on the image,
    such that it overlaps text_score as much as possible.
    Randomly samples several candidates and picks the best score.
    """
    H, W = text_score.shape[:2]
    ph, pw = patch_mask.shape[:2]

    # If patch bigger than image (shouldn't happen if we resized correctly),
    # fallback to center.
    if ph >= H or pw >= W:
        return max(0, (H - ph) // 2), max(0, (W - pw) // 2)

    mask_bool = patch_mask > 0

    best_score = -1.0
    best_y0, best_x0 = 0, 0

    for _ in range(PLACEMENT_TRIALS):
        y0 = random.randint(0, H - ph)
        x0 = random.randint(0, W - pw)

        region = text_score[y0:y0 + ph, x0:x0 + pw]
        score = region[mask_bool].sum()

        if score > best_score:
            best_score = score
            best_y0, best_x0 = y0, x0

    return best_y0, best_x0


def transplant_scratch_white(good_img: Image.Image, patch_mask: np.ndarray):
    """
    Paste a 'scratch' onto a copy of the good image using ONLY the mask shape.

    - Compute text_score_map.
    - Place the mask where text_score is highest.
    - Set those pixels to pure white [255,255,255].
    - Return synthetic image + synthetic mask.

    Returns:
        aug_img_np: HxWx3 uint8 synthetic bad image
        aug_mask_np: HxW uint8 synthetic mask (255 in scratch region)
    """
    g_np = np.array(good_img.convert("RGB")).copy()   # H,W,3
    H, W = g_np.shape[:2]

    # If patch is bigger than image, shrink
    ph, pw = patch_mask.shape[:2]
    if ph > H or pw > W:
        scale_h = H / ph
        scale_w = W / pw
        scale = min(scale_h, scale_w) * 0.9
        new_h = max(4, int(round(ph * scale)))
        new_w = max(4, int(round(pw * scale)))
        patch_mask = np.array(
            Image.fromarray(patch_mask).resize((new_w, new_h), Image.NEAREST)
        )
        ph, pw = patch_mask.shape[:2]

    # compute text score map
    text_score = compute_text_score_map(good_img)

    # choose placement based on text_score
    if ph >= H or pw >= W:
        y0 = max(0, (H - ph) // 2)
        x0 = max(0, (W - pw) // 2)
    else:
        y0, x0 = choose_position_for_patch(patch_mask, text_score)

    y1 = min(H, y0 + ph)
    x1 = min(W, x0 + pw)

    # crop mask to fit region exactly
    ph2 = y1 - y0
    pw2 = x1 - x0
    patch_mask = patch_mask[:ph2, :pw2]
    mask_bool = patch_mask > 0

    region = g_np[y0:y1, x0:x1, :]  # Hreg, Wreg, 3

    # make scratches pure white
    for c in range(3):
        ch = region[..., c]
        ch[mask_bool] = 255
        region[..., c] = ch

    g_np[y0:y1, x0:x1, :] = region

    # build output mask for whole image
    aug_mask = np.zeros((H, W), dtype=np.uint8)
    aug_mask[y0:y1, x0:x1][mask_bool] = 255

    return g_np, aug_mask


def make_panel(orig_img_np, synth_img_np, mask_np):
    """
    Build a 3-panel visualization:
      Left  : original good image
      Middle: synthetic bad image
      Right : synthetic bad image with GT mask overlaid (red)
    """
    H, W, _ = synth_img_np.shape

    # Ensure orig has same size as synth
    orig_resized = np.array(
        Image.fromarray(orig_img_np).resize((W, H), Image.BILINEAR)
    )

    # build overlay on synthetic
    overlay = synth_img_np.copy().astype(np.float32)
    mask_bool = mask_np > 0

    # color for mask overlay (red in RGB)
    color = np.array([255, 0, 0], dtype=np.float32)
    alpha = 0.5

    for c in range(3):
        ch = overlay[..., c]
        ch[mask_bool] = (1 - alpha) * ch[mask_bool] + alpha * color[c]
        overlay[..., c] = ch

    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    panel = np.concatenate([orig_resized, synth_img_np, overlay], axis=1)
    return panel


def main():
    # ensure dirs
    GOOD_DIR.mkdir(parents=True, exist_ok=True)
    BAD_DIR.mkdir(parents=True, exist_ok=True)
    MASK_DIR.mkdir(parents=True, exist_ok=True)

    SYNTH_IMG_DIR.mkdir(parents=True, exist_ok=True)
    SYNTH_MASK_DIR.mkdir(parents=True, exist_ok=True)
    SYNTH_PANEL_DIR.mkdir(parents=True, exist_ok=True)

    good_imgs = list_images(GOOD_DIR)
    bad_imgs = list_images(BAD_DIR)

    if not good_imgs:
        print("No GOOD images found. Exiting.")
        return
    if not bad_imgs:
        print("No BAD images found. Need some real bad images to get scratches from.")
        return

    n_good = len(good_imgs)
    n_bad_real = len(bad_imgs)

    print(f"#good (real) = {n_good}, #bad (real) = {n_bad_real}")

    if n_bad_real >= n_good:
        print("Already have >= as many bad images as good. Nothing to do.")
        return

    n_needed = n_good - n_bad_real
    print(f"Will create {n_needed} synthetic bad images (in '{SYNTH_ROOT}/').")

    # build map from bad filename -> mask path
    mask_paths = {p.name: p for p in list_images(MASK_DIR)}

    created = 0
    attempt = 0
    max_attempts = n_needed * 20  # safety

    while created < n_needed and attempt < max_attempts:
        attempt += 1

        good_path = random.choice(good_imgs)
        bad_path = random.choice(bad_imgs)

        bad_name = bad_path.name
        mask_path = mask_paths.get(bad_name, None)
        if mask_path is None:
            # skip if no mask
            continue

        good_im = load_image_rgb(good_path)
        mask_im = load_mask_gray(mask_path)

        # Extract scratch mask patch from real bad
        patch_mask = get_scratch_mask(mask_im)
        if patch_mask is None:
            continue

        # Resize patch mask randomly to fit good image
        H = good_im.size[1]
        W = good_im.size[0]
        patch_mask_resized = random_resize_mask(patch_mask, H, W)

        # Transplant onto good image using pure-white scratches over text
        aug_img_np, aug_mask_np = transplant_scratch_white(good_im, patch_mask_resized)

        # Check global scratch size (avoid tiny)
        frac = (aug_mask_np > 0).sum() / float(aug_mask_np.size)
        if frac < MIN_IMG_FRAC:
            # too small, skip
            continue

        # save synthetic image & mask
        good_stem = good_path.stem
        new_name = f"{good_stem}_synth_{created:04d}.png"

        out_img_path = SYNTH_IMG_DIR / new_name
        out_mask_path = SYNTH_MASK_DIR / new_name

        Image.fromarray(aug_img_np).save(out_img_path)
        Image.fromarray(aug_mask_np).save(out_mask_path)

        # save panel: original (good), synthetic, synthetic+mask overlay
        orig_np = np.array(good_im.convert("RGB"))
        panel_np = make_panel(orig_np, aug_img_np, aug_mask_np)
        out_panel_path = SYNTH_PANEL_DIR / new_name
        Image.fromarray(panel_np).save(out_panel_path)

        created += 1

        if created % 50 == 0 or created == n_needed:
            print(f"Created {created}/{n_needed} synthetic bad images...")

    print(f"Done. Created {created} synthetic bad images.")
    print(f"Synthetic images: {SYNTH_IMG_DIR}")
    print(f"Synthetic masks : {SYNTH_MASK_DIR}")
    print(f"Synthetic panels: {SYNTH_PANEL_DIR}")


if __name__ == "__main__":
    main()
