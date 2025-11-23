# generate_synthetic_pix2pix.py

import argparse
import random
from pathlib import Path

import cv2
import numpy as np
import torch

from pix2pix_models import Pix2PixGenerator

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
random.seed(42)


def list_images(folder: Path):
    return sorted(
        [p for p in folder.iterdir()
         if p.is_file() and p.suffix.lower() in IMG_EXTS]
    )


def find_text_region(gray_np):
    """
    Heuristic: find horizontal band where text density is high.
    Returns (y_min, y_max, x_min, x_max) or None.
    """
    H, W = gray_np.shape
    thr = cv2.adaptiveThreshold(
        gray_np,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        15,
        10,
    )
    row_sums = thr.sum(axis=1) / 255
    mask_rows = row_sums > (0.15 * W)
    ys = np.where(mask_rows)[0]
    if len(ys) == 0:
        return None

    y_min, y_max = ys.min(), ys.max()
    y_min = max(0, y_min - 5)
    y_max = min(H - 1, y_max + 5)
    x_min, x_max = int(0.05 * W), int(0.95 * W)
    return y_min, y_max, x_min, x_max


def load_generator(weights_path, in_channels, out_channels, device):
    G = Pix2PixGenerator(in_channels=in_channels, out_channels=out_channels).to(device)
    state = torch.load(weights_path, map_location=device)
    G.load_state_dict(state)
    G.eval()
    return G


def generate_scratch_patch(G, clean_patch_gray, mask_patch, device):
    """
    clean_patch_gray: HxW uint8
    mask_patch:       HxW float32 (0/1)
    Both MUST have the same spatial size. We enforce it here.
    """
    if clean_patch_gray.shape != mask_patch.shape:
        h, w = mask_patch.shape
        clean_patch_gray = cv2.resize(
            clean_patch_gray,
            (w, h),
            interpolation=cv2.INTER_LINEAR,
        )

    clean_norm = (clean_patch_gray.astype(np.float32) / 255.0) * 2.0 - 1.0
    cond_np = np.stack([clean_norm, mask_patch.astype(np.float32)], axis=0)  # (2,H,W)
    cond_t = torch.from_numpy(cond_np).unsqueeze(0).to(device)               # (1,2,H,W)

    with torch.no_grad():
        fake = G(cond_t)[0, 0].cpu().numpy()  # [-1,1]

    bad_gray = ((fake + 1.0) * 0.5 * 255.0)
    bad_gray = np.clip(bad_gray, 0, 255).astype(np.uint8)
    return bad_gray


def make_panel(orig_bgr, synth_bgr, full_mask):
    overlay = synth_bgr.copy()
    m = (full_mask > 0)[..., None].astype(np.float32)

    red_layer = np.zeros_like(overlay)
    red_layer[..., 2] = 255

    overlay = (overlay * (1 - 0.4 * m) + red_layer * (0.4 * m)).astype(np.uint8)
    panel = np.concatenate([orig_bgr, synth_bgr, overlay], axis=1)
    return panel


def collect_checkpoints(weights_dir: Path, step: int, include_best: bool):
    """
    Collect generator checkpoints:
      - generator_epoch_XXX.pth where epoch % step == 0
      - optionally generator_best.pth
    Returns list of (tag, path) pairs.
    """
    ckpts = []

    # epoch checkpoints
    for p in weights_dir.glob("generator_epoch_*.pth"):
        name = p.name  # e.g. generator_epoch_050.pth
        try:
            epoch_str = name.split("_")[-1].split(".")[0]  # "050"
            epoch = int(epoch_str)
        except ValueError:
            continue

        if epoch % step == 0:
            ckpts.append((f"epoch_{epoch:03d}", p))

    ckpts.sort(key=lambda x: x[0])

    # best checkpoint
    if include_best:
        best_path = weights_dir / "generator_best.pth"
        if best_path.exists():
            ckpts.append(("best", best_path))

    return ckpts


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic scratches using pix2pix G (multiple checkpoints)")
    parser.add_argument("--datafolder", required=True,
                        help="Root folder containing good/ and masks/ (e.g. anomaly_detection_test_data)")
    parser.add_argument("--weights_dir", required=True,
                        help="Folder containing generator_epoch_XXX.pth and optionally generator_best.pth")
    parser.add_argument("--out_root", default="synthetic_bad_pix2pix",
                        help="Output root folder")
    parser.add_argument("--patch_size", type=int, default=128,
                        help="Patch size used during training")
    parser.add_argument("--samples", type=int, default=100,
                        help="How many good images to process per checkpoint")
    parser.add_argument("--step", type=int, default=50,
                        help="Evaluate checkpoints every N epochs (e.g. 50)")
    parser.add_argument("--include_best", action="store_true",
                        help="Also generate using generator_best.pth if present")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_root = Path(args.datafolder)
    good_dir = data_root / "good"
    mask_dir = data_root / "masks"  # we reuse mask shapes from real bad images

    weights_dir = Path(args.weights_dir)
    out_root = Path(args.out_root)

    good_imgs = list_images(good_dir)
    mask_imgs = list_images(mask_dir)

    if len(good_imgs) == 0:
        print(f"No GOOD images found in {good_dir}.")
        return
    if len(mask_imgs) == 0:
        print(f"No mask images found in {mask_dir}.")
        return

    # collect checkpoints
    ckpts = collect_checkpoints(weights_dir, step=args.step, include_best=args.include_best)
    if not ckpts:
        print("No checkpoints found matching pattern and step.")
        return

    print(f"Found {len(good_imgs)} GOOD images, {len(mask_imgs)} masks.")
    print("Evaluating checkpoints:")
    for tag, path in ckpts:
        print(f"  {tag}: {path.name}")
    print(f"Outputs will be saved under: {out_root}")

    # pick subset of good images once (same set used for all checkpoints)
    chosen_goods = random.sample(good_imgs, min(args.samples, len(good_imgs)))
    print(f"Will process {len(chosen_goods)} good images per checkpoint.")

    for tag, ckpt_path in ckpts:
        print(f"\n=== Processing checkpoint {tag} ({ckpt_path.name}) ===")

        # subfolders per checkpoint
        epoch_root = out_root / f"generator_{tag}"
        out_imgs = epoch_root / "images"
        out_masks = epoch_root / "masks"
        out_panels = epoch_root / "panels"
        for d in [out_imgs, out_masks, out_panels]:
            d.mkdir(parents=True, exist_ok=True)

        # load generator for this checkpoint
        G = load_generator(ckpt_path, in_channels=2, out_channels=1, device=device)

        created = 0
        for img_path in chosen_goods:
            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None:
                continue
            H, W, _ = img_bgr.shape
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

            # find text band
            region = find_text_region(gray)
            if region is None:
                # fallback: whole image
                y_min, y_max, x_min, x_max = 0, H - 1, 0, W - 1
            else:
                y_min, y_max, x_min, x_max = region

            # choose random mask file, extract its bbox, resize to patch_size
            mask_path = random.choice(mask_imgs)
            mask_raw = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask_raw is None:
                continue

            mask_bin = (mask_raw > 0).astype(np.uint8)
            ys, xs = np.where(mask_bin > 0)
            if len(xs) == 0 or len(ys) == 0:
                continue

            my_min, my_max = ys.min(), ys.max() + 1
            mx_min, mx_max = xs.min(), xs.max() + 1

            mask_crop = mask_bin[my_min:my_max, mx_min:mx_max]
            mask_patch = cv2.resize(
                mask_crop,
                (args.patch_size, args.patch_size),
                interpolation=cv2.INTER_NEAREST,
            )
            mask_patch = (mask_patch > 0).astype(np.float32)

            # ensure patch fits inside text band (y_min..y_max, x_min..x_max)
            ph = pw = args.patch_size
            band_h = y_max - y_min + 1
            band_w = x_max - x_min + 1
            if band_h < ph or band_w < pw:
                # fallback to center of whole image
                y0 = max(0, (H - ph) // 2)
                x0 = max(0, (W - pw) // 2)
            else:
                max_y0 = y_max - ph + 1
                max_x0 = x_max - pw + 1
                y0 = random.randint(y_min, max_y0)
                x0 = random.randint(x_min, max_x0)

            # extract clean patch from GOOD image (may be smaller than ph/pw near borders)
            clean_patch_gray = gray[y0:y0 + ph, x0:x0 + pw]

            # generate bad patch with GAN (ensures shape matches mask_patch)
            bad_patch_gray = generate_scratch_patch(G, clean_patch_gray, mask_patch, device)

            # paste into image — but region can be smaller than ph/pw, so crop both
            synth_bgr = img_bgr.copy()
            region = synth_bgr[y0:y0 + ph, x0:x0 + pw, :]
            region_h, region_w = region.shape[:2]

            # crop mask and bad patch to match region size
            mask_sub = mask_patch[:region_h, :region_w]
            bad_sub = bad_patch_gray[:region_h, :region_w]
            mask_bool = mask_sub > 0.5

            for c in range(3):
                ch = region[..., c]
                ch[mask_bool] = bad_sub[mask_bool]
                region[..., c] = ch

            synth_bgr[y0:y0 + region_h, x0:x0 + region_w, :] = region

            # build full scratch mask for the whole image
            full_mask = np.zeros((H, W), dtype=np.uint8)
            window = full_mask[y0:y0 + region_h, x0:x0 + region_w]
            window[mask_bool] = 255
            full_mask[y0:y0 + region_h, x0:x0 + region_w] = window

            panel = make_panel(img_bgr, synth_bgr, full_mask)

            stem = img_path.stem
            cv2.imwrite(str(out_imgs / f"{stem}_synth.png"), synth_bgr)
            cv2.imwrite(str(out_masks / f"{stem}_synth.png"), full_mask)
            cv2.imwrite(str(out_panels / f"{stem}_synth_panel.png"), panel)

            created += 1

        print(f"Created {created} synthetic samples for checkpoint {tag}.")

    print(f"\nDone. All synthetic sets are saved under: {out_root}")


if __name__ == "__main__":
    main()
