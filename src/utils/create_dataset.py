import os
import shutil
import random
import argparse
from PIL import Image
import numpy as np

random.seed(42)

# Image extensions to process
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def list_images(folder):
    """List all image files in a folder (non-recursive)."""
    files = []
    for f in os.listdir(folder):
        path = os.path.join(folder, f)
        if not os.path.isfile(path):
            continue
        ext = os.path.splitext(f.lower())[1]
        if ext in IMG_EXTS:
            files.append(path)
    return files


def ensure_dirs(dst_root):
    """Create the train/val and mask directories under dst_root."""
    subdirs = [
        os.path.join(dst_root, "train", "good"),
        os.path.join(dst_root, "train", "bad"),
        os.path.join(dst_root, "val",   "good"),
        os.path.join(dst_root, "val",   "bad"),
        os.path.join(dst_root, "masks", "train", "good"),
        os.path.join(dst_root, "masks", "train", "bad"),
        os.path.join(dst_root, "masks", "val",   "good"),
        os.path.join(dst_root, "masks", "val",   "bad"),
    ]
    for d in subdirs:
        os.makedirs(d, exist_ok=True)


def create_zero_mask_like_image(img_path, dst_mask_path):
    """Create a single-channel zero mask matching the image size."""
    with Image.open(img_path) as im:
        w, h = im.size
    zero_mask = np.zeros((h, w), dtype=np.uint8)
    Image.fromarray(zero_mask).save(dst_mask_path)


def main():
    parser = argparse.ArgumentParser(
        description='Prepare balanced train/val dataset with masks for scratch detection.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--source',
        type=str,
        default='anomaly_detection_test_data',
        help='Source data directory containing good/, bad/, and masks/ folders'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data_synth',
        help='Output directory for processed dataset'
    )
    parser.add_argument(
        '--val-ratio',
        type=float,
        default=0.10,
        help='Validation split ratio (e.g., 0.10 for 10%%)'
    )
    
    args = parser.parse_args()
    
    src_root = args.source
    dst_root = args.output
    val_ratio = args.val_ratio
    bad_mask_root = os.path.join(src_root, "masks")
    
    # Validate inputs
    if not os.path.exists(src_root):
        print(f"Error: Source directory '{src_root}' does not exist!")
        return
    
    if not (0 < val_ratio < 1):
        print(f"Error: Validation ratio must be between 0 and 1, got {val_ratio}")
        return
    
    print(f"Configuration:")
    print(f"  Source:         {src_root}")
    print(f"  Output:         {dst_root}")
    print(f"  Val ratio:      {val_ratio:.2%}")
    print()
    
    ensure_dirs(dst_root)

    # -----------------------------
    # 1) Collect good & bad images
    # -----------------------------
    good_src_folder = os.path.join(src_root, "good")
    bad_src_folder  = os.path.join(src_root, "bad")

    good_paths_all = list_images(good_src_folder)
    bad_paths_all  = list_images(bad_src_folder)

    random.shuffle(good_paths_all)
    random.shuffle(bad_paths_all)

    n_good = len(good_paths_all)
    n_bad  = len(bad_paths_all)

    # -----------------------------
    # 2) Balance: same number of good and bad
    # -----------------------------
    common_n = min(n_good, n_bad)
    good_paths_all = good_paths_all[:common_n]
    bad_paths_all  = bad_paths_all[:common_n]

    print(f"Original: good={n_good}, bad={n_bad}")
    print(f"Balanced to: good={len(good_paths_all)}, bad={len(bad_paths_all)}")

    # -----------------------------
    # 3) Now do per-class train/val split
    # -----------------------------
    class_to_paths = {
        "good": good_paths_all,
        "bad": bad_paths_all,
    }

    for cls, img_paths in class_to_paths.items():
        img_paths.sort()
        random.shuffle(img_paths)

        n_total = len(img_paths)
        n_val   = max(1, int(n_total * val_ratio))  # at least 1 if possible

        val_paths   = img_paths[:n_val]
        train_paths = img_paths[n_val:]

        print(f"Class '{cls}': total_used={n_total}, train={len(train_paths)}, val={len(val_paths)}")

        # process both splits
        for phase, paths in [("train", train_paths), ("val", val_paths)]:
            for img_path in paths:
                fname = os.path.basename(img_path)

                # destination image path
                dst_img = os.path.join(dst_root, phase, cls, fname)
                shutil.copy2(img_path, dst_img)

                # destination mask path
                dst_mask = os.path.join(dst_root, "masks", phase, cls, fname)

                if cls == "good":
                    # GOOD: create an all-zero mask
                    create_zero_mask_like_image(img_path, dst_mask)

                else:  # cls == "bad"
                    # BAD: copy original mask if available, else warn + zero mask
                    src_mask_path = os.path.join(bad_mask_root, fname)
                    if os.path.exists(src_mask_path):
                        shutil.copy2(src_mask_path, dst_mask)
                    else:
                        print(f"[WARN] No mask found for bad image: {fname}. Using zero mask.")
                        create_zero_mask_like_image(img_path, dst_mask)

    print(f"\n✅ Dataset restructure complete!")
    print(f"Balanced train/val images + masks are in: '{dst_root}'")


if __name__ == "__main__":
    main()
