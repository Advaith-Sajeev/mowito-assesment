# validate_efficientnet.py

import os
import shutil
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import efficientnet_b0

import numpy as np
from sklearn.metrics import confusion_matrix, precision_score, recall_score
import matplotlib.pyplot as plt


# -------- Dataset with index so we can recover file paths --------
from torchvision.datasets import ImageFolder

class IndexedImageFolder(ImageFolder):
    def __getitem__(self, index):
        img, label = super().__getitem__(index)
        # return index so we can locate original file path later
        return img, label, index


# -------- Build same model head as in training --------
def build_efficientnet(num_classes=2):
    model = efficientnet_b0(weights=None)  # weights loaded from .pth
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


def main(
    data_root="data",
    model_path="best_efficientnet_b0.pth",
    img_size=224,
    batch_size=32,
    num_workers=4,
    out_dir="val_outputs_efficientnet"
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    val_dir = os.path.join(data_root, "val")

    val_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    val_dataset = IndexedImageFolder(val_dir, transform=val_tf)
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    # class_to_idx (e.g. {'bad': 0, 'good': 1} depending on alphabetical order)
    print("class_to_idx:", val_dataset.class_to_idx)
    bad_idx = val_dataset.class_to_idx["bad"]
    good_idx = val_dataset.class_to_idx["good"]

    # ----- Load model -----
    model = build_efficientnet(num_classes=2).to(device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"Loaded model weights from: {model_path}")

    # ----- Prepare output dirs for misclassified images -----
    mis_root = os.path.join(out_dir, "misclassified_val")
    fp_dir = os.path.join(mis_root, "false_positive_bad")   # predicted bad, actually good
    fn_dir = os.path.join(mis_root, "false_negative_good")  # predicted good, actually bad
    os.makedirs(fp_dir, exist_ok=True)
    os.makedirs(fn_dir, exist_ok=True)

    os.makedirs(out_dir, exist_ok=True)

    # ----- Run inference on val -----
    all_true = []
    all_pred = []

    with torch.no_grad():
        for inputs, labels, indices in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)

            all_true.append(labels.cpu().numpy())
            all_pred.append(preds.cpu().numpy())

            # handle misclassified images
            mismatch = preds != labels
            mismatch_indices = indices[mismatch.cpu()].numpy()
            preds_mis = preds[mismatch].cpu().numpy()
            labels_mis = labels[mismatch].cpu().numpy()

            for ds_idx, p, t in zip(mismatch_indices, preds_mis, labels_mis):
                # original file path
                src_path, _ = val_dataset.samples[ds_idx]
                fname = os.path.basename(src_path)
                # to avoid collisions, prefix with dataset index
                out_name = f"{ds_idx:05d}_{fname}"

                if p == bad_idx and t == good_idx:
                    dst = os.path.join(fp_dir, out_name)  # false positive (bad)
                elif p == good_idx and t == bad_idx:
                    dst = os.path.join(fn_dir, out_name)  # false negative (good)
                else:
                    # other misclassification types (shouldn't occur in pure binary,
                    # but keep for safety)
                    extra_dir = os.path.join(mis_root, "other")
                    os.makedirs(extra_dir, exist_ok=True)
                    dst = os.path.join(extra_dir, out_name)

                shutil.copy2(src_path, dst)

    all_true = np.concatenate(all_true)
    all_pred = np.concatenate(all_pred)

    # ----- Metrics for 'bad' class -----
    # Convert to binary for metrics: 1 = bad, 0 = good
    y_true_bin = (all_true == bad_idx).astype(int)
    y_pred_bin = (all_pred == bad_idx).astype(int)

    precision_bad = precision_score(y_true_bin, y_pred_bin, pos_label=1)
    recall_bad = recall_score(y_true_bin, y_pred_bin, pos_label=1)
    acc = (all_true == all_pred).mean()

    print(f"\nPrecision (bad): {precision_bad:.4f}")
    print(f"Recall    (bad): {recall_bad:.4f}")
    print(f"Accuracy       : {acc:.4f}")

    # ----- Confusion matrix -----
    # Order: [good, bad] for rows/cols
    labels_order = [good_idx, bad_idx]
    cm = confusion_matrix(all_true, all_pred, labels=labels_order)

    print("\nConfusion matrix (rows=true, cols=pred) in order [good, bad]:")
    print(cm)

    # Plot confusion matrix
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.set_title(
        f"EfficientNet Confusion Matrix (Val)\n"
        f"Prec(bad)={precision_bad:.3f}, Rec(bad)={recall_bad:.3f}"
    )
    plt.colorbar(im, ax=ax)

    class_names = ["good", "bad"]
    tick_marks = np.arange(len(class_names))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")

    # Annotate each cell with count
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    cm_path = os.path.join(out_dir, "confusion_matrix_val_efficientnet.png")
    plt.tight_layout()
    plt.savefig(cm_path, dpi=200)
    plt.close()

    print(f"\nConfusion matrix image saved to: {cm_path}")
    print(f"Misclassified images saved under: {mis_root}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument("--model_path", type=str, default="best_efficientnet_b0.pth")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--out_dir", type=str, default="val_outputs_efficientnet")
    args = parser.parse_args()

    main(
        data_root=args.data_root,
        model_path=args.model_path,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        out_dir=args.out_dir,
    )
