import argparse
import os

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import yaml
from albumentations.augmentations import transforms
from albumentations.core.composition import Compose
from albumentations import Resize
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

import archs
from dataset import Dataset


class GradCAM:
    """
    Simple Grad-CAM for a single target layer and single-class (binary) output.
    Hooks a target module (e.g., encoder3) and produces a normalized CAM heatmap.
    """

    def __init__(self, model, target_module):
        self.model = model
        self.target_module = target_module

        self.activations = None
        self.gradients = None

        self.fwd_hook = self.target_module.register_forward_hook(self._forward_hook)
        self.bwd_hook = self.target_module.register_backward_hook(self._backward_hook)

    def _forward_hook(self, module, input, output):
        # Feature maps
        self.activations = output.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        # Gradients wrt feature maps
        self.gradients = grad_output[0].detach()

    def generate(self, target_score):
        """
        target_score: scalar tensor (sum of logits for target class).
        Must call after forward(model(x)).
        """
        self.model.zero_grad()
        target_score.backward(retain_graph=True)

        # activations: [B, C, H, W]
        # gradients: [B, C, H, W]
        grads = self.gradients
        acts = self.activations

        # global average pooling on gradients -> channel weights
        weights = grads.mean(dim=(2, 3), keepdim=True)  # [B, C, 1, 1]

        # weighted sum of activations
        cam = (weights * acts).sum(dim=1, keepdim=True)  # [B, 1, H, W]
        cam = torch.relu(cam)

        # normalize to [0,1]
        cam_min, cam_max = cam.min(), cam.max()
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)

        return cam  # [B,1,Hc,Wc]


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--name",
        required=True,
        help="model name (same as used in training, e.g. scratch_UNext_cls)",
    )
    parser.add_argument(
        "--phase",
        default="val",
        choices=["train", "val"],
        help="dataset split to run Grad-CAM on",
    )
    parser.add_argument(
        "--only_bad_pred",
        action="store_true",
        help="only save panels for images predicted as bad (1, after size filter)",
    )
    parser.add_argument(
        "--only_bad_gt",
        action="store_true",
        help="only process images whose ground truth label is bad (1)",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=-1,
        help="max number of images to process (-1 = all)",
    )
    parser.add_argument(
        "--output_root",
        default="gradcam_outputs",
        help="root directory to save Grad-CAM panels",
    )
    parser.add_argument(
        "--cam_thresh",
        type=float,
        default=0.4,
        help="threshold on normalized CAM [0,1] to define scratch region (binary mask)",
    )
    parser.add_argument(
        "--scratch_size_threshold",
        type=float,
        default=0.0,
        help=(
            "minimum fraction of image area that must be highlighted to "
            "consider image as bad (after Grad-CAM). "
            "If 0.0, classification is NOT overridden by scratch size."
        ),
    )
    parser.add_argument(
        "--mask_root",
        type=str,
        default=None,
        help=(
            "Path to root folder of original GT masks "
            "(overrides mask_root from config.yml if given)."
        ),
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # ---- load config ----
    with open(f"models/{args.name}/config.yml", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    print("-" * 20)
    for key in config.keys():
        print(f"{key}: {config[key]}")
    print("-" * 20)

    cudnn.benchmark = True

    # data and mask roots
    data_root = config.get("data_root", "data")
    if args.mask_root is not None:
        mask_root = args.mask_root
    else:
        mask_root = config.get("mask_root", os.path.join(data_root, "masks"))

    cls_threshold = float(config.get("cls_threshold", 0.5))

    # ---- create model ----
    print(f"=> creating model {config['arch']}")
    model = archs.__dict__[config["arch"]](
        config["num_classes"], config["input_channels"], config["deep_supervision"]
    )
    model = model.cuda()

    # ---- load weights ----
    state_dict_path = f"models/{config['name']}/model.pth"
    print(f"=> loading weights from {state_dict_path}")
    model.load_state_dict(torch.load(state_dict_path))
    model.eval()

    # ---- Grad-CAM on encoder3 ----
    target_module = getattr(model, "encoder3")
    grad_cam = GradCAM(model, target_module)

    # ---- transforms ----
    transform = Compose(
        [
            Resize(config["input_h"], config["input_w"]),
            transforms.Normalize(),
            ToTensorV2(),
        ]
    )

    # ---- dataset (uses img_root + mask_root) ----
    dataset = Dataset(
        img_root=data_root,
        mask_root=mask_root,
        phase=args.phase,
        transform=transform,
        image_size=(config["input_h"], config["input_w"]),
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,  # Grad-CAM is per-image
        shuffle=False,
        num_workers=config.get("num_workers", 4),
        drop_last=False,
    )

    # ---- output dir ----
    out_dir = os.path.join(args.output_root, args.name, args.phase)
    os.makedirs(out_dir, exist_ok=True)

    count = 0
    total = len(loader)
    print(f"Running Grad-CAM on {total} samples from phase='{args.phase}'")

    # classification confusion for bad=1
    TP = FP = FN = TN = 0
    mask_ious = []

    for inputs, masks, labels, meta in tqdm(loader, total=len(loader)):
        if args.max_samples > 0 and count >= args.max_samples:
            break

        img_id = meta["img_id"][0]
        img_path = meta["img_path"][0]
        labels = labels.cuda().float()  # [1]
        gt = int(labels.item())

        # optional: only GT-bad images
        if args.only_bad_gt and gt != 1:
            continue

        inputs = inputs.cuda()  # [1,3,H,W]

        # Forward (seg_logits unused here)
        seg_logits, cls_logit = model(inputs)  # cls_logit shape: [1]
        probs = torch.sigmoid(cls_logit).view(-1)
        base_pred = int((probs >= cls_threshold).long().item())

        # ---- Grad-CAM for the "bad" class ----
        model.zero_grad()
        target_score = cls_logit.sum()  # scalar (binary classifier, single logit)
        cam = grad_cam.generate(target_score)  # [1,1,Hc,Wc]

        cam_np = cam.squeeze().detach().cpu().numpy()  # [Hc,Wc]
        cam_np = cv2.resize(
            cam_np,
            (config["input_w"], config["input_h"]),
            interpolation=cv2.INTER_LINEAR,
        )

        # ---- Binary mask from CAM ----
        H, W = cam_np.shape
        binary_mask = (cam_np >= args.cam_thresh).astype("uint8")
        scratch_fraction = float(binary_mask.mean())

        # optional classification override based on scratch size
        size_filtered_pred = base_pred
        if args.scratch_size_threshold > 0.0:
            if scratch_fraction < args.scratch_size_threshold:
                size_filtered_pred = 0

        # if user only wants images predicted as bad (after size filter)
        if args.only_bad_pred and size_filtered_pred != 1:
            continue

        # classification confusion update
        if size_filtered_pred == 1 and gt == 1:
            TP += 1
        elif size_filtered_pred == 1 and gt == 0:
            FP += 1
        elif size_filtered_pred == 0 and gt == 1:
            FN += 1
        elif size_filtered_pred == 0 and gt == 0:
            TN += 1

        # ---- load original image for visualization ----
        orig = cv2.imread(img_path)  # BGR
        if orig is None:
            orig = inputs[0].detach().cpu().numpy().transpose(1, 2, 0)
            orig = (orig - orig.min()) / (orig.max() - orig.min() + 1e-8)
            orig = (orig * 255.0).astype("uint8")
        else:
            orig = cv2.resize(
                orig,
                (config["input_w"], config["input_h"]),
                interpolation=cv2.INTER_LINEAR,
            )
        orig_bgr = orig.copy()

        # ---- GT mask (from mask_root) ----
        gt_mask = masks[0].detach().cpu().numpy()  # [C,H,W] or [H,W]
        if gt_mask.ndim == 3:
            gt_mask = gt_mask[0]  # assume first channel
        gt_mask_bin = (gt_mask >= 0.5).astype("uint8")

        # mask IoU (pixel-wise)
        inter = np.logical_and(gt_mask_bin == 1, binary_mask == 1).sum()
        union = np.logical_or(gt_mask_bin == 1, binary_mask == 1).sum()
        mask_iou = float(inter) / float(union) if union > 0 else 0.0
        mask_ious.append(mask_iou)

        # ---- draw GT mask (blue) + predicted mask (red) on right panel ----
        right_panel = orig_bgr.astype(np.float32)

        # GT in blue (BGR: 255, 0, 0)
        gt_mask_3 = gt_mask_bin[..., None]  # [H,W,1]
        blue_color = np.array([255.0, 0.0, 0.0], dtype=np.float32)
        right_panel = np.where(
            gt_mask_3 == 1,
            0.5 * right_panel + 0.5 * blue_color,
            right_panel,
        )

        # Pred in red (BGR: 0, 0, 255)
        pred_mask_3 = binary_mask[..., None]
        red_color = np.array([0.0, 0.0, 255.0], dtype=np.float32)
        right_panel = np.where(
            pred_mask_3 == 1,
            0.5 * right_panel + 0.5 * red_color,
            right_panel,
        )

        right_panel = np.clip(right_panel, 0, 255).astype("uint8")

        # ---- create side-by-side panel (left: original, right: GT+pred masks) ----
        left_panel = orig_bgr.copy()
        panel = np.concatenate([left_panel, right_panel], axis=1)  # [H, 2W, 3]

        # ---- add metrics text at bottom in yellow ----
        txt = (
            f"gt={gt} pred={size_filtered_pred} "
            f"prob_bad={probs.item():.3f} "
            f"area={scratch_fraction:.3f} "
            f"maskIoU={mask_iou:.3f}"
        )
        H_panel, W_panel, _ = panel.shape
        bar_h = 24
        overlay_img = panel.copy()
        cv2.rectangle(
            overlay_img,
            (0, H_panel - bar_h),
            (W_panel, H_panel),
            (0, 0, 0),
            -1,
        )
        alpha = 0.5
        panel = cv2.addWeighted(overlay_img, alpha, panel, 1 - alpha, 0)
        cv2.putText(
            panel,
            txt,
            (10, H_panel - 7),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),  # yellow
            1,
            cv2.LINE_AA,
        )

        # ---- filename (info encoded) ----
        base_name = (
            f"{img_id}_gt{gt}_pred{size_filtered_pred}"
            f"_prob{probs.item():.3f}_area{scratch_fraction:.4f}"
        )
        panel_path = os.path.join(out_dir, base_name + "_panel.png")

        cv2.imwrite(panel_path, panel)

        count += 1

    # ---- summary metrics ----
    print(f"Saved Grad-CAM panels for {count} images to: {out_dir}")

    # classification metrics
    total_cls = TP + FP + FN + TN
    if total_cls > 0:
        recall_bad = TP / (TP + FN + 1e-8)
        precision_bad = TP / (TP + FP + 1e-8) if (TP + FP) > 0 else 0.0
        accuracy = (TP + TN) / (total_cls + 1e-8)
        f1_bad = (
            2 * precision_bad * recall_bad / (precision_bad + recall_bad + 1e-8)
            if (precision_bad + recall_bad) > 0
            else 0.0
        )
        print("==== Grad-CAM classification (after size filter) ====")
        print(f"TP={TP} FP={FP} FN={FN} TN={TN}")
        print(f"Recall (bad=1):   {recall_bad:.4f}")
        print(f"Precision (bad=1):{precision_bad:.4f}")
        print(f"F1 (bad=1):       {f1_bad:.4f}")
        print(f"Accuracy:         {accuracy:.4f}")
    else:
        print("No samples contributed to classification metrics (maybe filtered by flags).")

    if len(mask_ious) > 0:
        mean_mask_iou = float(np.mean(mask_ious))
        print(f"Mean mask IoU (CAM vs GT mask):  {mean_mask_iou:.4f}")
    else:
        print("No mask IoU computed.")


if __name__ == "__main__":
    main()
