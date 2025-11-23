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
from utils import AverageMeter

import matplotlib.pyplot as plt  # NEW: for confusion matrix figure


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_path',
        type=str,
        default=None,
        help='Direct path to model.pth file (e.g., ../../models/unext_multitask/model.pth)'
    )
    parser.add_argument(
        '--name',
        default=None,
        help='model name (legacy: looks in models/{name}/model.pth)'
    )
    parser.add_argument(
        '--data_root',
        default='../../data',
        help='Path to data directory containing train/val folders'
    )
    parser.add_argument(
        '--output_root',
        default='outputs',
        help='root folder to save panels'
    )
    return parser.parse_args()


def compute_iou_dice(pred_mask, gt_mask):
    """
    pred_mask, gt_mask: numpy arrays of shape (H, W), values {0,1}
    Returns (IoU, Dice). If both masks are empty, returns (1.0, 1.0).
    """
    pred_bool = pred_mask.astype(bool)
    gt_bool = gt_mask.astype(bool)

    inter = np.logical_and(pred_bool, gt_bool).sum()
    union = np.logical_or(pred_bool, gt_bool).sum()
    pred_sum = pred_bool.sum()
    gt_sum = gt_bool.sum()

    if union == 0:
        # both empty -> perfect match on background
        return 1.0, 1.0

    iou = inter / (union + 1e-8)
    dice = 2.0 * inter / (pred_sum + gt_sum + 1e-8)
    return float(iou), float(dice)


def main():
    args = parse_args()
    
    # Determine model path and config path
    if args.model_path:
        # User provided direct model path
        state_dict_path = args.model_path
        if not os.path.exists(state_dict_path):
            raise FileNotFoundError(f"Model file not found: {state_dict_path}")
        
        # Try to find config.yml in same directory as model
        model_dir = os.path.dirname(state_dict_path)
        config_path = os.path.join(model_dir, 'config.yml')
        
        if os.path.exists(config_path):
            print(f"Loading config from: {config_path}")
            with open(config_path, 'r') as f:
                config = yaml.load(f, Loader=yaml.FullLoader)
        else:
            # Use default config if no config.yml found
            print(f"Warning: No config.yml found at {config_path}")
            print("Using default configuration")
            config = {
                'name': os.path.basename(model_dir),
                'arch': 'UNext',
                'num_classes': 1,
                'input_channels': 3,
                'deep_supervision': False,
                'input_h': 256,
                'input_w': 256,
                'num_workers': 4,
                'cls_threshold': 0.5,
            }
    elif args.name:
        # Legacy mode: use name to find model in models/{name}/
        config_path = f'models/{args.name}/config.yml'
        state_dict_path = f'models/{args.name}/model.pth'
        
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
    else:
        raise ValueError("Either --model_path or --name must be provided")
    
    # Override data_root if provided
    if args.data_root:
        config['data_root'] = args.data_root

    print('-' * 20)
    for key in config.keys():
        print(f'{key}: {config[key]}')
    print('-' * 20)

    cudnn.benchmark = True

    # paths & thresholds
    data_root = config.get('data_root', 'data')
    mask_root = config.get('mask_root', os.path.join(data_root, 'masks'))
    cls_threshold = float(config.get('cls_threshold', 0.5))

    # ---- create model ----
    print(f"=> creating model {config['arch']}")
    model = archs.__dict__[config['arch']](
        config['num_classes'],
        config['input_channels'],
        config['deep_supervision']
    )
    model = model.cuda()

    # ---- load weights ----
    print(f"=> loading weights from {state_dict_path}")
    model.load_state_dict(torch.load(state_dict_path))
    model.eval()

    # ---- transforms for validation ----
    val_transform = Compose([
        Resize(config['input_h'], config['input_w']),
        transforms.Normalize(),
        ToTensorV2(),
    ])

    # ---- dataset & loader ----
    val_dataset = Dataset(
        img_root=data_root,
        mask_root=mask_root,
        phase='val',
        transform=val_transform,
        image_size=(config['input_h'], config['input_w'])
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,   # easier to make per-image panels
        shuffle=False,
        num_workers=config['num_workers'],
        drop_last=False
    )

    # ---- metrics ----
    iou_meter = AverageMeter()
    dice_meter = AverageMeter()

    # classification confusion for "bad" = 1
    TP = FP = FN = TN = 0

    # ---- output dir for panels ----
    panel_dir = os.path.join(args.output_root, config['name'], 'val_panels')
    os.makedirs(panel_dir, exist_ok=True)

    print(f"Saving panels to: {panel_dir}")

    with torch.no_grad():
        for inputs, targets, labels, meta in tqdm(val_loader, total=len(val_loader)):
            # batch_size = 1
            inputs = inputs.cuda()           # (1,3,H,W)
            targets = targets.cuda()         # (1,1,H,W)
            labels = labels.cuda().float()   # (1,)

            img_id = meta['img_id'][0]
            img_path = meta['img_path'][0]
            gt_label = int(labels.item())

            # ---- forward ----
            seg_logits, cls_logit = model(inputs)

            # classification prediction
            cls_prob = torch.sigmoid(cls_logit).view(-1)[0].item()
            pred_label = int(cls_prob >= cls_threshold)

            # confusion matrix
            if pred_label == 1 and gt_label == 1:
                TP += 1
            elif pred_label == 1 and gt_label == 0:
                FP += 1
            elif pred_label == 0 and gt_label == 1:
                FN += 1
            else:
                TN += 1

            # ---- masks (GT and Pred) ----
            # GT mask is already resized & in [0,1]
            gt_mask = targets[0, 0].detach().cpu().numpy()  # (H,W)
            gt_mask_bin = (gt_mask >= 0.5).astype(np.uint8)

            # Predicted mask from seg_logits
            seg_prob = torch.sigmoid(seg_logits)[0, 0].detach().cpu().numpy()  # (H,W)
            pred_mask_bin = (seg_prob >= 0.5).astype(np.uint8)

            # ---- IoU & Dice per image ----
            iou_val, dice_val = compute_iou_dice(pred_mask_bin, gt_mask_bin)
            iou_meter.update(iou_val, 1)
            dice_meter.update(dice_val, 1)

            # ---- load original image ----
            orig = cv2.imread(img_path)  # BGR
            if orig is None:
                # fallback from tensor (de-normalized approx)
                x = inputs[0].detach().cpu().numpy().transpose(1, 2, 0)
                x = (x - x.min()) / (x.max() - x.min() + 1e-8)
                orig = (x * 255.0).astype('uint8')
            else:
                orig = cv2.resize(
                    orig,
                    (config['input_w'], config['input_h']),
                    interpolation=cv2.INTER_LINEAR,
                )

            left_panel = orig.copy()
            right_panel = orig.copy().astype(np.float32)

            # ---- overlay GT mask in blue (BGR: 255,0,0) ----
            gt_indices = gt_mask_bin.astype(bool)
            right_panel[gt_indices] = 0.5 * right_panel[gt_indices] + 0.5 * np.array([255, 0, 0], dtype=np.float32)

            # ---- overlay Pred mask in red (BGR: 0,0,255) ----
            pred_indices = pred_mask_bin.astype(bool)
            right_panel[pred_indices] = 0.5 * right_panel[pred_indices] + 0.5 * np.array([0, 0, 255], dtype=np.float32)

            right_panel = np.clip(right_panel, 0, 255).astype('uint8')

            # ---- side-by-side (left: original, right: GT+Pred overlays) ----
            panel = np.concatenate([left_panel, right_panel], axis=1)  # (H, 2W, 3)

            # ---- add bottom text with metrics ----
            text = (
                f"IoU={iou_val:.3f}  Dice={dice_val:.3f}  "
                f"gt_label={gt_label}  pred_label={pred_label}  "
                f"prob_bad={cls_prob:.3f}"
            )
            H, W, _ = panel.shape
            bar_h = 26

            overlay = panel.copy()
            cv2.rectangle(
                overlay,
                (0, H - bar_h),
                (W, H),
                (0, 0, 0),
                -1,
            )
            alpha = 0.5
            panel = cv2.addWeighted(overlay, alpha, panel, 1 - alpha, 0)

            cv2.putText(
                panel,
                text,
                (10, H - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),  # yellow
                1,
                cv2.LINE_AA,
            )

            # ---- save panel ----
            out_name = f"{img_id}_gt{gt_label}_pred{pred_label}.png"
            out_path = os.path.join(panel_dir, out_name)
            cv2.imwrite(out_path, panel)

            # ---- print per-image metrics ----
            print(
                f"{img_id}: IoU={iou_val:.4f}, Dice={dice_val:.4f}, "
                f"gt_label={gt_label}, pred_label={pred_label}, prob_bad={cls_prob:.3f}"
            )

    # ---- final aggregate metrics ----
    print("============== SUMMARY METRICS ==============")
    print("Segmentation IoU (avg over all val images):  %.4f" % iou_meter.avg)
    print("Segmentation Dice (avg over all val images): %.4f" % dice_meter.avg)

    total = TP + FP + FN + TN
    if total > 0:
        recall_bad = TP / (TP + FN + 1e-8)
        precision_bad = TP / (TP + FP + 1e-8) if (TP + FP) > 0 else 0.0
        accuracy = (TP + TN) / (total + 1e-8)
        f1_bad = (
            2 * precision_bad * recall_bad / (precision_bad + recall_bad + 1e-8)
            if (precision_bad + recall_bad) > 0
            else 0.0
        )

        print("Classification metrics (bad = positive class):")
        print("  Recall (bad):    %.4f" % recall_bad)
        print("  Precision (bad): %.4f" % precision_bad)
        print("  F1 (bad):        %.4f" % f1_bad)
        print("  Accuracy:        %.4f" % accuracy)
        print(f"  TP={TP} FP={FP} FN={FN} TN={TN}")

        # ============================================================
        #      NEW: SAVE CONFUSION MATRIX & HTML SUMMARY REPORT
        # ============================================================
        cm = np.array([[TN, FP],
                       [FN, TP]])

        # Directory for confusion matrix & report
        report_dir = os.path.join(args.output_root, config['name'])
        os.makedirs(report_dir, exist_ok=True)

        # ---- Confusion matrix figure ----
        fig, ax = plt.subplots(figsize=(5, 5))
        im = ax.imshow(cm, cmap='Blues')

        ax.set_title(
            f"Confusion Matrix (Val)\nPrec(bad)={precision_bad:.3f}  Rec(bad)={recall_bad:.3f}"
        )
        plt.colorbar(im)

        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['Pred Good', 'Pred Bad'])
        ax.set_yticklabels(['True Good', 'True Bad'])
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')

        thresh = cm.max() / 2.0
        for i in range(2):
            for j in range(2):
                ax.text(
                    j, i, int(cm[i, j]),
                    ha='center', va='center',
                    color='white' if cm[i, j] > thresh else 'black'
                )

        cm_save_path = os.path.join(report_dir, "confusion_matrix_val.png")
        plt.tight_layout()
        plt.savefig(cm_save_path, dpi=200)
        plt.close()

        print(f"\nSaved confusion matrix to: {cm_save_path}")

        # ---- Simple HTML report ----
        html_path = os.path.join(report_dir, "val_report.html")
        rel_cm_path = "confusion_matrix_val.png"  # relative to HTML file

        html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Validation Report - {config['name']}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1, h2 {{ color: #333; }}
        table {{ border-collapse: collapse; margin-top: 10px; }}
        th, td {{ border: 1px solid #aaa; padding: 6px 10px; }}
        th {{ background-color: #eee; }}
        .metric-table {{ margin-bottom: 20px; }}
        img {{ max-width: 500px; border: 1px solid #ccc; }}
        .small {{ color: #666; font-size: 0.9em; }}
    </style>
</head>
<body>
    <h1>Validation Report – {config['name']}</h1>

    <h2>Segmentation Metrics</h2>
    <table class="metric-table">
        <tr><th>Average IoU</th><td>{iou_meter.avg:.4f}</td></tr>
        <tr><th>Average Dice</th><td>{dice_meter.avg:.4f}</td></tr>
    </table>

    <h2>Classification Metrics (bad = positive class)</h2>
    <table class="metric-table">
        <tr><th>Precision (bad)</th><td>{precision_bad:.4f}</td></tr>
        <tr><th>Recall (bad)</th><td>{recall_bad:.4f}</td></tr>
        <tr><th>F1 (bad)</th><td>{f1_bad:.4f}</td></tr>
        <tr><th>Accuracy</th><td>{accuracy:.4f}</td></tr>
        <tr><th>TP</th><td>{TP}</td></tr>
        <tr><th>FP</th><td>{FP}</td></tr>
        <tr><th>FN</th><td>{FN}</td></tr>
        <tr><th>TN</th><td>{TN}</td></tr>
        <tr><th>Total Samples</th><td>{total}</td></tr>
    </table>

    <h2>Confusion Matrix</h2>
    <p class="small">Rows: True class (Good, Bad) – Columns: Predicted class (Good, Bad)</p>
    <img src="{rel_cm_path}" alt="Confusion Matrix">

    <h2>Panels</h2>
    <p class="small">
        Panels (original + GT & predicted overlays) are saved in:<br>
        <code>{panel_dir}</code>
    </p>
</body>
</html>
"""
        with open(html_path, "w") as f:
            f.write(html)

        print(f"Saved HTML validation report to: {html_path}")
        # ============================================================

    else:
        print("No classification samples? (this should not happen for val split)")

    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
