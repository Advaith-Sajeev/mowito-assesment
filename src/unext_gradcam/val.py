import argparse
import os
import shutil

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import yaml
import pandas as pd
from albumentations.augmentations import transforms
from albumentations.core.composition import Compose
from albumentations import Resize
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

import archs
from dataset import Dataset
from utils import AverageMeter


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--name',
        required=True,
        help='model name (same as used in training, e.g. scratch_UNext_cls)'
    )
    parser.add_argument(
        '--csv_out',
        default=None,
        help='path to save predictions CSV '
             '(default: models/<name>/val_predictions.csv)'
    )
    parser.add_argument(
        '--miscls_dir',
        default=None,
        help='directory to save misclassified images '
             '(default: models/<name>/misclassified)'
    )
    parser.add_argument(
        '--tta',
        action='store_true',
        help='enable simple test-time augmentation (original + horizontal flip)'
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # ---- load training config ----
    with open(f'models/{args.name}/config.yml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    print('-' * 20)
    for key in config.keys():
        print(f'{key}: {config[key]}')
    print('-' * 20)

    cudnn.benchmark = True

    # default paths if not present in config (for backward compat)
    data_root = config.get('data_root', 'data')
    mask_root = config.get('mask_root', os.path.join(data_root, 'masks'))
    cls_threshold = float(config.get('cls_threshold', 0.5))
    pos_weight_val = float(config.get('pos_weight', 1.0))

    # default CSV and misclassified paths
    if args.csv_out is None:
        args.csv_out = os.path.join('models', args.name, 'val_predictions.csv')
    if args.miscls_dir is None:
        args.miscls_dir = os.path.join('models', args.name, 'misclassified')

    os.makedirs(os.path.dirname(args.csv_out), exist_ok=True)
    os.makedirs(args.miscls_dir, exist_ok=True)

    print(f"=> creating model {config['arch']}")
    model = archs.__dict__[config['arch']](
        config['num_classes'],
        config['input_channels'],
        config['deep_supervision']
    )
    model = model.cuda()

    # ---- load weights ----
    state_dict_path = f'models/{config["name"]}/model.pth'
    print(f"=> loading weights from {state_dict_path}")
    model.load_state_dict(torch.load(state_dict_path))
    model.eval()

    # ---- loss (same as in train) ----
    if pos_weight_val != 1.0:
        pos_w = torch.tensor([pos_weight_val], dtype=torch.float32).cuda()
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w).cuda()
    else:
        criterion = nn.BCEWithLogitsLoss().cuda()

    # ---- transforms for validation ----
    val_transform = Compose([
        Resize(config['input_h'], config['input_w']),
        transforms.Normalize(),
        ToTensorV2(),
    ])

    # ---- dataset & loader: we use the val split from data_root ----
    val_dataset = Dataset(
        img_root=data_root,
        mask_root=mask_root,
        phase='val',
        transform=val_transform,
        image_size=(config['input_h'], config['input_w'])
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        drop_last=False
    )

    # ---- meters for loss & accuracy ----
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    # classification confusion counts for "bad" class = 1
    TP = FP = FN = TN = 0

    # rows for CSV
    rows = []

    with torch.no_grad():
        for inputs, _, labels, meta in tqdm(val_loader, total=len(val_loader)):
            inputs = inputs.cuda()          # (B,3,H,W)
            labels = labels.cuda().float()  # (B,)  0. or 1.

            # ---- forward: seg + cls (we ignore seg_logits) ----
            if args.tta:
                # simple TTA: original + horizontal flip
                # original
                _, cls_logit_orig = model(inputs)
                # horizontal flip
                inputs_flipped = torch.flip(inputs, dims=[3])  # flip width dimension
                _, cls_logit_flip = model(inputs_flipped)
                # average logits
                cls_logit = (cls_logit_orig + cls_logit_flip) / 2.0
            else:
                _, cls_logit = model(inputs)

            # loss
            loss = criterion(cls_logit, labels)

            # predictions
            probs = torch.sigmoid(cls_logit).view(-1)  # (B,)
            preds = (probs >= cls_threshold).long()
            gt = labels.long().view(-1)

            correct = (preds == gt).sum().item()
            acc = correct / labels.size(0)

            loss_meter.update(loss.item(), inputs.size(0))
            acc_meter.update(acc, inputs.size(0))

            # confusion counts
            TP += ((preds == 1) & (gt == 1)).sum().item()
            FP += ((preds == 1) & (gt == 0)).sum().item()
            FN += ((preds == 0) & (gt == 1)).sum().item()
            TN += ((preds == 0) & (gt == 0)).sum().item()

            # ---- collect per-sample info for CSV and misclassified saving ----
            batch_size = preds.size(0)
            for i in range(batch_size):
                img_id = meta['img_id'][i]
                img_path = meta.get('img_path', [None] * batch_size)[i]

                p = float(probs[i].item())
                y_true = int(gt[i].item())
                y_pred = int(preds[i].item())
                is_correct = int(y_true == y_pred)

                rows.append({
                    'img_id': img_id,
                    'img_path': img_path,
                    'label': y_true,
                    'pred': y_pred,
                    'prob_bad': p,
                    'correct': is_correct,
                })

                # save misclassified image for debugging
                if y_true != y_pred and img_path is not None and os.path.isfile(img_path):
                    # create a name like <img_id>_true<y>_pred<p>.<ext>
                    base_name = os.path.basename(img_path)
                    root, ext = os.path.splitext(base_name)
                    out_name = f"{root}_true{y_true}_pred{y_pred}{ext}"
                    out_path = os.path.join(args.miscls_dir, out_name)
                    # avoid overwriting by adding a suffix if needed
                    if os.path.exists(out_path):
                        k = 1
                        while True:
                            alt_name = f"{root}_true{y_true}_pred{y_pred}_{k}{ext}"
                            alt_path = os.path.join(args.miscls_dir, alt_name)
                            if not os.path.exists(alt_path):
                                out_path = alt_path
                                break
                            k += 1
                    shutil.copy2(img_path, out_path)

    # ---- final metrics ----
    recall_bad = TP / (TP + FN + 1e-8)
    precision_bad = TP / (TP + FP + 1e-8)
    accuracy = (TP + TN) / (TP + TN + FP + FN + 1e-8)
    f1_bad = 2 * precision_bad * recall_bad / (precision_bad + recall_bad + 1e-8)

    print('================ EVAL RESULTS ================')
    print('Val loss:           %.4f' % loss_meter.avg)
    print('Val accuracy:       %.4f' % acc_meter.avg)
    print('Recall (bad = 1):   %.4f' % recall_bad)
    print('Precision (bad = 1):%.4f' % precision_bad)
    print('F1 (bad = 1):       %.4f' % f1_bad)
    print('Confusion matrix (bad=1):')
    print('  TP=%d  FP=%d  FN=%d  TN=%d' % (TP, FP, FN, TN))

    # ---- save CSV of predictions ----
    df = pd.DataFrame(rows)
    df.to_csv(args.csv_out, index=False)
    print(f"Saved predictions CSV to: {args.csv_out}")
    print(f"Saved misclassified images to: {args.miscls_dir}")

    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
