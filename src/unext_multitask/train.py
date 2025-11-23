import os
from collections import OrderedDict

import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import yaml
import albumentations as A
from albumentations.augmentations import transforms
from albumentations.core.composition import Compose
from albumentations import RandomRotate90, Resize
from albumentations.pytorch import ToTensorV2
from torch.optim import lr_scheduler
from tqdm import tqdm

import archs
import losses
from dataset import Dataset
from metrics import iou_score
from utils import AverageMeter
from archs import UNext  # keep for clarity


# =========================
# CONFIG AS A DICTIONARY
# =========================
CONFIG = {
    # ---- basic run info ----
    "name": "scratch_UNext_multitask_synth",   # folder under models/
    "epochs": 2000,
    "batch_size": 64,

    # ---- model ----
    "arch": "UNext",
    "deep_supervision": False,
    "input_channels": 3,
    "num_classes": 1,          # segmentation output channels
    "input_w": 256,
    "input_h": 256,

    # ---- dataset ----
    # data_root should contain: train/good, train/bad, val/good, val/bad
    "data_root": "data_synth",
    # mask_root should contain: train/good, train/bad, val/good, val/bad
    # If None, will default to data_root/masks
    "mask_root": None,

    # ---- loss ----
    "loss": "BCEDiceLoss",     # or "BCEWithLogitsLoss"
    
    # ---- optimizer ----
    "optimizer": "Adam",       # "Adam" or "SGD"
    "lr": 1e-3,
    "momentum": 0.9,
    "weight_decay": 1e-4,
    "nesterov": False,

    # ---- scheduler ----
    "scheduler": "CosineAnnealingLR",  # "CosineAnnealingLR", "ReduceLROnPlateau", "MultiStepLR", "ConstantLR"
    "min_lr": 1e-5,
    "factor": 0.1,             # for ReduceLROnPlateau
    "patience": 2,             # for ReduceLROnPlateau
    "milestones": "1,2",       # for MultiStepLR
    "gamma": 2/3,              # for MultiStepLR
    "early_stopping": -1,      # -1 to disable

    # ---- dataloader ----
    "num_workers": 4,

    # ---- classification head ----
    "cls_loss_weight": 0.5,    # L = L_seg + cls_loss_weight * L_cls
    "cls_threshold": 0.5,      # threshold on sigmoid(cls_logit) to classify bad (1)
}


def train(config, train_loader, model, seg_criterion, cls_criterion, optimizer):
    avg_meters = {
        "loss": AverageMeter(),
        "seg_loss": AverageMeter(),
        "cls_loss": AverageMeter(),
        "iou": AverageMeter(),
    }

    model.train()

    pbar = tqdm(total=len(train_loader))
    for inputs, targets, labels, _ in train_loader:
        inputs = inputs.cuda()
        targets = targets.cuda()
        labels = labels.cuda().float()  # 0. or 1.

        # forward
        seg_logits, cls_logit = model(inputs)

        # losses
        seg_loss = seg_criterion(seg_logits, targets)
        cls_loss = cls_criterion(cls_logit, labels)
        loss = seg_loss + config["cls_loss_weight"] * cls_loss

        # metrics (seg)
        iou, dice = iou_score(seg_logits, targets)

        # backward + optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # update meters
        avg_meters["loss"].update(loss.item(), inputs.size(0))
        avg_meters["seg_loss"].update(seg_loss.item(), inputs.size(0))
        avg_meters["cls_loss"].update(cls_loss.item(), inputs.size(0))
        avg_meters["iou"].update(iou, inputs.size(0))

        postfix = OrderedDict(
            [
                ("loss", avg_meters["loss"].avg),
                ("seg_loss", avg_meters["seg_loss"].avg),
                ("cls_loss", avg_meters["cls_loss"].avg),
                ("iou", avg_meters["iou"].avg),
            ]
        )
        pbar.set_postfix(postfix)
        pbar.update(1)
    pbar.close()

    return OrderedDict(
        [
            ("loss", avg_meters["loss"].avg),
            ("seg_loss", avg_meters["seg_loss"].avg),
            ("cls_loss", avg_meters["cls_loss"].avg),
            ("iou", avg_meters["iou"].avg),
        ]
    )


def validate(config, val_loader, model, seg_criterion, cls_criterion):
    avg_meters = {
        "loss": AverageMeter(),
        "seg_loss": AverageMeter(),
        "cls_loss": AverageMeter(),
        "iou": AverageMeter(),
        "dice": AverageMeter(),
    }

    # classification confusion counts for "bad" class = 1
    TP = FP = FN = TN = 0

    model.eval()

    with torch.no_grad():
        pbar = tqdm(total=len(val_loader))
        for inputs, targets, labels, _ in val_loader:
            inputs = inputs.cuda()
            targets = targets.cuda()
            labels = labels.cuda().float()

            seg_logits, cls_logit = model(inputs)

            seg_loss = seg_criterion(seg_logits, targets)
            cls_loss = cls_criterion(cls_logit, labels)
            loss = seg_loss + config["cls_loss_weight"] * cls_loss

            iou, dice = iou_score(seg_logits, targets)

            avg_meters["loss"].update(loss.item(), inputs.size(0))
            avg_meters["seg_loss"].update(seg_loss.item(), inputs.size(0))
            avg_meters["cls_loss"].update(cls_loss.item(), inputs.size(0))
            avg_meters["iou"].update(iou, inputs.size(0))
            avg_meters["dice"].update(dice, inputs.size(0))

            # classification metrics
            probs = torch.sigmoid(cls_logit)
            preds = (probs >= config["cls_threshold"]).long()
            gt = labels.long()

            TP += ((preds == 1) & (gt == 1)).sum().item()
            FP += ((preds == 1) & (gt == 0)).sum().item()
            FN += ((preds == 0) & (gt == 1)).sum().item()
            TN += ((preds == 0) & (gt == 0)).sum().item()

            postfix = OrderedDict(
                [
                    ("loss", avg_meters["loss"].avg),
                    ("iou", avg_meters["iou"].avg),
                    ("dice", avg_meters["dice"].avg),
                ]
            )
            pbar.set_postfix(postfix)
            pbar.update(1)
        pbar.close()

    recall_bad = TP / (TP + FN + 1e-8)
    precision_bad = TP / (TP + FP + 1e-8)
    accuracy = (TP + TN) / (TP + TN + FP + FN + 1e-8)

    return OrderedDict(
        [
            ("loss", avg_meters["loss"].avg),
            ("seg_loss", avg_meters["seg_loss"].avg),
            ("cls_loss", avg_meters["cls_loss"].avg),
            ("iou", avg_meters["iou"].avg),
            ("dice", avg_meters["dice"].avg),
            ("recall_bad", recall_bad),
            ("precision_bad", precision_bad),
            ("accuracy", accuracy),
        ]
    )


def main():
    # clone the dict so we don't accidentally modify the global CONFIG
    config = dict(CONFIG)

    if config["deep_supervision"]:
        raise NotImplementedError(
            "Deep supervision is not supported in this multi-task UNeXt setup. "
            "Please set deep_supervision = False in CONFIG."
        )

    if config["mask_root"] is None:
        config["mask_root"] = os.path.join(config["data_root"], "masks")

    if config["name"] is None:
        config["name"] = "%s_%s_multitask" % ("scratch", config["arch"])

    os.makedirs("models/%s" % config["name"], exist_ok=True)

    print("-" * 20)
    for key in config:
        print(f"{key}: {config[key]}")
    print("-" * 20)

    # save config to disk
    with open("models/%s/config.yml" % config["name"], "w") as f:
        yaml.dump(config, f)

    # define loss functions
    if config["loss"] == "BCEWithLogitsLoss":
        seg_criterion = nn.BCEWithLogitsLoss().cuda()
    else:
        seg_criterion = losses.__dict__[config["loss"]]().cuda()

    cls_criterion = nn.BCEWithLogitsLoss().cuda()

    cudnn.benchmark = True

    # create model
    model = archs.__dict__[config["arch"]](
        config["num_classes"], config["input_channels"], config["deep_supervision"]
    )
    model = model.cuda()

    # optimizer
    params = filter(lambda p: p.requires_grad, model.parameters())
    if config["optimizer"] == "Adam":
        optimizer = optim.Adam(
            params, lr=config["lr"], weight_decay=config["weight_decay"]
        )
    elif config["optimizer"] == "SGD":
        optimizer = optim.SGD(
            params,
            lr=config["lr"],
            momentum=config["momentum"],
            nesterov=config["nesterov"],
            weight_decay=config["weight_decay"],
        )
    else:
        raise NotImplementedError

    # scheduler
    if config["scheduler"] == "CosineAnnealingLR":
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config["epochs"], eta_min=config["min_lr"]
        )
    elif config["scheduler"] == "ReduceLROnPlateau":
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=config["factor"],
            patience=config["patience"],
            verbose=1,
            min_lr=config["min_lr"],
        )
    elif config["scheduler"] == "MultiStepLR":
        scheduler = lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[int(e) for e in config["milestones"].split(",")],
            gamma=config["gamma"],
        )
    elif config["scheduler"] == "ConstantLR":
        scheduler = None
    else:
        raise NotImplementedError

    # Albumentations transforms (image + mask)
    train_transform = Compose(
        [
            RandomRotate90(),
            A.HorizontalFlip(p=0.5),  # FIXED: use albumentations.HorizontalFlip
            Resize(config["input_h"], config["input_w"]),
            transforms.Normalize(),
            ToTensorV2(),
        ]
    )

    val_transform = Compose(
        [
            Resize(config["input_h"], config["input_w"]),
            transforms.Normalize(),
            ToTensorV2(),
        ]
    )

    # Data loading with new Dataset
    train_dataset = Dataset(
        img_root=config["data_root"],
        mask_root=config["mask_root"],
        phase="train",
        transform=train_transform,
        image_size=(config["input_h"], config["input_w"]),
    )

    val_dataset = Dataset(
        img_root=config["data_root"],
        mask_root=config["mask_root"],
        phase="val",
        transform=val_transform,
        image_size=(config["input_h"], config["input_w"]),
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        drop_last=False,
    )

    # logging dict
    log = OrderedDict(
        [
            ("epoch", []),
            ("lr", []),
            ("loss", []),
            ("seg_loss", []),
            ("cls_loss", []),
            ("iou", []),
            ("val_loss", []),
            ("val_seg_loss", []),
            ("val_cls_loss", []),
            ("val_iou", []),
            ("val_dice", []),
            ("val_recall_bad", []),
            ("val_precision_bad", []),
            ("val_accuracy", []),
        ]
    )

    best_iou = 0.0
    trigger = 0
    for epoch in range(config["epochs"]):
        print("Epoch [%d/%d]" % (epoch + 1, config["epochs"]))

        # train for one epoch
        train_log = train(
            config, train_loader, model, seg_criterion, cls_criterion, optimizer
        )

        # evaluate on validation set
        val_log = validate(config, val_loader, model, seg_criterion, cls_criterion)

        if config["scheduler"] == "CosineAnnealingLR":
            scheduler.step()
        elif config["scheduler"] == "ReduceLROnPlateau":
            scheduler.step(val_log["loss"])

        print(
            "loss %.4f (seg %.4f, cls %.4f) - iou %.4f - "
            "val_loss %.4f (seg %.4f, cls %.4f) - val_iou %.4f - "
            "val_recall_bad %.4f - val_precision_bad %.4f"
            % (
                train_log["loss"],
                train_log["seg_loss"],
                train_log["cls_loss"],
                train_log["iou"],
                val_log["loss"],
                val_log["seg_loss"],
                val_log["cls_loss"],
                val_log["iou"],
                val_log["recall_bad"],
                val_log["precision_bad"],
            )
        )

        log["epoch"].append(epoch)
        log["lr"].append(config["lr"])
        log["loss"].append(train_log["loss"])
        log["seg_loss"].append(train_log["seg_loss"])
        log["cls_loss"].append(train_log["cls_loss"])
        log["iou"].append(train_log["iou"])
        log["val_loss"].append(val_log["loss"])
        log["val_seg_loss"].append(val_log["seg_loss"])
        log["val_cls_loss"].append(val_log["cls_loss"])
        log["val_iou"].append(val_log["iou"])
        log["val_dice"].append(val_log["dice"])
        log["val_recall_bad"].append(val_log["recall_bad"])
        log["val_precision_bad"].append(val_log["precision_bad"])
        log["val_accuracy"].append(val_log["accuracy"])

        pd.DataFrame(log).to_csv("models/%s/log.csv" % config["name"], index=False)

        trigger += 1

        if val_log["iou"] > best_iou:
            torch.save(
                model.state_dict(), "models/%s/model.pth" % config["name"]
            )
            best_iou = val_log["iou"]
            print("=> saved best model")
            trigger = 0

        # early stopping
        if config["early_stopping"] >= 0 and trigger >= config["early_stopping"]:
            print("=> early stopping")
            break

        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
