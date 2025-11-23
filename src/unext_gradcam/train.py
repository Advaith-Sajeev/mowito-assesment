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
from dataset import Dataset
from utils import AverageMeter
from archs import UNext  # for clarity


# =========================
# CONFIG AS A DICTIONARY
# =========================
CONFIG = {
    # ---- basic run info ----
    "name": "scratch_UNext_cls",   # folder under models/
    "epochs": 150,
    "batch_size": 32,

    # ---- model ----
    "arch": "UNext",
    "deep_supervision": False,
    "input_channels": 3,
    "num_classes": 1,          # seg output channels (still needed by UNeXt, but unused in loss)
    "input_w": 256,
    "input_h": 256,

    # ---- dataset ----
    # data_root should contain: train/good, train/bad, val/good, val/bad
    "data_root": "data",
    # mask_root should contain: train/good, train/bad, val/good, val/bad
    # If None, will default to data_root/masks
    "mask_root": None,

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
    "gamma": 2 / 3,            # for MultiStepLR
    "early_stopping": -1,      # -1 to disable

    # ---- dataloader ----
    "num_workers": 4,

    # ---- classification head ----
    "cls_threshold": 0.5,      # threshold on sigmoid(cls_logit) to classify bad (1)
    # optional: set >1.0 (e.g. 3.92) to upweight the minority class in BCE
    "pos_weight": 1.0,
}


def train(config, train_loader, model, criterion, optimizer):
    avg_meters = {
        "loss": AverageMeter(),
        "acc": AverageMeter(),
    }

    model.train()

    pbar = tqdm(total=len(train_loader))
    for inputs, _, labels, _ in train_loader:
        inputs = inputs.cuda()
        labels = labels.cuda().float()  # shape (B,)

        # forward
        seg_logits, cls_logit = model(inputs)  # we ignore seg_logits
        loss = criterion(cls_logit, labels)

        # predictions
        probs = torch.sigmoid(cls_logit)
        preds = (probs >= config["cls_threshold"]).long()
        gt = labels.long()
        correct = (preds == gt).sum().item()
        acc = correct / labels.size(0)

        # backward + optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # update meters
        avg_meters["loss"].update(loss.item(), inputs.size(0))
        avg_meters["acc"].update(acc, inputs.size(0))

        postfix = OrderedDict(
            [
                ("loss", avg_meters["loss"].avg),
                ("acc", avg_meters["acc"].avg),
            ]
        )
        pbar.set_postfix(postfix)
        pbar.update(1)
    pbar.close()

    return OrderedDict(
        [
            ("loss", avg_meters["loss"].avg),
            ("acc", avg_meters["acc"].avg),
        ]
    )


def validate(config, val_loader, model, criterion):
    avg_meters = {
        "loss": AverageMeter(),
        "acc": AverageMeter(),
    }

    # classification confusion counts for "bad" class = 1
    TP = FP = FN = TN = 0

    model.eval()

    with torch.no_grad():
        pbar = tqdm(total=len(val_loader))
        for inputs, _, labels, _ in val_loader:
            inputs = inputs.cuda()
            labels = labels.cuda().float()

            seg_logits, cls_logit = model(inputs)  # ignore seg_logits

            loss = criterion(cls_logit, labels)

            probs = torch.sigmoid(cls_logit)
            preds = (probs >= config["cls_threshold"]).long()
            gt = labels.long()

            correct = (preds == gt).sum().item()
            acc = correct / labels.size(0)

            avg_meters["loss"].update(loss.item(), inputs.size(0))
            avg_meters["acc"].update(acc, inputs.size(0))

            TP += ((preds == 1) & (gt == 1)).sum().item()
            FP += ((preds == 1) & (gt == 0)).sum().item()
            FN += ((preds == 0) & (gt == 1)).sum().item()
            TN += ((preds == 0) & (gt == 0)).sum().item()

            postfix = OrderedDict(
                [
                    ("loss", avg_meters["loss"].avg),
                    ("acc", avg_meters["acc"].avg),
                ]
            )
            pbar.set_postfix(postfix)
            pbar.update(1)
        pbar.close()

    recall_bad = TP / (TP + FN + 1e-8)
    precision_bad = TP / (TP + FP + 1e-8)
    accuracy = (TP + TN) / (TP + TN + FP + FN + 1e-8)
    f1_bad = 2 * precision_bad * recall_bad / (precision_bad + recall_bad + 1e-8)

    return OrderedDict(
        [
            ("loss", avg_meters["loss"].avg),
            ("acc", avg_meters["acc"].avg),
            ("recall_bad", recall_bad),
            ("precision_bad", precision_bad),
            ("accuracy", accuracy),
            ("f1_bad", f1_bad),
        ]
    )


def main():
    # clone the dict so we don't accidentally modify the global CONFIG
    config = dict(CONFIG)

    if config["deep_supervision"]:
        raise NotImplementedError(
            "Deep supervision is not supported in this classification setup. "
            "Please set deep_supervision = False in CONFIG."
        )

    if config["mask_root"] is None:
        config["mask_root"] = os.path.join(config["data_root"], "masks")

    if config["name"] is None:
        config["name"] = "%s_%s_cls" % ("scratch", config["arch"])

    os.makedirs("models/%s" % config["name"], exist_ok=True)

    print("-" * 20)
    for key in config:
        print(f"{key}: {config[key]}")
    print("-" * 20)

    # save config to disk
    with open("models/%s/config.yml" % config["name"], "w") as f:
        yaml.dump(config, f)

    # define classification loss (with optional pos_weight for imbalance)
    if config["pos_weight"] != 1.0:
        pos_w = torch.tensor([config["pos_weight"]], dtype=torch.float32).cuda()
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w).cuda()
    else:
        criterion = nn.BCEWithLogitsLoss().cuda()

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

    # Albumentations transforms (image only; mask ignored)
    train_transform = Compose(
        [
            RandomRotate90(),
            A.HorizontalFlip(p=0.5),
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

    # Data loading with Dataset (we ignore masks in loops)
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
            ("train_acc", []),
            ("val_loss", []),
            ("val_acc", []),
            ("val_recall_bad", []),
            ("val_precision_bad", []),
            ("val_f1_bad", []),
        ]
    )

    best_f1 = 0.0
    trigger = 0
    for epoch in range(config["epochs"]):
        print("Epoch [%d/%d]" % (epoch + 1, config["epochs"]))

        # train for one epoch
        train_log = train(config, train_loader, model, criterion, optimizer)

        # evaluate on validation set
        val_log = validate(config, val_loader, model, criterion)

        # step scheduler
        if config["scheduler"] == "CosineAnnealingLR":
            scheduler.step()
        elif config["scheduler"] == "ReduceLROnPlateau":
            scheduler.step(val_log["loss"])

        print(
            "loss %.4f - acc %.4f - "
            "val_loss %.4f - val_acc %.4f - "
            "val_recall_bad %.4f - val_precision_bad %.4f - val_f1_bad %.4f"
            % (
                train_log["loss"],
                train_log["acc"],
                val_log["loss"],
                val_log["acc"],
                val_log["recall_bad"],
                val_log["precision_bad"],
                val_log["f1_bad"],
            )
        )

        log["epoch"].append(epoch)
        log["lr"].append(config["lr"])
        log["loss"].append(train_log["loss"])
        log["train_acc"].append(train_log["acc"])
        log["val_loss"].append(val_log["loss"])
        log["val_acc"].append(val_log["acc"])
        log["val_recall_bad"].append(val_log["recall_bad"])
        log["val_precision_bad"].append(val_log["precision_bad"])
        log["val_f1_bad"].append(val_log["f1_bad"])

        pd.DataFrame(log).to_csv("models/%s/log.csv" % config["name"], index=False)

        trigger += 1

        # use F1 of bad class as checkpoint criterion
        if val_log["f1_bad"] > best_f1:
            torch.save(
                model.state_dict(), "models/%s/model.pth" % config["name"]
            )
            best_f1 = val_log["f1_bad"]
            print("=> saved best model (F1_bad = %.4f)" % best_f1)
            trigger = 0

        # early stopping
        if config["early_stopping"] >= 0 and trigger >= config["early_stopping"]:
            print("=> early stopping")
            break

        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
