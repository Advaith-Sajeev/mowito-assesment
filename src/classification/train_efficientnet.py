# efficientnet_train.py

import os
import copy
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

from sklearn.metrics import precision_score, recall_score


def get_dataloaders(data_root, img_size=224, batch_size=32, num_workers=4):
    train_dir = os.path.join(data_root, "train")
    val_dir   = os.path.join(data_root, "val")

    train_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.RandomRotation(2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    val_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = datasets.ImageFolder(train_dir, transform=train_tf)
    val_dataset   = datasets.ImageFolder(val_dir,   transform=val_tf)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, train_dataset, val_dataset


def build_efficientnet(num_classes=2, pretrained=True):
    if pretrained:
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1
        model = efficientnet_b0(weights=weights)
    else:
        model = efficientnet_b0(weights=None)

    # Replace classifier head
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


def train_efficientnet(
    data_root="data",
    num_epochs=15,
    batch_size=32,
    lr=1e-4,
    img_size=224,
    num_workers=4,
    save_path="best_efficientnet_b0.pth"
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader, train_dataset, _ = get_dataloaders(
        data_root=data_root,
        img_size=img_size,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    print("class_to_idx:", train_dataset.class_to_idx)
    bad_class_idx = train_dataset.class_to_idx["bad"]

    model = build_efficientnet(num_classes=2, pretrained=True).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_recall_bad = 0.0

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 30)

        # ---------- TRAIN ----------
        model.train()
        running_loss = 0.0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Train loss: {epoch_loss:.4f}")

        # ---------- VALIDATION ----------
        model.eval()
        val_loss = 0.0
        all_true = []
        all_pred = []

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)

                preds = torch.argmax(outputs, dim=1)

                all_true.append(labels.cpu())
                all_pred.append(preds.cpu())

        val_loss = val_loss / len(val_loader.dataset)
        all_true = torch.cat(all_true).numpy()
        all_pred = torch.cat(all_pred).numpy()

        # binary metrics for "bad" class
        y_true_bin = (all_true == bad_class_idx).astype(int)
        y_pred_bin = (all_pred == bad_class_idx).astype(int)

        recall_bad = recall_score(y_true_bin, y_pred_bin, pos_label=1)
        precision_bad = precision_score(y_true_bin, y_pred_bin, pos_label=1)
        acc = (all_true == all_pred).mean()

        print(f"Val loss: {val_loss:.4f}")
        print(f"Val accuracy:      {acc:.4f}")
        print(f"Val BAD precision: {precision_bad:.4f}")
        print(f"Val BAD recall:    {recall_bad:.4f}")

        # save best model by recall on bad
        if recall_bad > best_recall_bad:
            best_recall_bad = recall_bad
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(best_model_wts, save_path)
            print(f"--> New best model saved to {save_path} (recall_bad={best_recall_bad:.4f})")

    print("\nTraining complete.")
    print(f"Best BAD recall on val: {best_recall_bad:.4f}")
    model.load_state_dict(best_model_wts)
    return model


if __name__ == "__main__":
    trained_model = train_efficientnet(
        data_root="data",
        num_epochs=50,
        batch_size=64,
        lr=1e-4,
        img_size=224,
        num_workers=4,
        save_path="best_efficientnet_b0.pth",
    )
