# train_scratch_pix2pix.py

import argparse
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from scratch_pix2pix_dataset import ScratchPix2PixDataset
from pix2pix_models import Pix2PixGenerator, PatchDiscriminator


def parse_args():
    parser = argparse.ArgumentParser(description="Train pix2pix GAN for scratch synthesis")
    parser.add_argument(
        "--datafolder",
        required=True,
        help="Root folder containing bad/ and masks/ (e.g. anomaly_detection_test_data)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size",
    )
    parser.add_argument(
        "--patch_size",
        type=int,
        default=128,
        help="Patch size (H=W) to train on",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-4,
        help="Learning rate for G and D",
    )
    parser.add_argument(
        "--beta1",
        type=float,
        default=0.5,
        help="Adam beta1",
    )
    parser.add_argument(
        "--lambda_L1",
        type=float,
        default=100.0,
        help="Weight for L1 reconstruction loss",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="scratch_pix2pix_ckpt",
        help="Folder to save checkpoints",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.outdir, exist_ok=True)

    # ------------------------
    # Dataset & DataLoader
    # ------------------------
    bad_dir = os.path.join(args.datafolder, "bad")
    mask_dir = os.path.join(args.datafolder, "masks")

    dataset = ScratchPix2PixDataset(
        bad_dir=bad_dir,
        mask_dir=mask_dir,
        patch_size=args.patch_size,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        drop_last=True,
    )

    # ------------------------
    # Models
    # ------------------------
    G = Pix2PixGenerator(in_channels=2, out_channels=1).to(device)
    D = PatchDiscriminator(in_channels=3).to(device)

    criterion_GAN = nn.BCEWithLogitsLoss()
    criterion_L1 = nn.L1Loss()

    opt_G = optim.Adam(G.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    opt_D = optim.Adam(D.parameters(), lr=args.lr, betas=(args.beta1, 0.999))

    best_G_loss = float("inf")

    print("Starting pix2pix training")
    print(f"Device: {device}")
    print(f"Dataset size: {len(dataset)} patches")

    for epoch in range(1, args.epochs + 1):
        G.train()
        D.train()

        running_G = 0.0
        running_D = 0.0
        count = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{args.epochs}")
        for cond, real in pbar:
            # cond: (B,2,H,W), real: (B,1,H,W)
            cond = cond.to(device)
            real = real.to(device)

            B = cond.size(0)

            # -------------------------------
            # 1) Train Discriminator
            # -------------------------------
            opt_D.zero_grad()

            # Real pair
            real_pair = torch.cat([cond, real], dim=1)  # (B,3,H,W)
            pred_real = D(real_pair)
            label_real = torch.ones_like(pred_real, device=device)
            loss_D_real = criterion_GAN(pred_real, label_real)

            # Fake pair
            fake = G(cond)
            fake_pair = torch.cat([cond, fake.detach()], dim=1)
            pred_fake = D(fake_pair)
            label_fake = torch.zeros_like(pred_fake, device=device)
            loss_D_fake = criterion_GAN(pred_fake, label_fake)

            loss_D = (loss_D_real + loss_D_fake) * 0.5
            loss_D.backward()
            opt_D.step()

            # -------------------------------
            # 2) Train Generator
            # -------------------------------
            opt_G.zero_grad()

            fake = G(cond)
            fake_pair = torch.cat([cond, fake], dim=1)
            pred_fake_for_G = D(fake_pair)
            label_real_for_G = torch.ones_like(pred_fake_for_G, device=device)

            loss_G_GAN = criterion_GAN(pred_fake_for_G, label_real_for_G)
            loss_G_L1 = criterion_L1(fake, real) * args.lambda_L1

            loss_G = loss_G_GAN + loss_G_L1
            loss_G.backward()
            opt_G.step()

            running_G += loss_G.item()
            running_D += loss_D.item()
            count += 1

            pbar.set_postfix(
                {
                    "loss_D": f"{loss_D.item():.4f}",
                    "loss_G": f"{loss_G.item():.4f}",
                }
            )

        avg_G = running_G / max(1, count)
        avg_D = running_D / max(1, count)
        print(f"Epoch {epoch}: avg_D_loss={avg_D:.4f}, avg_G_loss={avg_G:.4f}")

        # Save "best" G by lowest avg_G
        if avg_G < best_G_loss:
            best_G_loss = avg_G
            best_G_path = os.path.join(args.outdir, "generator_best.pth")
            best_D_path = os.path.join(args.outdir, "discriminator_best.pth")
            torch.save(G.state_dict(), best_G_path)
            torch.save(D.state_dict(), best_D_path)
            print(
                f"=> New best G (avg_G_loss={avg_G:.4f}) at epoch {epoch}. "
                f"Saved to {best_G_path} and {best_D_path}"
            )

        # Also save periodic checkpoints
        if epoch % 10 == 0 or epoch == args.epochs:
            ckpt_G = os.path.join(args.outdir, f"generator_epoch_{epoch:03d}.pth")
            ckpt_D = os.path.join(args.outdir, f"discriminator_epoch_{epoch:03d}.pth")
            torch.save(G.state_dict(), ckpt_G)
            torch.save(D.state_dict(), ckpt_D)
            print(f"Saved checkpoints at epoch {epoch} to {args.outdir}")

    print("Training finished.")


if __name__ == "__main__":
    main()
