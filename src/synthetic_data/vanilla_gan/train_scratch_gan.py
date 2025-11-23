import argparse
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from scratch_patch_dataset import ScratchPatchDataset
from gan_models import ScratchGenerator, ScratchDiscriminator


def parse_args():
    parser = argparse.ArgumentParser(description="Train scratch GAN on mask patches only")
    parser.add_argument(
        "--datafolder",
        required=True,
        help="Root folder containing bad/ and masks/ (e.g. anomaly_detection_test_data)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size",
    )
    parser.add_argument(
        "--patch_size",
        type=int,
        default=64,
        help="Square patch size (HxW) to train on",
    )
    parser.add_argument(
        "--z_dim",
        type=int,
        default=128,
        help="Latent dimension for generator input",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-4,
        help="Learning rate for both G and D",
    )
    parser.add_argument(
        "--beta1",
        type=float,
        default=0.5,
        help="Adam beta1",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="scratch_gan_ckpt",
        help="Folder to save checkpoints",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.outdir, exist_ok=True)

    # ----------------------------------------------------
    # Dataset: returns mask patches of shape (1, H, W) in [0,1]
    # ----------------------------------------------------
    bad_dir = os.path.join(args.datafolder, "bad")
    mask_dir = os.path.join(args.datafolder, "masks")

    dataset = ScratchPatchDataset(
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

    # ----------------------------------------------------
    # Models: mask-only GAN
    #   G: z -> mask_logits (B,1,H,W)
    #   D: mask (B,1,H,W) -> (B,) logits
    # ----------------------------------------------------
    G = ScratchGenerator(z_dim=args.z_dim).to(device)
    D = ScratchDiscriminator(in_channels=1).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optim_G = optim.Adam(G.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    optim_D = optim.Adam(D.parameters(), lr=args.lr, betas=(args.beta1, 0.999))

    print("Starting mask-only GAN training")
    print(f"Device: {device}")
    print(f"Dataset size: {len(dataset)} patches")

    best_G_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        G.train()
        D.train()

        sum_D_loss = 0.0
        sum_G_loss = 0.0
        n_batches = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{args.epochs}")
        for batch in pbar:
            # batch: [B, 1, H, W] (mask in [0,1])
            batch = batch.to(device)
            real_masks = batch
            B = real_masks.size(0)

            # ------------------------------------------------
            # 1) Train Discriminator
            # ------------------------------------------------
            optim_D.zero_grad()

            # Real
            out_real = D(real_masks)  # (B,)
            real_labels = torch.ones_like(out_real, device=device)
            loss_real = criterion(out_real, real_labels)

            # Fake
            z = torch.randn(B, args.z_dim, device=device)  # (B, z_dim)
            mask_logits_fake = G(z)                        # (B,1,H,W)
            fake_masks = torch.sigmoid(mask_logits_fake)   # (B,1,H,W) in [0,1]

            out_fake = D(fake_masks.detach())              # (B,)
            fake_labels = torch.zeros_like(out_fake, device=device)
            loss_fake = criterion(out_fake, fake_labels)

            loss_D = loss_real + loss_fake
            loss_D.backward()
            optim_D.step()

            # ------------------------------------------------
            # 2) Train Generator
            # ------------------------------------------------
            optim_G.zero_grad()

            z = torch.randn(B, args.z_dim, device=device)
            mask_logits_fake = G(z)
            fake_masks = torch.sigmoid(mask_logits_fake)

            out_fake_for_G = D(fake_masks)
            target_for_G = torch.ones_like(out_fake_for_G, device=device)
            loss_G = criterion(out_fake_for_G, target_for_G)

            loss_G.backward()
            optim_G.step()

            sum_D_loss += loss_D.item()
            sum_G_loss += loss_G.item()
            n_batches += 1

            pbar.set_postfix(
                {
                    "loss_D": f"{loss_D.item():.4f}",
                    "loss_G": f"{loss_G.item():.4f}",
                }
            )

        # epoch-level averages
        avg_D_loss = sum_D_loss / n_batches
        avg_G_loss = sum_G_loss / n_batches
        print(f"Epoch {epoch}: avg_D_loss={avg_D_loss:.4f}, avg_G_loss={avg_G_loss:.4f}")

        # save best generator by lowest avg_G_loss
        if avg_G_loss < best_G_loss:
            best_G_loss = avg_G_loss
            best_G_path = os.path.join(args.outdir, "generator_best.pth")
            best_D_path = os.path.join(args.outdir, "discriminator_best.pth")
            torch.save(G.state_dict(), best_G_path)
            torch.save(D.state_dict(), best_D_path)
            print(
                f"=> New best G (avg_G_loss={avg_G_loss:.4f}) at epoch {epoch}. "
                f"Saved to {best_G_path} and {best_D_path}"
            )

        # also keep periodic checkpoints if you like
        if epoch % 10 == 0 or epoch == args.epochs:
            ckpt_path_G = os.path.join(args.outdir, f"generator_epoch_{epoch:03d}.pth")
            ckpt_path_D = os.path.join(args.outdir, f"discriminator_epoch_{epoch:03d}.pth")
            torch.save(G.state_dict(), ckpt_path_G)
            torch.save(D.state_dict(), ckpt_path_D)
            print(f"Saved checkpoints at epoch {epoch} to {args.outdir}")

    print("Training finished.")


if __name__ == "__main__":
    main()
