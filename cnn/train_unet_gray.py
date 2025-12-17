import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm

from dataset_puzzle_gray import PuzzleEdgesDatasetGray, BasicGrayTransform
from unet_gray import UNetGray


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0

    for images, masks in tqdm(loader, desc="Train", leave=False):
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    return running_loss / len(loader.dataset)


def eval_one_epoch(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for images, masks in tqdm(loader, desc="Val", leave=False):
            images = images.to(device)
            masks = masks.to(device)

            logits = model(images)
            loss = criterion(logits, masks)

            running_loss += loss.item() * images.size(0)

    return running_loss / len(loader.dataset)


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    transform = BasicGrayTransform(size=(512, 512))

    train_ds = PuzzleEdgesDatasetGray(root_dir=args.data_dir, split="train",
                                      transform=transform)
    val_ds = PuzzleEdgesDatasetGray(root_dir=args.data_dir, split="val",
                                    transform=transform)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = UNetGray(in_channels=1, out_channels=1)
    model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = Adam(model.parameters(), lr=args.lr)

    best_val_loss = float("inf")
    save_path = Path(args.out_model)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = eval_one_epoch(model, val_loader, criterion, device)

        print(f"  train loss: {train_loss:.4f}")
        print(f"  val   loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            print(f"  -> saved best model to {save_path}")

    print("Training finished. Best val loss:", best_val_loss)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--out-model", type=str, default="models/unet_puzzle_gray.pth")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    main(args)
