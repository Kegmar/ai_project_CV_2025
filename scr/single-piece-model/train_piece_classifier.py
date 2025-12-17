import argparse
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import timm

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

def accuracy_top1(logits, y):
    return (logits.argmax(1) == y).float().mean().item()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="single_piece_dataset")
    ap.add_argument("--imgsz", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--model", type=str, default="mobilenetv3_small_100")
    ap.add_argument("--out", type=str, default="runs/piece_cls")
    args = ap.parse_args()

    data_dir = Path(args.data)
    train_dir = data_dir / "train"
    val_dir   = data_dir / "val"
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Augmentations (small dataset -> help generalization)
    train_tf = transforms.Compose([
        transforms.Resize((args.imgsz, args.imgsz)),
        transforms.RandomRotation(180),
        transforms.RandomPerspective(distortion_scale=0.25, p=0.4),
        transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.15, hue=0.02),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    val_tf = transforms.Compose([
        transforms.Resize((args.imgsz, args.imgsz)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    train_ds = datasets.ImageFolder(str(train_dir), transform=train_tf)
    val_ds   = datasets.ImageFolder(str(val_dir),   transform=val_tf)

    # Save class mapping (index -> folder name)
    # ImageFolder sorts class names, so this will match training labels.
    classes = train_ds.classes
    (out_dir / "classes.txt").write_text("\n".join(classes) + "\n", encoding="utf-8")
    print("[INFO] num_classes:", len(classes))
    print("[INFO] saved:", (out_dir / "classes.txt").resolve())

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                            num_workers=4, pin_memory=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("[INFO] device:", device)

    model = timm.create_model(args.model, pretrained=True, num_classes=len(classes))
    model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best_val = 0.0
    best_path = out_dir / "best.pth"

    for epoch in range(1, args.epochs + 1):
        model.train()
        tr_loss = 0.0
        tr_acc = 0.0

        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            opt.step()

            tr_loss += loss.item()
            tr_acc += accuracy_top1(logits.detach(), y)

        sched.step()

        model.eval()
        va_acc = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                logits = model(x)
                va_acc += accuracy_top1(logits, y)

        tr_loss /= max(1, len(train_loader))
        tr_acc  /= max(1, len(train_loader))
        va_acc  /= max(1, len(val_loader))

        print(f"[E{epoch:03d}] train_loss={tr_loss:.4f} train_acc={tr_acc:.3f} val_acc={va_acc:.3f}")

        if va_acc > best_val:
            best_val = va_acc
            torch.save({"model": model.state_dict(), "classes": classes, "imgsz": args.imgsz}, best_path)
            print(f"  [SAVE] best -> {best_path} (val_acc={best_val:.3f})")

    # --- Export ONNX (static input) ---
    onnx_path = out_dir / "piece_cls.onnx"
    model.eval().cpu()
    dummy = torch.zeros(1, 3, args.imgsz, args.imgsz, dtype=torch.float32)
    torch.onnx.export(
        model, dummy, str(onnx_path),
        input_names=["images"],
        output_names=["logits"],
        opset_version=13,
        do_constant_folding=True
    )
    print("[DONE] Exported:", onnx_path.resolve())
    print("[DONE] Best ckpt:", best_path.resolve(), "best_val_acc=", best_val)

if __name__ == "__main__":
    main()
