"""
RoadRishi — SegFormer Fine-Tuning Script (Kaggle-Ready)
========================================================
Run this on Kaggle with the DeepGlobe Road Extraction dataset attached.
Dataset: https://www.kaggle.com/datasets/balraj98/deepglobe-road-extraction-dataset

Instructions:
1. On Kaggle, click "Copy & Edit" on the DeepGlobe dataset page
2. Paste this entire script into a single code cell (or upload as .py)
3. Enable GPU: Settings → Accelerator → GPU T4 x2
4. Click "Run All"
5. Download roadrishi_finetuned.pth from the output panel when done

Expected runtime: ~3-5 hours on free T4 GPU
"""

# ── Cell 1: Install extra dependencies ────────────────────────────────────────
import subprocess
subprocess.run(["pip", "install", "-q", "transformers>=4.37", "albumentations>=1.4", "evaluate"], check=True)

# ── Cell 2: Imports ────────────────────────────────────────────────────────────
import os
import json
import random
import numpy as np
from pathlib import Path
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    SegformerForSemanticSegmentation,
    SegformerImageProcessor,
)
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ── Cell 3: Configuration ──────────────────────────────────────────────────────
CFG = {
    # Paths — Kaggle mounts DeepGlobe here
    "data_root":       "/kaggle/input/deepglobe-road-extraction-dataset/train",
    "output_dir":      "/kaggle/working/checkpoints",

    # Model
    "model_name":      "nvidia/segformer-b3-finetuned-ade-512-512",
    "num_labels":      2,          # 0=background, 1=road

    # Training — tuned for Kaggle T4 x2 (2x 16 GB VRAM)
    "img_size":        512,
    "batch_size":      8,          # 8 across 2x T4 = 4 per GPU
    "grad_accum":      1,          # no accumulation needed with T4 x2
    "num_epochs":      15,         # ~3-4 hrs on T4 x2
    "lr":              5e-5,
    "weight_decay":    1e-4,
    "val_split":       0.1,
    "seed":            42,

    # Loss weights
    "bce_weight":      1.0,
    "dice_weight":     1.0,
}

os.makedirs(CFG["output_dir"], exist_ok=True)
random.seed(CFG["seed"])
np.random.seed(CFG["seed"])
torch.manual_seed(CFG["seed"])

# ── Cell 4: Augmentation pipelines ────────────────────────────────────────────
train_transform = A.Compose([
    A.RandomResizedCrop(CFG["img_size"], CFG["img_size"], scale=(0.5, 1.0)),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.3),
    A.RandomRotate90(p=0.3),
    A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1, p=0.5),
    A.GaussNoise(var_limit=(10, 50), p=0.3),
    A.GaussianBlur(blur_limit=3, p=0.2),
    A.ElasticTransform(alpha=50, sigma=5, p=0.2),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

val_transform = A.Compose([
    A.Resize(CFG["img_size"], CFG["img_size"]),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

# ── Cell 5: Dataset class ──────────────────────────────────────────────────────
class DeepGlobeDataset(Dataset):
    """
    DeepGlobe Road Extraction Dataset loader.
    Expects folder structure:
      data_root/
        <id>_sat.jpg   — RGB satellite image
        <id>_mask.png  — Binary road mask (white=road, black=background)
    """
    def __init__(self, image_paths, mask_paths, transform=None):
        self.image_paths = image_paths
        self.mask_paths  = mask_paths
        self.transform   = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = np.array(Image.open(self.image_paths[idx]).convert("RGB"))
        mask  = np.array(Image.open(self.mask_paths[idx]).convert("L"))

        # Binarize: white pixels (>127) = road (class 1)
        mask = (mask > 127).astype(np.uint8)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]   # (C, H, W) float tensor
            mask  = augmented["mask"]    # (H, W) uint8 tensor

        return {
            "pixel_values": image.float(),
            "labels":       mask.long(),
        }


def build_dataloaders(data_root, cfg):
    data_root = Path(data_root)
    sat_files  = sorted(data_root.glob("*_sat.jpg"))
    mask_files = [data_root / f.name.replace("_sat.jpg", "_mask.png") for f in sat_files]

    # Filter pairs where mask actually exists
    pairs = [(s, m) for s, m in zip(sat_files, mask_files) if m.exists()]
    print(f"Found {len(pairs)} valid image-mask pairs")

    random.shuffle(pairs)
    n_val = max(1, int(len(pairs) * cfg["val_split"]))
    train_pairs = pairs[n_val:]
    val_pairs   = pairs[:n_val]

    train_ds = DeepGlobeDataset(
        [p[0] for p in train_pairs],
        [p[1] for p in train_pairs],
        transform=train_transform,
    )
    val_ds = DeepGlobeDataset(
        [p[0] for p in val_pairs],
        [p[1] for p in val_pairs],
        transform=val_transform,
    )

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=cfg["batch_size"], shuffle=False, num_workers=2, pin_memory=True)

    print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")
    return train_loader, val_loader


# ── Cell 6: Loss functions ─────────────────────────────────────────────────────
class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        # logits: (B, C, H, W), targets: (B, H, W)
        probs = torch.softmax(logits, dim=1)[:, 1]  # road class probability
        targets_f = targets.float()
        intersection = (probs * targets_f).sum(dim=(1, 2))
        union        = probs.sum(dim=(1, 2)) + targets_f.sum(dim=(1, 2))
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()


class CombinedLoss(nn.Module):
    def __init__(self, bce_w=1.0, dice_w=1.0):
        super().__init__()
        self.bce  = nn.CrossEntropyLoss()
        self.dice = DiceLoss()
        self.bce_w  = bce_w
        self.dice_w = dice_w

    def forward(self, logits, targets):
        return self.bce_w * self.bce(logits, targets) + self.dice_w * self.dice(logits, targets)


# ── Cell 7: IoU metric ─────────────────────────────────────────────────────────
def compute_miou(preds, targets, num_classes=2):
    """Compute mean IoU over a batch."""
    preds   = preds.view(-1)
    targets = targets.view(-1)
    iou_per_class = []
    for cls in range(num_classes):
        pred_mask   = (preds == cls)
        target_mask = (targets == cls)
        intersection = (pred_mask & target_mask).sum().float()
        union        = (pred_mask | target_mask).sum().float()
        if union == 0:
            iou_per_class.append(torch.tensor(1.0))
        else:
            iou_per_class.append(intersection / union)
    return torch.stack(iou_per_class).mean().item()


# ── Cell 8: Model setup ────────────────────────────────────────────────────────
def build_model(cfg):
    print(f"Loading base model: {cfg['model_name']}")
    model = SegformerForSemanticSegmentation.from_pretrained(
        cfg["model_name"],
        num_labels=cfg["num_labels"],
        id2label={0: "background", 1: "road"},
        label2id={"background": 0, "road": 1},
        ignore_mismatched_sizes=True,
    )
    # Use both GPUs if available
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs via DataParallel")
        model = nn.DataParallel(model)
    model = model.to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,}")
    return model


# ── Cell 9: Training loop ──────────────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, criterion, epoch):
    model.train()
    total_loss = 0.0
    total_miou = 0.0

    pbar = tqdm(loader, desc=f"Epoch {epoch} [train]", leave=False)
    grad_accum = CFG.get("grad_accum", 1)
    optimizer.zero_grad()

    for step, batch in enumerate(pbar):
        pixel_values = batch["pixel_values"].to(DEVICE)
        labels       = batch["labels"].to(DEVICE)

        outputs = model(pixel_values=pixel_values)
        # Handle DataParallel wrapper
        logits = outputs.logits if hasattr(outputs, "logits") else outputs["logits"]

        upsampled = nn.functional.interpolate(
            logits, size=labels.shape[-2:], mode="bilinear", align_corners=False
        )

        loss = criterion(upsampled, labels) / grad_accum
        loss.backward()

        if (step + 1) % grad_accum == 0 or (step + 1) == len(loader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

        loss_val = loss.item() * grad_accum

        preds = upsampled.argmax(dim=1)
        miou  = compute_miou(preds.cpu(), labels.cpu())

        total_loss += loss_val
        total_miou += miou
        pbar.set_postfix(loss=f"{loss_val:.4f}", miou=f"{miou:.3f}")

    n = len(loader)
    return total_loss / n, total_miou / n


@torch.no_grad()
def evaluate(model, loader, criterion, epoch):
    model.eval()
    total_loss = 0.0
    total_miou = 0.0

    for batch in tqdm(loader, desc=f"Epoch {epoch} [val]  ", leave=False):
        pixel_values = batch["pixel_values"].to(DEVICE)
        labels       = batch["labels"].to(DEVICE)

        outputs  = model(pixel_values=pixel_values)
        upsampled = nn.functional.interpolate(
            outputs.logits, size=labels.shape[-2:], mode="bilinear", align_corners=False
        )
        loss  = criterion(upsampled, labels)
        preds = upsampled.argmax(dim=1)
        miou  = compute_miou(preds.cpu(), labels.cpu())

        total_loss += loss.item()
        total_miou += miou

    n = len(loader)
    return total_loss / n, total_miou / n


# ── Cell 10: Main training run ─────────────────────────────────────────────────
def main():
    train_loader, val_loader = build_dataloaders(CFG["data_root"], CFG)
    model     = build_model(CFG)
    criterion = CombinedLoss(bce_w=CFG["bce_weight"], dice_w=CFG["dice_weight"])
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=CFG["lr"], weight_decay=CFG["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=CFG["num_epochs"], eta_min=1e-6
    )

    best_miou = 0.0
    log = []

    print("\n" + "=" * 60)
    print("  Starting RoadRishi SegFormer Fine-Tuning")
    print(f"  Epochs: {CFG['num_epochs']}  |  LR: {CFG['lr']}  |  Batch: {CFG['batch_size']}")
    print("=" * 60)

    for epoch in range(1, CFG["num_epochs"] + 1):
        train_loss, train_miou = train_one_epoch(model, train_loader, optimizer, criterion, epoch)
        val_loss,   val_miou   = evaluate(model, val_loader, criterion, epoch)
        scheduler.step()

        row = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "train_miou": round(train_miou, 4),
            "val_loss":   round(val_loss,   4),
            "val_miou":   round(val_miou,   4),
            "lr":         round(scheduler.get_last_lr()[0], 8),
        }
        log.append(row)

        print(
            f"Epoch {epoch:2d}/{CFG['num_epochs']} | "
            f"Train Loss: {train_loss:.4f}  mIoU: {train_miou:.4f} | "
            f"Val Loss: {val_loss:.4f}  mIoU: {val_miou:.4f}"
        )

        # Save best checkpoint
        if val_miou > best_miou:
            best_miou = val_miou
            ckpt_path = os.path.join(CFG["output_dir"], "roadrishi_finetuned.pth")
            torch.save({
                "epoch":      epoch,
                "model_state_dict": model.state_dict(),
                "val_miou":   val_miou,
                "val_loss":   val_loss,
                "config":     CFG,
            }, ckpt_path)
            print(f"  ✅ Best checkpoint saved (val_mIoU={val_miou:.4f})")

    # Save training log
    log_path = os.path.join(CFG["output_dir"], "training_log.json")
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)

    print("\n" + "=" * 60)
    print(f"  Training complete! Best val mIoU: {best_miou:.4f}")
    print(f"  Checkpoint: {CFG['output_dir']}/roadrishi_finetuned.pth")
    print(f"  Log:        {log_path}")
    print("=" * 60)
    print("\nNext step: Download roadrishi_finetuned.pth from the output panel")
    print("Place it at: training/checkpoints/roadrishi_finetuned.pth")


if __name__ == "__main__":
    main()
