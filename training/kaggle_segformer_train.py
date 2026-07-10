"""
RoadRishi — SegFormer Fine-Tuning Script (Kaggle-Ready, Crash-Proof)
=====================================================================
Run this on Kaggle with the DeepGlobe Road Extraction dataset attached.
Dataset: https://www.kaggle.com/datasets/balraj98/deepglobe-road-extraction-dataset

Instructions:
1. On Kaggle, click "Copy & Edit" on the DeepGlobe dataset page
2. Paste this entire script into a single code cell (or upload as .py)
3. Enable GPU: Settings → Accelerator → GPU T4 x2
4. Click "Run All"
5. Download roadrishi_finetuned.pth from the output panel when done

Expected runtime: ~2.5–3 hours on free T4 x2 GPU
  ▸ img_size=384, batch_size=4, grad_accum=2, epochs=8, AMP enabled

Crash defences in this script:
  ✅ CUDA OOM mid-batch → skips batch, clears cache, continues training
  ✅ NaN loss detection → skips corrupt step, logs warning
  ✅ Periodic checkpoint every 2 epochs (not just best) → survives crashes
  ✅ persistent_workers=False → no Kaggle Docker deadlocks
  ✅ VRAM headroom check before training starts
  ✅ DataParallel logits guard in ALL forward paths
  ✅ AMP GradScaler health logging
"""

# ── Cell 1: Verify pre-installed dependencies ──────────────────────────────────
import importlib, sys

_MIN_VERSIONS = {
    "transformers":   (4, 37),
    "albumentations": (1, 4),
}

for pkg, (maj, min_) in _MIN_VERSIONS.items():
    try:
        mod = importlib.import_module(pkg)
        ver_str = getattr(mod, "__version__", "unknown")
        parts = [int(x) for x in ver_str.split(".")[:2] if x.isdigit()]
        ok = tuple(parts) >= (maj, min_)
        status = "✅" if ok else "⚠️  TOO OLD"
        print(f"{status}  {pkg} {ver_str}  (need >={maj}.{min_})")
        if not ok:
            print(f"   → Enable Kaggle Internet (Settings → Internet → ON) and re-run.")
    except ImportError:
        print(f"❌  {pkg} not found — enable Kaggle Internet to install it.")
        sys.exit(1)

# ── Cell 2: Imports ────────────────────────────────────────────────────────────
import os
import json
import math
import time
import random
import warnings
import numpy as np
from pathlib import Path
from PIL import Image

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader
from transformers import SegformerForSemanticSegmentation
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning)  # suppress albumentations noise

print(f"PyTorch:  {torch.__version__}")
print(f"CUDA:     {torch.cuda.is_available()}")
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print(f"  GPU {i}: {p.name}  ({p.total_memory // 1024**2} MB VRAM)")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device:   {DEVICE}")

# ── Cell 3: Configuration ──────────────────────────────────────────────────────
def _find_deepglobe_root() -> str:
    """Auto-detect DeepGlobe train split inside /kaggle/input."""
    import glob as _glob
    mask_candidates = _glob.glob("/kaggle/input/**/*_mask.png", recursive=True)
    if mask_candidates:
        dirs = list({str(Path(p).parent) for p in mask_candidates})
        train_dirs = [d for d in dirs if Path(d).name == "train"]
        chosen = train_dirs[0] if train_dirs else dirs[0]
        n = len([p for p in mask_candidates if str(Path(p).parent) == chosen])
        print(f"[AUTO-DETECT] DeepGlobe root: {chosen}  ({n} mask files)")
        return chosen
    # Fallback with diagnostic output
    sat_candidates = _glob.glob("/kaggle/input/**/*_sat.jpg", recursive=True)
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        print("[WARN] /kaggle/input contents:")
        for p in sorted(kaggle_input.iterdir()):
            print(f"  {p}")
    if sat_candidates:
        print(f"[WARN] sat images found but NO masks — likely wrong split.")
    fallback = "/kaggle/input/deepglobe-road-extraction-dataset/train"
    print(f"[WARN] Falling back to: {fallback}")
    return fallback


CFG = {
    # Paths
    "data_root":    _find_deepglobe_root(),
    "output_dir":   "/kaggle/working/checkpoints",

    # Model
    "model_name":   "nvidia/segformer-b3-finetuned-ade-512-512",
    "num_labels":   2,          # 0=background, 1=road

    # ── Speed/stability tuned for ~3 h on T4 x2 ───────────────────────────────
    #   img_size=384 → 44% fewer pixels vs 512 (biggest speedup)
    #   batch_size=4 → 4 per GPU; grad_accum=2 → effective batch=8
    #   num_epochs=8 → ~20 min/epoch × 8 = ~2.7 h
    #   AMP fp16     → ~35% faster, ~40% less VRAM
    "img_size":     384,
    "batch_size":   4,
    "grad_accum":   2,
    "num_epochs":   8,
    "lr":           5e-5,
    "weight_decay": 1e-4,
    "val_split":    0.1,
    "num_workers":  0,          # 0 = main process only; eliminates Kaggle Docker worker deadlocks
    "seed":         42,

    # Loss
    "bce_weight":   1.0,
    "dice_weight":  1.0,

    # AMP — flip to False only if you see persistent NaN losses
    "use_amp":      True,

    # Save a checkpoint every N epochs (in addition to best)
    "save_every":   2,
}

os.makedirs(CFG["output_dir"], exist_ok=True)
random.seed(CFG["seed"])
np.random.seed(CFG["seed"])
torch.manual_seed(CFG["seed"])
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(CFG["seed"])

print(f"\n{'='*60}")
print(f"  Config: img={CFG['img_size']}  bs={CFG['batch_size']}  "
      f"accum={CFG['grad_accum']}  eff_bs={CFG['batch_size']*CFG['grad_accum']}  "
      f"ep={CFG['num_epochs']}  AMP={CFG['use_amp']}")
print(f"{'='*60}\n")

# ── Cell 4: VRAM headroom check ────────────────────────────────────────────────
def check_vram():
    """Warn if we're likely to OOM before training even starts."""
    if not torch.cuda.is_available():
        return
    for i in range(torch.cuda.device_count()):
        total  = torch.cuda.get_device_properties(i).total_memory / 1024**3
        reserv = torch.cuda.memory_reserved(i) / 1024**3
        free   = total - reserv
        print(f"  GPU {i} VRAM: {free:.1f} GB free / {total:.1f} GB total")
        if free < 3.0:
            print(f"  ⚠️  GPU {i} has <3 GB free — consider restarting the kernel first.")

check_vram()

# ── Cell 5: Augmentation pipelines ────────────────────────────────────────────
_SZ = CFG["img_size"]

train_transform = A.Compose([
    A.RandomResizedCrop(size=(_SZ, _SZ), scale=(0.5, 1.0)),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.3),
    A.RandomRotate90(p=0.3),
    A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1, p=0.5),
    A.GaussNoise(std_range=(0.04, 0.22), p=0.3),
    A.GaussianBlur(blur_limit=3, p=0.2),
    A.ElasticTransform(p=0.2),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

val_transform = A.Compose([
    A.Resize(height=_SZ, width=_SZ),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

# ── Cell 6: Dataset ────────────────────────────────────────────────────────────
class DeepGlobeDataset(Dataset):
    """
    DeepGlobe Road Extraction loader.
      data_root/<id>_sat.jpg  — RGB satellite image
      data_root/<id>_mask.png — Binary mask (white=road, black=background)
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
        mask  = (mask > 127).astype(np.uint8)

        if self.transform:
            aug   = self.transform(image=image, mask=mask)
            image = aug["image"]
            mask  = aug["mask"]

        return {"pixel_values": image.float(), "labels": mask.long()}


def build_dataloaders(data_root, cfg):
    data_root = Path(data_root)
    sat_files  = sorted(data_root.glob("*_sat.jpg"))
    mask_files = [data_root / f.name.replace("_sat.jpg", "_mask.png") for f in sat_files]
    pairs = [(s, m) for s, m in zip(sat_files, mask_files) if m.exists()]
    print(f"Found {len(pairs)} image-mask pairs in: {data_root}")

    if not pairs:
        print("ERROR: 0 pairs. First 20 files in data_root:")
        for f in sorted(data_root.iterdir())[:20]:
            print(f"  {f}")
        raise FileNotFoundError(
            f"No *_sat.jpg/*_mask.png pairs in {data_root}.\n"
            "Attach the DeepGlobe dataset via Kaggle → Add Data."
        )

    random.shuffle(pairs)
    n_val = max(1, int(len(pairs) * cfg["val_split"]))
    train_pairs, val_pairs = pairs[n_val:], pairs[:n_val]

    train_ds = DeepGlobeDataset([p[0] for p in train_pairs],
                                [p[1] for p in train_pairs], train_transform)
    val_ds   = DeepGlobeDataset([p[0] for p in val_pairs],
                                [p[1] for p in val_pairs],   val_transform)

    nw = cfg.get("num_workers", 2)
    # persistent_workers=False → avoids Docker/Kaggle worker deadlocks
    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"],
                              shuffle=True,  num_workers=nw,
                              pin_memory=True, persistent_workers=False)
    val_loader   = DataLoader(val_ds,   batch_size=cfg["batch_size"] * 2,
                              shuffle=False, num_workers=nw,
                              pin_memory=True, persistent_workers=False)

    print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")
    return train_loader, val_loader


# ── Cell 7: Loss ───────────────────────────────────────────────────────────────
class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs     = torch.softmax(logits, dim=1)[:, 1]
        targets_f = targets.float()
        inter = (probs * targets_f).sum(dim=(1, 2))
        union = probs.sum(dim=(1, 2)) + targets_f.sum(dim=(1, 2))
        return 1.0 - ((2.0 * inter + self.smooth) / (union + self.smooth)).mean()


class CombinedLoss(nn.Module):
    def __init__(self, bce_w=1.0, dice_w=1.0):
        super().__init__()
        self.bce    = nn.CrossEntropyLoss()
        self.dice   = DiceLoss()
        self.bce_w  = bce_w
        self.dice_w = dice_w

    def forward(self, logits, targets):
        return self.bce_w * self.bce(logits, targets) + \
               self.dice_w * self.dice(logits, targets)


# ── Cell 8: mIoU ──────────────────────────────────────────────────────────────
def compute_miou(preds, targets, num_classes=2):
    preds, targets = preds.view(-1), targets.view(-1)
    ious = []
    for cls in range(num_classes):
        pm = (preds == cls)
        tm = (targets == cls)
        inter = (pm & tm).sum().float()
        union = (pm | tm).sum().float()
        ious.append(inter / union if union > 0 else torch.tensor(1.0))
    return torch.stack(ious).mean().item()


# ── Cell 9: Safe logits extractor ─────────────────────────────────────────────
def get_logits(outputs):
    """Works with plain model AND DataParallel wrapper."""
    if hasattr(outputs, "logits"):
        return outputs.logits
    if isinstance(outputs, dict) and "logits" in outputs:
        return outputs["logits"]
    # DataParallel sometimes returns a tuple
    if isinstance(outputs, (list, tuple)):
        return outputs[0]
    raise ValueError(f"Cannot extract logits from output type: {type(outputs)}")


# ── Cell 10: Model ─────────────────────────────────────────────────────────────
def build_model(cfg):
    print(f"Loading: {cfg['model_name']}")
    model = SegformerForSemanticSegmentation.from_pretrained(
        cfg["model_name"],
        num_labels=cfg["num_labels"],
        id2label={0: "background", 1: "road"},
        label2id={"background": 0, "road": 1},
        ignore_mismatched_sizes=True,
    )
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs via DataParallel")
        model = nn.DataParallel(model)
    model = model.to(DEVICE)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    return model


# ── Cell 11: Save checkpoint helper ───────────────────────────────────────────
def save_checkpoint(model, epoch, val_miou, val_loss, filename):
    state = (model.module if isinstance(model, nn.DataParallel)
             else model).state_dict()
    torch.save({
        "epoch":            epoch,
        "model_state_dict": state,
        "val_miou":         val_miou,
        "val_loss":         val_loss,
        "config":           CFG,
    }, filename)


# ── Cell 12: Train one epoch (OOM-safe, NaN-safe) ─────────────────────────────
def train_one_epoch(model, loader, optimizer, criterion, scaler, epoch):
    model.train()
    total_loss  = 0.0
    total_miou  = 0.0
    valid_steps = 0
    skipped_oom = 0
    skipped_nan = 0
    use_amp     = CFG.get("use_amp", True)
    grad_accum  = CFG.get("grad_accum", 1)

    pbar = tqdm(loader, desc=f"Epoch {epoch} [train]", leave=True)
    optimizer.zero_grad()

    for step, batch in enumerate(pbar):
        try:
            pixel_values = batch["pixel_values"].to(DEVICE, non_blocking=True)
            labels       = batch["labels"].to(DEVICE, non_blocking=True)

            with autocast(enabled=use_amp):
                outputs   = model(pixel_values=pixel_values)
                logits    = get_logits(outputs)
                upsampled = nn.functional.interpolate(
                    logits, size=labels.shape[-2:],
                    mode="bilinear", align_corners=False
                )
                loss = criterion(upsampled, labels) / grad_accum

            # ── NaN guard ──────────────────────────────────────────────────
            if not math.isfinite(loss.item() * grad_accum):
                skipped_nan += 1
                optimizer.zero_grad()
                print(f"\n  ⚠️  NaN loss at step {step} — skipping batch "
                      f"(total skipped: {skipped_nan})")
                if skipped_nan > 10:
                    print("  ❌ >10 NaN steps — disabling AMP and continuing")
                    CFG["use_amp"] = False
                    use_amp = False
                continue

            scaler.scale(loss).backward()

            if (step + 1) % grad_accum == 0 or (step + 1) == len(loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                old_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                if scaler.get_scale() < old_scale:
                    # AMP skipped the step due to inf/NaN gradients — not fatal
                    skipped_nan += 1
                optimizer.zero_grad()

            loss_val = loss.item() * grad_accum
            with torch.no_grad():
                preds = upsampled.argmax(dim=1)
                miou  = compute_miou(preds.cpu(), labels.cpu())

            total_loss  += loss_val
            total_miou  += miou
            valid_steps += 1
            pbar.set_postfix(loss=f"{loss_val:.4f}", miou=f"{miou:.3f}",
                             oom=skipped_oom, nan=skipped_nan)

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                skipped_oom += 1
                torch.cuda.empty_cache()
                optimizer.zero_grad()
                print(f"\n  ⚠️  OOM at step {step} — skipping batch "
                      f"(total skipped: {skipped_oom})")
                if skipped_oom > 20:
                    raise RuntimeError(
                        "Too many OOM errors (>20). "
                        "Reduce batch_size further or img_size in CFG."
                    ) from e
            else:
                raise

    if valid_steps == 0:
        return float("nan"), float("nan")
    if skipped_oom + skipped_nan > 0:
        print(f"  ℹ️  Epoch {epoch}: skipped {skipped_oom} OOM + {skipped_nan} NaN batches "
              f"out of {len(loader)} total")
    return total_loss / valid_steps, total_miou / valid_steps


# ── Cell 13: Evaluate ─────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, loader, criterion, epoch):
    model.eval()
    total_loss = 0.0
    total_miou = 0.0
    valid_batches = 0
    use_amp = CFG.get("use_amp", True)

    for batch in tqdm(loader, desc=f"Epoch {epoch} [val]  ", leave=False):
        try:
            pixel_values = batch["pixel_values"].to(DEVICE, non_blocking=True)
            labels       = batch["labels"].to(DEVICE, non_blocking=True)

            with autocast(enabled=use_amp):
                outputs   = model(pixel_values=pixel_values)
                logits    = get_logits(outputs)
                upsampled = nn.functional.interpolate(
                    logits, size=labels.shape[-2:],
                    mode="bilinear", align_corners=False
                )
                loss = criterion(upsampled, labels)

            if not math.isfinite(loss.item()):
                continue

            preds = upsampled.argmax(dim=1)
            miou  = compute_miou(preds.cpu(), labels.cpu())
            total_loss    += loss.item()
            total_miou    += miou
            valid_batches += 1

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
            else:
                raise

    if valid_batches == 0:
        return float("nan"), float("nan")
    return total_loss / valid_batches, total_miou / valid_batches


# ── Cell 14: Main ─────────────────────────────────────────────────────────────
def main():
    train_loader, val_loader = build_dataloaders(CFG["data_root"], CFG)
    model     = build_model(CFG)
    criterion = CombinedLoss(bce_w=CFG["bce_weight"], dice_w=CFG["dice_weight"])
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=CFG["lr"], weight_decay=CFG["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=CFG["num_epochs"], eta_min=1e-6
    )
    scaler  = GradScaler(enabled=CFG.get("use_amp", True))
    best_miou = 0.0
    log = []

    print("\n" + "=" * 60)
    print("  Starting RoadRishi SegFormer Fine-Tuning")
    print(f"  Epochs={CFG['num_epochs']}  LR={CFG['lr']}  "
          f"Batch={CFG['batch_size']} (eff={CFG['batch_size']*CFG['grad_accum']})  "
          f"img={CFG['img_size']}  AMP={CFG['use_amp']}")
    print("=" * 60)
    check_vram()

    for epoch in range(1, CFG["num_epochs"] + 1):
        t0 = time.time()

        train_loss, train_miou = train_one_epoch(
            model, train_loader, optimizer, criterion, scaler, epoch
        )
        val_loss, val_miou = evaluate(model, val_loader, criterion, epoch)
        scheduler.step()

        elapsed = time.time() - t0
        remaining = elapsed * (CFG["num_epochs"] - epoch)
        print(
            f"Epoch {epoch:2d}/{CFG['num_epochs']} | "
            f"Train {train_loss:.4f}/{train_miou:.4f} | "
            f"Val {val_loss:.4f}/{val_miou:.4f} | "
            f"{elapsed/60:.1f} min | ETA {remaining/3600:.1f} h"
        )

        torch.cuda.empty_cache()

        row = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4) if math.isfinite(train_loss) else None,
            "train_miou": round(train_miou, 4) if math.isfinite(train_miou) else None,
            "val_loss":   round(val_loss,   4) if math.isfinite(val_loss)   else None,
            "val_miou":   round(val_miou,   4) if math.isfinite(val_miou)   else None,
            "lr":         round(scheduler.get_last_lr()[0], 8),
            "epoch_secs": round(elapsed, 1),
        }
        log.append(row)

        # ── Save best checkpoint ───────────────────────────────────────────
        if math.isfinite(val_miou) and val_miou > best_miou:
            best_miou = val_miou
            best_path = os.path.join(CFG["output_dir"], "roadrishi_finetuned.pth")
            save_checkpoint(model, epoch, val_miou, val_loss, best_path)
            print(f"  ✅ Best checkpoint saved → val_mIoU={val_miou:.4f}")

        # ── Periodic checkpoint (every N epochs) — survives crashes ───────
        if epoch % CFG.get("save_every", 2) == 0:
            periodic_path = os.path.join(
                CFG["output_dir"], f"checkpoint_epoch{epoch:02d}.pth"
            )
            save_checkpoint(model, epoch, val_miou, val_loss, periodic_path)
            print(f"  💾 Periodic checkpoint saved → epoch {epoch}")

        # ── Flush training log after every epoch ──────────────────────────
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
