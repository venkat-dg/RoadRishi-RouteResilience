"""
RoadRishi — SegFormer Fine-Tuning Script (Kaggle-Ready, Crash-Proof + Resume, Early stopping implemented)
==============================================================================
Dataset: https://www.kaggle.com/datasets/balraj98/deepglobe-road-extraction-dataset

Instructions:
1. Enable GPU T4 x2 in Kaggle Settings
2. Run All
3. If it crashes, just Run All again — it auto-resumes from the last checkpoint

Expected runtime: ~2.5–3.5 hours on T4 x2
  img_size=384  batch=8  grad_accum=2  epochs=8  AMP=True  num_workers=0

Crash defences:
  ✅ CUDA OOM mid-batch  → skip batch, empty_cache, continue
  ✅ NaN loss            → skip step; auto-disable AMP after 10 NaNs
  ✅ Periodic checkpoint → saved every 2 epochs with full optimizer state
  ✅ AUTO-RESUME         → detects last checkpoint and continues from next epoch
  ✅ num_workers=0       → no Docker worker deadlocks between epochs
  ✅ VRAM check          → warns before training if headroom is low
"""

# ── Forced Albumentations Upgrade ──────────────────────────────────────────────
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "albumentations>=1.4.15"], check=True)

# ── Dependency check ───────────────────────────────────────────────────────────
import importlib, sys
for pkg, (maj, mn) in {"transformers": (4, 37), "albumentations": (1, 4)}.items():
    try:
        mod = importlib.import_module(pkg)
        ver = getattr(mod, "__version__", "0.0")
        parts = [int(x) for x in ver.split(".")[:2] if x.isdigit()]
        ok = tuple(parts) >= (maj, mn)
        print(f"{'✅' if ok else '⚠️ TOO OLD'}  {pkg} {ver}")
    except ImportError:
        print(f"❌  {pkg} not found"); sys.exit(1)

# ── Imports ────────────────────────────────────────────────────────────────────
import os, json, math, time, random, warnings
import numpy as np
from pathlib import Path
from PIL import Image

import torch
import torch.nn as nn
# Use the non-deprecated AMP API (torch >= 2.0; torch.cuda.amp still works but warns on >=2.4)
try:
    from torch.amp import autocast, GradScaler          # torch >= 2.0
    _AMP_DEVICE = "cuda"
except ImportError:
    from torch.cuda.amp import autocast, GradScaler     # torch < 2.0 fallback
    _AMP_DEVICE = None  # old API doesn't need device arg
from torch.utils.data import Dataset, DataLoader
from transformers import SegformerForSemanticSegmentation
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning)

print(f"PyTorch {torch.__version__} | CUDA {torch.cuda.is_available()}")
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print(f"  GPU {i}: {p.name}  {p.total_memory//1024**2} MB")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Auto-detect dataset ────────────────────────────────────────────────────────
def _find_deepglobe_root() -> str:
    import glob as _g
    masks = _g.glob("/kaggle/input/**/*_mask.png", recursive=True)
    if masks:
        dirs = list({str(Path(p).parent) for p in masks})
        chosen = next((d for d in dirs if Path(d).name == "train"), dirs[0])
        print(f"[AUTO-DETECT] {chosen}  ({len([m for m in masks if str(Path(m).parent)==chosen])} masks)")
        return chosen
    fallback = "/kaggle/input/deepglobe-road-extraction-dataset/train"
    print(f"[WARN] No masks found. Falling back to {fallback}")
    return fallback

# ── Config ─────────────────────────────────────────────────────────────────────
CFG = {
    "data_root":    _find_deepglobe_root(),
    "output_dir":   "/kaggle/working/checkpoints",
    "model_name":   "nvidia/segformer-b3-finetuned-ade-512-512",
    "num_labels":   2,
    "img_size":     384,
    "batch_size":   8,       # Increased from 4 to 8 to utilize the 15GB VRAM on T4x2 GPUs
    "grad_accum":   2,
    "num_epochs":   8,
    "lr":           5e-5,
    "weight_decay": 1e-4,
    "val_split":    0.1,
    "num_workers":  2,       # Set to 2 so disk I/O doesn't bottleneck GPU execution
    "seed":         42,
    "bce_weight":   1.0,
    "dice_weight":  1.0,
    "use_amp":      True,
    "save_every":   2,
    "early_stopping_patience": 3,
}

os.makedirs(CFG["output_dir"], exist_ok=True)
random.seed(CFG["seed"]); np.random.seed(CFG["seed"])
torch.manual_seed(CFG["seed"])
if torch.cuda.is_available(): torch.cuda.manual_seed_all(CFG["seed"])

print(f"\nimg={CFG['img_size']} bs={CFG['batch_size']} accum={CFG['grad_accum']} "
      f"eff_bs={CFG['batch_size']*CFG['grad_accum']} ep={CFG['num_epochs']} "
      f"AMP={CFG['use_amp']} workers={CFG['num_workers']}\n")

# ── VRAM check ─────────────────────────────────────────────────────────────────
def check_vram():
    if not torch.cuda.is_available(): return
    for i in range(torch.cuda.device_count()):
        tot  = torch.cuda.get_device_properties(i).total_memory / 1024**3
        used = torch.cuda.memory_reserved(i) / 1024**3
        free = tot - used
        flag = "⚠️" if free < 3 else "✅"
        print(f"  {flag} GPU {i}: {free:.1f}/{tot:.1f} GB free")

# ── Augmentations ──────────────────────────────────────────────────────────────
_SZ = CFG["img_size"]
train_transform = A.Compose([
    A.RandomResizedCrop(size=(_SZ, _SZ), scale=(0.5, 1.0)),
    A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.3), A.RandomRotate90(p=0.3),
    A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1, p=0.5),
    A.GaussNoise(std_range=(0.04, 0.22), p=0.3),
    A.GaussianBlur(blur_limit=3, p=0.2), A.ElasticTransform(p=0.2),
    A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)), ToTensorV2(),
])
val_transform = A.Compose([
    A.Resize(height=_SZ, width=_SZ),
    A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)), ToTensorV2(),
])

# ── Dataset ────────────────────────────────────────────────────────────────────
class DeepGlobeDataset(Dataset):
    def __init__(self, imgs, masks, tfm=None):
        self.imgs, self.masks, self.tfm = imgs, masks, tfm
    def __len__(self): return len(self.imgs)
    def __getitem__(self, i):
        img  = np.array(Image.open(self.imgs[i]).convert("RGB"))
        mask = np.array(Image.open(self.masks[i]).convert("L"))
        mask = (mask > 127).astype(np.uint8)
        if self.tfm:
            aug = self.tfm(image=img, mask=mask)
            img, mask = aug["image"], aug["mask"]
        return {"pixel_values": img.float(), "labels": mask.long()}

def build_dataloaders(cfg):
    root = Path(cfg["data_root"])
    sats  = sorted(root.glob("*_sat.jpg"))
    masks = [root / s.name.replace("_sat.jpg","_mask.png") for s in sats]
    pairs = [(s,m) for s,m in zip(sats,masks) if m.exists()]
    if not pairs:
        raise FileNotFoundError(f"No image-mask pairs in {root}")
    print(f"Pairs: {len(pairs)}")
    random.shuffle(pairs)
    n = max(1, int(len(pairs)*cfg["val_split"]))
    tr, va = pairs[n:], pairs[:n]
    nw = cfg["num_workers"]
    _pin = torch.cuda.is_available()   # pin_memory requires CUDA; crashes silently on CPU
    tl = DataLoader(DeepGlobeDataset([p[0] for p in tr],[p[1] for p in tr],train_transform),
                    batch_size=cfg["batch_size"], shuffle=True, num_workers=nw,
                    pin_memory=_pin, persistent_workers=False)
    vl = DataLoader(DeepGlobeDataset([p[0] for p in va],[p[1] for p in va],val_transform),
                    batch_size=cfg["batch_size"]*2, shuffle=False, num_workers=nw,
                    pin_memory=_pin, persistent_workers=False)
    print(f"Train {len(tr)} | Val {len(va)}")
    return tl, vl

# ── Loss ───────────────────────────────────────────────────────────────────────
class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0): super().__init__(); self.s = smooth
    def forward(self, logits, targets):
        p = torch.softmax(logits,1)[:,1]; t = targets.float()
        i = (p*t).sum((1,2)); u = p.sum((1,2))+t.sum((1,2))
        return 1 - ((2*i+self.s)/(u+self.s)).mean()

class CombinedLoss(nn.Module):
    def __init__(self, bw, dw):
        super().__init__()
        weights = torch.tensor([1.0, 4.0], device=DEVICE)
        self.bce = nn.CrossEntropyLoss(weight=weights)
        self.dice = DiceLoss()
        self.bw = bw
        self.dw = dw
    def forward(self, logits, targets):
        return self.bw*self.bce(logits,targets) + self.dw*self.dice(logits,targets)

# ── mIoU and metrics (mIoU was extended with more verbose metrics computations)───────────────────────────────────────────────────────────────────────
def compute_miou(preds, targets, nc=2):
    p,t = preds.view(-1), targets.view(-1)
    ious = []
    for c in range(nc):
        pm,tm = (p==c),(t==c)
        inter=(pm&tm).sum().float(); union=(pm|tm).sum().float()
        ious.append(inter/union if union>0 else torch.tensor(1.0))
    return torch.stack(ious).mean().item()

def compute_metrics(preds, targets):
    preds   = preds.reshape(-1)
    targets = targets.reshape(-1)

    # Road class = 1
    tp = ((preds == 1) & (targets == 1)).sum().float()
    fp = ((preds == 1) & (targets == 0)).sum().float()
    fn = ((preds == 0) & (targets == 1)).sum().float()
    tn = ((preds == 0) & (targets == 0)).sum().float()

    eps = 1e-7

    iou_road = tp / (tp + fp + fn + eps)
    iou_bg   = tn / (tn + fp + fn + eps)

    miou = (iou_road + iou_bg) / 2

    precision = tp / (tp + fp + eps)
    recall    = tp / (tp + fn + eps)

    f1 = 2 * precision * recall / (precision + recall + eps)

    dice = 2 * tp / (2 * tp + fp + fn + eps)

    return {
        "miou": miou.item(),
        "dice": dice.item(),
        "f1": f1.item(),
        "precision": precision.item(),
        "recall": recall.item(),
    }

# ── Safe logits extractor ──────────────────────────────────────────────────────
def get_logits(out):
    if hasattr(out,"logits"): return out.logits
    if isinstance(out,dict) and "logits" in out: return out["logits"]
    if isinstance(out,(list,tuple)): return out[0]
    raise ValueError(f"Cannot extract logits from {type(out)}")

# ── Model ──────────────────────────────────────────────────────────────────────
def build_model(cfg):
    print(f"Loading {cfg['model_name']}")
    m = SegformerForSemanticSegmentation.from_pretrained(
        cfg["model_name"], num_labels=cfg["num_labels"],
        id2label={0:"background",1:"road"}, label2id={"background":0,"road":1},
        ignore_mismatched_sizes=True)
    if torch.cuda.device_count() > 1:
        print(f"DataParallel x{torch.cuda.device_count()}")
        m = nn.DataParallel(m)
    m = m.to(DEVICE)
    print(f"Params: {sum(p.numel() for p in m.parameters()):,}")
    return m

# ── Checkpoint helpers ─────────────────────────────────────────────────────────
def _save_full(model, optimizer, scheduler, scaler, epoch, best_miou, val_loss, log, path):
    """Save everything needed to resume training."""
    raw = model.module if isinstance(model, nn.DataParallel) else model
    torch.save({
        "epoch":              epoch,
        "model_state_dict":   raw.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict":  scaler.state_dict(),
        "best_miou":          best_miou,
        "val_loss":           val_loss,
        "log":                log,
        "config":             CFG,
    }, path)

def find_resume_checkpoint(output_dir):
    """Returns path of latest resumable checkpoint, or None."""
    out = Path(output_dir)
    periodics = sorted(out.glob("checkpoint_epoch*.pth"))
    if periodics:
        return str(periodics[-1])
    best = out / "roadrishi_finetuned.pth"
    if best.exists():
        try:
            ckpt = torch.load(best, map_location="cpu")
            if "optimizer_state_dict" in ckpt:
                return str(best)
        except Exception:
            pass
    return None

def load_checkpoint(path, model, optimizer, scheduler, scaler):
    """Load all training state. Returns (start_epoch, best_miou, log)."""
    print(f"\n  🔄 Resuming from: {path}")
    ckpt = torch.load(path, map_location=DEVICE)
    raw  = model.module if isinstance(model, nn.DataParallel) else model
    raw.load_state_dict(ckpt["model_state_dict"])
    if "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if "scheduler_state_dict" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    if ckpt.get("scaler_state_dict"):
        scaler.load_state_dict(ckpt["scaler_state_dict"])
    start = ckpt.get("epoch", 0) + 1
    best  = ckpt.get("best_miou", ckpt.get("val_miou", 0.0))
    log   = ckpt.get("log", [])
    print(f"  ✅ Resuming from epoch {start}  (best mIoU so far: {best:.4f})\n")
    return start, best, log

# ── Safe float format (handles nan/inf without crashing) ─────────────────────
def _fmt(v, fmt=".4f"):
    return f"{v:{fmt}}" if isinstance(v, float) and math.isfinite(v) else str(v)

# ── Training loop (OOM-safe, NaN-safe) ────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, criterion, scaler, epoch):
    model.train()
    valid = skipped_oom = skipped_nan = 0
    total_loss = 0
    total_miou = 0
    total_dice = 0
    total_f1   = 0
    use_amp    = CFG.get("use_amp", True)
    grad_accum = CFG.get("grad_accum", 1)
    pbar = tqdm(loader, desc=f"Epoch {epoch} [train]", leave=True)
    optimizer.zero_grad()

    for step, batch in enumerate(pbar):
        try:
            pv = batch["pixel_values"].to(DEVICE, non_blocking=True)
            lb = batch["labels"].to(DEVICE, non_blocking=True)

            # ── Forward pass inside AMP context ───────────────────────────
            _dev = _AMP_DEVICE if _AMP_DEVICE else "cuda"
            with autocast(device_type=_dev, enabled=use_amp):
                out    = model(pixel_values=pv)
                logits = get_logits(out)
                up     = nn.functional.interpolate(logits, size=lb.shape[-2:],
                                                   mode="bilinear", align_corners=False)
                loss   = criterion(up, lb) / grad_accum

            # ── NaN guard: check BEFORE backward to avoid graph corruption ─
            raw_loss = loss.item() * grad_accum
            if not math.isfinite(raw_loss):
                skipped_nan += 1
                optimizer.zero_grad()   # discard any partial gradients
                torch.cuda.empty_cache()
                if skipped_nan > 10:
                    print("\n  ⚠️  >10 NaN steps — disabling AMP and continuing in fp32")
                    CFG["use_amp"] = False
                    use_amp = False
                continue

            scaler.scale(loss).backward()

            if (step+1) % grad_accum == 0 or (step+1) == len(loader):
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                old_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                # Scaler shrinks when it detects inf/NaN gradients — count it
                if scaler.get_scale() < old_scale:
                    skipped_nan += 1
                optimizer.zero_grad()

            lv = raw_loss
            with torch.no_grad():
                # miou = compute_miou(up.argmax(1).cpu(), lb.cpu())
                metrics = compute_metrics(
                    up.argmax(1),
                    lb
                )
            total_loss += lv
            total_miou += metrics["miou"]
            total_dice += metrics["dice"]
            total_f1   += metrics["f1"]
            valid += 1
            pbar.set_postfix(
                loss=f"{lv:.4f}",
                miou=f"{metrics['miou']:.3f}",
                dice=f"{metrics['dice']:.3f}",
                oom=skipped_oom,
                nan=skipped_nan
            )

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                skipped_oom += 1
                torch.cuda.empty_cache(); optimizer.zero_grad()
                if skipped_oom > 20:
                    raise RuntimeError("Too many OOM errors. Reduce batch_size.") from e
            else:
                raise

    if valid == 0: 
        return (
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan")
        )
    if skipped_oom + skipped_nan:
        print(f"  ℹ️  Skipped {skipped_oom} OOM + {skipped_nan} NaN out of {len(loader)}")
    return (
        total_loss / valid,
        total_miou / valid,
        total_dice / valid,
        total_f1 / valid
    )

# ── Eval loop ──────────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, loader, criterion, epoch):
    model.eval()
    valid = 0
    total_loss = 0
    total_miou = 0
    total_dice = 0
    total_f1   = 0
    use_amp = CFG.get("use_amp", True)
    for batch in tqdm(loader, desc=f"Epoch {epoch} [val]  ", leave=False):
        try:
            pv = batch["pixel_values"].to(DEVICE, non_blocking=True)
            lb = batch["labels"].to(DEVICE, non_blocking=True)
            _dev = _AMP_DEVICE if _AMP_DEVICE else "cuda"
            with autocast(device_type=_dev, enabled=use_amp):
                out    = model(pixel_values=pv)
                logits = get_logits(out)
                up     = nn.functional.interpolate(logits, size=lb.shape[-2:],
                                                   mode="bilinear", align_corners=False)
                loss   = criterion(up, lb)
            if not math.isfinite(loss.item()): continue
            total_loss += loss.item()
            metrics = compute_metrics(
                up.argmax(1),
                lb
            )

            total_miou += metrics["miou"]
            total_dice += metrics["dice"]
            total_f1   += metrics["f1"]
            valid += 1
        except RuntimeError as e:
            if "out of memory" in str(e).lower(): torch.cuda.empty_cache()
            else: raise
    if valid == 0:
        return (
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan")
        )
    return (
        total_loss / valid,
        total_miou / valid,
        total_dice / valid,
        total_f1 / valid
    )

# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    tl, vl   = build_dataloaders(CFG)
    model     = build_model(CFG)
    criterion = CombinedLoss(CFG["bce_weight"], CFG["dice_weight"])
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=CFG["lr"], weight_decay=CFG["weight_decay"])
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    #                 optimizer, T_max=CFG["num_epochs"], eta_min=1e-6)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=2,
        T_mult=2,
        eta_min=1e-6
    )
    scaler    = GradScaler("cuda", enabled=CFG.get("use_amp", True))
    best_miou = 0.0
    log = []
    start_epoch = 1
    epochs_without_improvement = 0

    # ── Auto-resume ────────────────────────────────────────────────────────────
    resume = find_resume_checkpoint(CFG["output_dir"])
    if resume:
        start_epoch, best_miou, log = load_checkpoint(
            resume, model, optimizer, scheduler, scaler)
        if start_epoch > CFG["num_epochs"]:
            print("✅ All epochs already complete."); return
    else:
        print("ℹ️  No checkpoint found — starting fresh.")

    print("\n" + "="*60)
    print(f"  RoadRishi SegFormer  |  epochs {start_epoch}–{CFG['num_epochs']}")
    print(f"  img={CFG['img_size']}  bs={CFG['batch_size']}  "
          f"eff_bs={CFG['batch_size']*CFG['grad_accum']}  AMP={CFG['use_amp']}")
    print("="*60)
    check_vram()

    log_path = os.path.join(CFG["output_dir"], "training_log.json")

    for epoch in range(start_epoch, CFG["num_epochs"]+1):
        t0 = time.time()
        train_loss, train_miou, train_dice, train_f1 = train_one_epoch(
            model, tl, optimizer, criterion, scaler, epoch)
        val_loss, val_miou, val_dice, val_f1 = evaluate(model, vl, criterion, epoch)
        scheduler.step()
        torch.cuda.empty_cache()

        elapsed = time.time()-t0
        eta     = elapsed * (CFG["num_epochs"]-epoch)
        # _fmt() prevents ValueError crash when loss=nan (e.g. all batches skipped)
        print(
            f"Epoch {epoch:2d}/{CFG['num_epochs']} | "
            f"Train Loss={_fmt(train_loss)} "
            f"mIoU={_fmt(train_miou)} "
            f"Dice={_fmt(train_dice)} "
            f"F1={_fmt(train_f1)} | "
            f"Val Loss={_fmt(val_loss)} "
            f"mIoU={_fmt(val_miou)} "
            f"Dice={_fmt(val_dice)} "
            f"F1={_fmt(val_f1)} | "
            f"{elapsed/60:.1f} min | ETA {eta/3600:.1f} h"
        )

        row = {
            "epoch":      epoch,
            "train_loss": round(train_loss,4) if math.isfinite(train_loss) else None,
            "train_miou": round(train_miou,4) if math.isfinite(train_miou) else None,
            "val_loss":   round(val_loss,4)   if math.isfinite(val_loss)   else None,
            "val_miou":   round(val_miou,4)   if math.isfinite(val_miou)   else None,
            "lr":         round(scheduler.get_last_lr()[0],8),
            "secs":       round(elapsed,1),
            "train_dice": round(train_dice,4) if math.isfinite(train_dice) else None,
            "train_f1":   round(train_f1,4)   if math.isfinite(train_f1)   else None,
            "val_dice":   round(val_dice,4)   if math.isfinite(val_dice)   else None,
            "val_f1":     round(val_f1,4)     if math.isfinite(val_f1)     else None,
        }
        log.append(row)

        # Best checkpoint (full state for resume)
        if math.isfinite(val_miou) and val_miou > best_miou + 1e-4:
            best_miou = val_miou
            epochs_without_improvement = 0

            best_path = os.path.join(
                CFG["output_dir"],
                "roadrishi_finetuned.pth"
            )

            _save_full(
                model,
                optimizer,
                scheduler,
                scaler,
                epoch,
                best_miou,
                val_loss,
                log,
                best_path
            )

            print(f"  ✅ Best saved → val_mIoU={val_miou:.4f}")

        else:
            epochs_without_improvement += 1

        # Periodic checkpoint every N epochs
        if epoch % CFG.get("save_every",2) == 0:
            ppath = os.path.join(CFG["output_dir"], f"checkpoint_epoch{epoch:02d}.pth")
            _save_full(model, optimizer, scheduler, scaler,
                       epoch, best_miou, val_loss, log, ppath)
            print(f"  💾 Periodic checkpoint → epoch {epoch}")
        
        if epochs_without_improvement >= CFG["early_stopping_patience"]:
            print(
                f"\n🛑 Early stopping triggered "
                f"({epochs_without_improvement} epochs without improvement)"
            )
            break

        # Flush log after every epoch
        with open(log_path,"w") as f:   
            json.dump(log, f, indent=2)

    print("\n" + "="*60)
    print(f"  Done! Best val mIoU: {best_miou:.4f}")
    print(f"  Download: {CFG['output_dir']}/roadrishi_finetuned.pth")
    print("="*60)


if __name__ == "__main__":
    main()
