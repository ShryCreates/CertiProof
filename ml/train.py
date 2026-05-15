"""
CertiProof — Training Pipeline
===================================
Full training loop for the 6-channel EfficientNet-B4 forgery detector.

Dataset layout (actual):
    training_data/
        genuine/images/   ← label 1
        fake/images/      ← label 0

Features
--------
- Stratified Train / Val / Test split : 70 / 15 / 15
- Class-weighted loss                 : handles imbalance automatically
- Augmentations                       : HorizontalFlip, Rotation, ColorJitter, RandomErasing
- Optimiser                           : AdamW (lr=3e-4)
- Scheduler                           : CosineAnnealingLR
- Loss                                : CrossEntropyLoss with class weights
- Mixed precision                     : torch.cuda.amp (CUDA only)
- Metrics                             : accuracy, precision, recall, F1, ROC-AUC
- Experiment tracking                 : MLflow
- Checkpointing                       : best model by validation AUC → checkpoints/best_model.pth
"""

import logging
import os
import time
from pathlib import Path

import mlflow
import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from tqdm import tqdm

from forgery_detector import MODEL_CONFIG, build_model, generate_ela, preprocess

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("train")

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
DATA_DIR = ROOT / "training_data"
GENUINE_DIR = DATA_DIR / "genuine" / "images"   # ← fixed: data is in /images subdir
FAKE_DIR    = DATA_DIR / "fake"    / "images"   # ← fixed: data is in /images subdir
CHECKPOINT_DIR = ROOT / "checkpoints"
CHECKPOINT_DIR.mkdir(exist_ok=True)
BEST_MODEL_PATH = CHECKPOINT_DIR / "best_model.pth"

# ── Device ────────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Windows: multiprocessing with DataLoader requires num_workers=0
NUM_WORKERS = 0 if os.name == "nt" else min(4, os.cpu_count() or 1)
logger.info("Using device: %s  |  num_workers: %d", DEVICE, NUM_WORKERS)


# ═══════════════════════════════════════════════════════════════════════════════
# DATASET
# ═══════════════════════════════════════════════════════════════════════════════

class CertificateDataset(Dataset):
    """
    Loads certificate images from genuine/images/ and fake/images/ directories.
    Each sample → 6-channel tensor (3 RGB + 3 ELA).
    Labels: genuine = 1, fake = 0.
    """

    _augment_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=5, fill=255),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.1), ratio=(0.3, 3.3), value=1.0),
    ])

    _base_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    def __init__(self, image_paths: list[Path], labels: list[int], augment: bool = False) -> None:
        assert len(image_paths) == len(labels)
        self.image_paths = image_paths
        self.labels = labels
        self.augment = augment

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        import cv2
        path  = self.image_paths[idx]
        label = self.labels[idx]

        try:
            # ── Preprocess ────────────────────────────────────────────────────
            rgb_float = preprocess(path)                        # (H, W, 3) float32 [0,1]

            # ── ELA ───────────────────────────────────────────────────────────
            ela_uint8   = generate_ela(path)
            ela_resized = cv2.resize(
                ela_uint8,
                (MODEL_CONFIG["image_width"], MODEL_CONFIG["image_height"]),
                interpolation=cv2.INTER_AREA,
            )
            ela_float = ela_resized.astype(np.float32) / 255.0

        except Exception as e:
            logger.warning("Skipping corrupt image %s: %s", path.name, e)
            # Return a blank tensor so the batch doesn't crash
            blank = torch.zeros(6, MODEL_CONFIG["image_height"], MODEL_CONFIG["image_width"])
            return blank, label

        # ── Augmentation ──────────────────────────────────────────────────────
        rgb_pil = Image.fromarray((rgb_float * 255).astype(np.uint8))
        if self.augment:
            rgb_tensor = self._augment_transform(rgb_pil)
        else:
            rgb_tensor = self._base_transform(rgb_pil)

        ela_tensor = torch.from_numpy(ela_float).permute(2, 0, 1).float()
        combined   = torch.cat([rgb_tensor, ela_tensor], dim=0)  # (6, H, W)
        return combined, label


# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADING & STRATIFIED SPLIT
# ═══════════════════════════════════════════════════════════════════════════════

def load_dataset() -> tuple[list[Path], list[int]]:
    """Scan genuine/images/ and fake/images/ and return (paths, labels)."""
    supported = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
    paths:  list[Path] = []
    labels: list[int]  = []

    for label, directory in [(1, GENUINE_DIR), (0, FAKE_DIR)]:
        if not directory.exists():
            raise RuntimeError(f"Directory not found: {directory}")
        found = sorted([p for p in directory.iterdir() if p.suffix.lower() in supported])
        logger.info("  [label=%d] Found %d images in %s", label, len(found), directory)
        paths.extend(found)
        labels.extend([label] * len(found))

    if not paths:
        raise RuntimeError("No images found. Check training_data/genuine/images/ and training_data/fake/images/")

    n_genuine = labels.count(1)
    n_fake    = labels.count(0)
    logger.info("Total: %d images  (genuine=%d  fake=%d  ratio=1:%.1f)",
                len(paths), n_genuine, n_fake, n_fake / max(n_genuine, 1))
    return paths, labels


def make_stratified_splits(
    paths:       list[Path],
    labels:      list[int],
    train_ratio: float = 0.70,
    val_ratio:   float = 0.15,
    seed:        int   = 42,
) -> tuple[CertificateDataset, CertificateDataset, CertificateDataset]:
    """
    Stratified split — each subset preserves the genuine/fake ratio.
    Prevents all genuine samples ending up in one split.
    """
    rng = np.random.default_rng(seed)

    genuine_idx = [i for i, l in enumerate(labels) if l == 1]
    fake_idx    = [i for i, l in enumerate(labels) if l == 0]

    def _split_class(idx_list):
        arr = np.array(idx_list)
        rng.shuffle(arr)
        n = len(arr)
        n_train = int(n * train_ratio)
        n_val   = int(n * val_ratio)
        return arr[:n_train].tolist(), arr[n_train:n_train + n_val].tolist(), arr[n_train + n_val:].tolist()

    g_train, g_val, g_test = _split_class(genuine_idx)
    f_train, f_val, f_test = _split_class(fake_idx)

    def _make_ds(g_idx, f_idx, augment):
        combined = g_idx + f_idx
        rng.shuffle(combined := np.array(combined))
        return CertificateDataset(
            image_paths=[paths[i]  for i in combined],
            labels     =[labels[i] for i in combined],
            augment    =augment,
        )

    train_ds = _make_ds(g_train, f_train, augment=True)
    val_ds   = _make_ds(g_val,   f_val,   augment=False)
    test_ds  = _make_ds(g_test,  f_test,  augment=False)

    logger.info("Stratified split → train=%d  val=%d  test=%d",
                len(train_ds), len(val_ds), len(test_ds))
    return train_ds, val_ds, test_ds


def compute_class_weights(labels: list[int]) -> torch.Tensor:
    """
    Compute inverse-frequency class weights for CrossEntropyLoss.
    weight[c] = total_samples / (num_classes * count[c])
    """
    n_total   = len(labels)
    n_genuine = labels.count(1)
    n_fake    = labels.count(0)
    w_fake    = n_total / (2 * n_fake)
    w_genuine = n_total / (2 * n_genuine)
    weights   = torch.tensor([w_fake, w_genuine], dtype=torch.float32)
    logger.info("Class weights → fake=%.4f  genuine=%.4f", w_fake, w_genuine)
    return weights


# ═══════════════════════════════════════════════════════════════════════════════
# METRICS
# ═══════════════════════════════════════════════════════════════════════════════

def compute_metrics(
    all_labels: list[int],
    all_preds:  list[int],
    all_probs:  list[float],
) -> dict[str, float]:
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = float("nan")

    return {
        "accuracy":  accuracy_score(all_labels, all_preds),
        "precision": precision_score(all_labels, all_preds, zero_division=0),
        "recall":    recall_score(all_labels, all_preds, zero_division=0),
        "f1":        f1_score(all_labels, all_preds, zero_division=0),
        "roc_auc":   auc,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# TRAINING LOOP
# ═══════════════════════════════════════════════════════════════════════════════

def train_one_epoch(
    model:     nn.Module,
    loader:    DataLoader,
    optimiser: torch.optim.Optimizer,
    criterion: nn.Module,
    scaler:    GradScaler,
    epoch:     int,
) -> float:
    model.train()
    total_loss = 0.0
    use_amp    = DEVICE.type == "cuda"

    pbar = tqdm(loader, desc=f"Epoch {epoch:03d} [train]", leave=False, unit="batch")

    for inputs, targets in pbar:
        inputs  = inputs.to(DEVICE, non_blocking=True)
        targets = targets.to(DEVICE, non_blocking=True)

        optimiser.zero_grad(set_to_none=True)

        with autocast(device_type=DEVICE.type, enabled=use_amp):
            logits = model(inputs)
            loss   = criterion(logits, targets)

        scaler.scale(loss).backward()
        scaler.unscale_(optimiser)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimiser)
        scaler.update()

        total_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / len(loader)


# ═══════════════════════════════════════════════════════════════════════════════
# VALIDATION LOOP
# ═══════════════════════════════════════════════════════════════════════════════

def validate(
    model:      nn.Module,
    loader:     DataLoader,
    criterion:  nn.Module,
    epoch:      int,
    split_name: str = "val",
) -> tuple[float, dict[str, float]]:
    model.eval()
    total_loss  = 0.0
    all_labels: list[int]   = []
    all_preds:  list[int]   = []
    all_probs:  list[float] = []
    use_amp     = DEVICE.type == "cuda"

    pbar = tqdm(loader, desc=f"Epoch {epoch:03d} [{split_name}]", leave=False, unit="batch")

    with torch.no_grad():
        for inputs, targets in pbar:
            inputs  = inputs.to(DEVICE, non_blocking=True)
            targets = targets.to(DEVICE, non_blocking=True)

            with autocast(device_type=DEVICE.type, enabled=use_amp):
                logits = model(inputs)
                loss   = criterion(logits, targets)

            total_loss += loss.item()

            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = torch.argmax(logits, dim=1)

            all_labels.extend(targets.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())
            all_probs.extend(probs.cpu().tolist())

    metrics = compute_metrics(all_labels, all_preds, all_probs)
    return total_loss / len(loader), metrics


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN TRAINING ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def train() -> None:
    logger.info("═" * 60)
    logger.info("CertiProof — Training Pipeline")
    logger.info("═" * 60)

    # ── Dataset ───────────────────────────────────────────────────────────────
    paths, labels = load_dataset()
    train_ds, val_ds, test_ds = make_stratified_splits(paths, labels)

    train_loader = DataLoader(
        train_ds,
        batch_size  = MODEL_CONFIG["batch_size"],
        shuffle     = True,
        num_workers = NUM_WORKERS,
        pin_memory  = DEVICE.type == "cuda",
        drop_last   = False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size  = MODEL_CONFIG["batch_size"],
        shuffle     = False,
        num_workers = NUM_WORKERS,
        pin_memory  = DEVICE.type == "cuda",
    )
    test_loader = DataLoader(
        test_ds,
        batch_size  = MODEL_CONFIG["batch_size"],
        shuffle     = False,
        num_workers = NUM_WORKERS,
        pin_memory  = DEVICE.type == "cuda",
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = build_model().to(DEVICE)

    # ── Class weights ─────────────────────────────────────────────────────────
    class_weights = compute_class_weights(labels).to(DEVICE)

    # ── Optimiser & Scheduler ─────────────────────────────────────────────────
    optimiser = torch.optim.AdamW(
        model.parameters(),
        lr           = MODEL_CONFIG["learning_rate"],
        weight_decay = 1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser,
        T_max   = MODEL_CONFIG["epochs"],
        eta_min = 1e-6,
    )

    # ── Loss & Scaler ─────────────────────────────────────────────────────────
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)
    scaler    = GradScaler(device=DEVICE.type)

    # ── MLflow ────────────────────────────────────────────────────────────────
    # Use a local path to avoid issues with spaces in Windows usernames
    mlflow_dir = ROOT / "mlruns"
    mlflow_dir.mkdir(exist_ok=True)
    mlflow.set_tracking_uri(mlflow_dir.as_uri())
    mlflow.set_experiment("certiproof_forgery_detection")

    with mlflow.start_run(run_name=f"efficientnet_b4_{int(time.time())}"):
        mlflow.log_params({
            "model":        MODEL_CONFIG["model_name"],
            "epochs":       MODEL_CONFIG["epochs"],
            "batch_size":   MODEL_CONFIG["batch_size"],
            "lr":           MODEL_CONFIG["learning_rate"],
            "dropout":      MODEL_CONFIG["dropout"],
            "image_size":   f"{MODEL_CONFIG['image_width']}x{MODEL_CONFIG['image_height']}",
            "device":       str(DEVICE),
            "train_samples": len(train_ds),
            "val_samples":   len(val_ds),
            "test_samples":  len(test_ds),
            "n_genuine":     labels.count(1),
            "n_fake":        labels.count(0),
        })

        best_val_auc: float = 0.0
        epochs = MODEL_CONFIG["epochs"]

        for epoch in range(1, epochs + 1):
            t0 = time.time()

            train_loss             = train_one_epoch(model, train_loader, optimiser, criterion, scaler, epoch)
            val_loss, val_metrics  = validate(model, val_loader, criterion, epoch, "val")

            scheduler.step()
            current_lr  = scheduler.get_last_lr()[0]
            epoch_time  = time.time() - t0

            logger.info(
                "Epoch %03d/%03d | train_loss=%.4f  val_loss=%.4f  "
                "acc=%.4f  auc=%.4f  f1=%.4f  lr=%.2e  time=%.1fs",
                epoch, epochs, train_loss, val_loss,
                val_metrics["accuracy"], val_metrics["roc_auc"], val_metrics["f1"],
                current_lr, epoch_time,
            )

            mlflow.log_metrics({
                "train_loss":    train_loss,
                "val_loss":      val_loss,
                "val_accuracy":  val_metrics["accuracy"],
                "val_precision": val_metrics["precision"],
                "val_recall":    val_metrics["recall"],
                "val_f1":        val_metrics["f1"],
                "val_auc":       val_metrics["roc_auc"],
                "learning_rate": current_lr,
            }, step=epoch)

            # ── Save best checkpoint ──────────────────────────────────────────
            val_auc = val_metrics["roc_auc"]
            if not np.isnan(val_auc) and val_auc > best_val_auc:
                best_val_auc = val_auc
                torch.save({
                    "epoch":              epoch,
                    "model_state_dict":   model.state_dict(),
                    "optimiser_state_dict": optimiser.state_dict(),
                    "val_auc":            best_val_auc,
                    "val_metrics":        val_metrics,
                    "model_config":       MODEL_CONFIG,
                }, BEST_MODEL_PATH)
                logger.info("  ✓ Best model saved  (val_auc=%.4f)", best_val_auc)
                mlflow.log_metric("best_val_auc", best_val_auc, step=epoch)

        # ── Final test evaluation ─────────────────────────────────────────────
        logger.info("Loading best checkpoint for test evaluation…")
        if BEST_MODEL_PATH.exists():
            ckpt = torch.load(BEST_MODEL_PATH, map_location=DEVICE, weights_only=True)
            model.load_state_dict(ckpt["model_state_dict"])
            logger.info("  Loaded epoch %d  (val_auc=%.4f)", ckpt["epoch"], ckpt["val_auc"])

        test_loss, test_metrics = validate(model, test_loader, criterion, epoch=0, split_name="test")

        logger.info("═" * 60)
        logger.info("TEST RESULTS")
        logger.info("  loss      : %.4f", test_loss)
        logger.info("  accuracy  : %.4f", test_metrics["accuracy"])
        logger.info("  precision : %.4f", test_metrics["precision"])
        logger.info("  recall    : %.4f", test_metrics["recall"])
        logger.info("  f1        : %.4f", test_metrics["f1"])
        logger.info("  roc_auc   : %.4f", test_metrics["roc_auc"])
        logger.info("═" * 60)

        mlflow.log_metrics({
            "test_loss":      test_loss,
            "test_accuracy":  test_metrics["accuracy"],
            "test_precision": test_metrics["precision"],
            "test_recall":    test_metrics["recall"],
            "test_f1":        test_metrics["f1"],
            "test_auc":       test_metrics["roc_auc"],
        })

        if BEST_MODEL_PATH.exists():
            mlflow.log_artifact(str(BEST_MODEL_PATH), artifact_path="checkpoints")

    logger.info("Training complete. Best model → %s", BEST_MODEL_PATH)


if __name__ == "__main__":
    train()
