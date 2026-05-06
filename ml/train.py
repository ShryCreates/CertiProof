"""
CertValidator — Training Pipeline
===================================
Full training loop for the 6-channel EfficientNet-B4 forgery detector.

Dataset layout expected:
    training_data/
        genuine/   ← label 1
        fake/      ← label 0

Features
--------
- Train / Val / Test split  : 70 / 15 / 15
- Augmentations             : HorizontalFlip, Rotation, ColorJitter, RandomErasing
- Optimiser                 : AdamW (lr=3e-4)
- Scheduler                 : CosineAnnealingLR
- Loss                      : CrossEntropyLoss
- Mixed precision           : torch.cuda.amp
- Metrics                   : accuracy, precision, recall, F1, ROC-AUC
- Experiment tracking       : MLflow
- Checkpointing             : best model by validation AUC → checkpoints/best_model.pth
"""

import logging
import os
import time
from pathlib import Path
from typing import Optional

import mlflow
import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset, random_split
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

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("train")

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
DATA_DIR = ROOT / "training_data"
GENUINE_DIR = DATA_DIR / "genuine"
FAKE_DIR = DATA_DIR / "fake"
CHECKPOINT_DIR = ROOT / "checkpoints"
CHECKPOINT_DIR.mkdir(exist_ok=True)
BEST_MODEL_PATH = CHECKPOINT_DIR / "best_model.pth"

# ── Device ────────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info("Using device: %s", DEVICE)


# ═══════════════════════════════════════════════════════════════════════════════
# DATASET
# ═══════════════════════════════════════════════════════════════════════════════

class CertificateDataset(Dataset):
    """
    Loads certificate images from genuine/ and fake/ directories.

    Each sample is a 6-channel tensor: 3 RGB channels + 3 ELA channels.
    Labels: genuine = 1, fake = 0.

    Parameters
    ----------
    image_paths : list[Path]
        Absolute paths to image files.
    labels : list[int]
        Corresponding labels (0 = fake, 1 = genuine).
    augment : bool
        Whether to apply training augmentations.
    """

    # Augmentation pipeline (applied to the RGB image as a PIL Image)
    _augment_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=5, fill=255),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
        transforms.ToTensor(),                          # → (3, H, W) float [0,1]
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.1), ratio=(0.3, 3.3), value=1.0),
    ])

    # Inference-only transform (no augmentation)
    _base_transform = transforms.Compose([
        transforms.ToTensor(),                          # → (3, H, W) float [0,1]
    ])

    def __init__(
        self,
        image_paths: list[Path],
        labels: list[int],
        augment: bool = False,
    ) -> None:
        assert len(image_paths) == len(labels), "Paths and labels must have equal length"
        self.image_paths = image_paths
        self.labels = labels
        self.augment = augment

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        path = self.image_paths[idx]
        label = self.labels[idx]

        # ── Preprocess (deskew, CLAHE, pad) ──────────────────────────────────
        rgb_float = preprocess(path)                    # (H, W, 3) float32 [0,1]

        # ── ELA ───────────────────────────────────────────────────────────────
        ela_uint8 = generate_ela(path)                  # (orig_H, orig_W, 3) uint8
        import cv2
        ela_resized = cv2.resize(
            ela_uint8,
            (MODEL_CONFIG["image_width"], MODEL_CONFIG["image_height"]),
            interpolation=cv2.INTER_AREA,
        )
        ela_float = ela_resized.astype(np.float32) / 255.0  # (H, W, 3) [0,1]

        # ── Augmentation (RGB only, as PIL) ───────────────────────────────────
        rgb_pil = Image.fromarray((rgb_float * 255).astype(np.uint8))

        if self.augment:
            rgb_tensor = self._augment_transform(rgb_pil)   # (3, H, W)
        else:
            rgb_tensor = self._base_transform(rgb_pil)      # (3, H, W)

        # ── ELA tensor ────────────────────────────────────────────────────────
        ela_tensor = torch.from_numpy(ela_float).permute(2, 0, 1).float()  # (3, H, W)

        # ── Concatenate → 6-channel tensor ────────────────────────────────────
        combined = torch.cat([rgb_tensor, ela_tensor], dim=0)  # (6, H, W)

        return combined, label


def load_dataset() -> tuple[list[Path], list[int]]:
    """
    Scan genuine/ and fake/ directories and return (paths, labels).

    Supported extensions: .jpg, .jpeg, .png, .bmp, .tiff
    """
    supported = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
    paths: list[Path] = []
    labels: list[int] = []

    for label, directory in [(1, GENUINE_DIR), (0, FAKE_DIR)]:
        if not directory.exists():
            logger.warning("Directory not found: %s — skipping", directory)
            continue
        found = [p for p in directory.iterdir() if p.suffix.lower() in supported]
        logger.info("Found %d images in %s", len(found), directory)
        paths.extend(found)
        labels.extend([label] * len(found))

    if not paths:
        raise RuntimeError(
            f"No images found in {DATA_DIR}. "
            "Add images to training_data/genuine/ and training_data/fake/"
        )

    logger.info("Total dataset: %d images (%d genuine, %d fake)",
                len(paths), labels.count(1), labels.count(0))
    return paths, labels


def make_splits(
    paths: list[Path],
    labels: list[int],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[CertificateDataset, CertificateDataset, CertificateDataset]:
    """
    Split dataset into train / val / test subsets.

    Returns
    -------
    train_ds, val_ds, test_ds : CertificateDataset
    """
    n = len(paths)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_test = n - n_train - n_val

    rng = torch.Generator().manual_seed(seed)
    indices = torch.randperm(n, generator=rng).tolist()

    train_idx = indices[:n_train]
    val_idx = indices[n_train : n_train + n_val]
    test_idx = indices[n_train + n_val :]

    def _subset(idx_list: list[int], augment: bool) -> CertificateDataset:
        return CertificateDataset(
            image_paths=[paths[i] for i in idx_list],
            labels=[labels[i] for i in idx_list],
            augment=augment,
        )

    train_ds = _subset(train_idx, augment=True)
    val_ds = _subset(val_idx, augment=False)
    test_ds = _subset(test_idx, augment=False)

    logger.info("Split → train=%d  val=%d  test=%d", len(train_ds), len(val_ds), len(test_ds))
    return train_ds, val_ds, test_ds


# ═══════════════════════════════════════════════════════════════════════════════
# METRICS
# ═══════════════════════════════════════════════════════════════════════════════

def compute_metrics(
    all_labels: list[int],
    all_preds: list[int],
    all_probs: list[float],
) -> dict[str, float]:
    """
    Compute classification metrics.

    Parameters
    ----------
    all_labels : list[int]   Ground-truth labels
    all_preds  : list[int]   Predicted class indices
    all_probs  : list[float] Predicted probability for class 1 (genuine)

    Returns
    -------
    dict with accuracy, precision, recall, f1, roc_auc
    """
    # Guard against single-class batches (can happen with tiny datasets)
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = float("nan")

    return {
        "accuracy": accuracy_score(all_labels, all_preds),
        "precision": precision_score(all_labels, all_preds, zero_division=0),
        "recall": recall_score(all_labels, all_preds, zero_division=0),
        "f1": f1_score(all_labels, all_preds, zero_division=0),
        "roc_auc": auc,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# TRAINING LOOP
# ═══════════════════════════════════════════════════════════════════════════════

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimiser: torch.optim.Optimizer,
    criterion: nn.Module,
    scaler: GradScaler,
    epoch: int,
) -> float:
    """
    Run one training epoch with mixed-precision.

    Returns
    -------
    float : mean training loss for the epoch
    """
    model.train()
    total_loss = 0.0
    n_batches = len(loader)

    pbar = tqdm(loader, desc=f"Epoch {epoch:03d} [train]", leave=False, unit="batch")

    for batch_idx, (inputs, targets) in enumerate(pbar):
        inputs = inputs.to(DEVICE, non_blocking=True)
        targets = targets.to(DEVICE, non_blocking=True)

        optimiser.zero_grad(set_to_none=True)

        # Mixed-precision forward pass
        with autocast(device_type=DEVICE.type, enabled=DEVICE.type == "cuda"):
            logits = model(inputs)
            loss = criterion(logits, targets)

        # Backward + optimiser step via scaler
        scaler.scale(loss).backward()
        scaler.unscale_(optimiser)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimiser)
        scaler.update()

        total_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    mean_loss = total_loss / n_batches
    return mean_loss


# ═══════════════════════════════════════════════════════════════════════════════
# VALIDATION LOOP
# ═══════════════════════════════════════════════════════════════════════════════

def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    epoch: int,
    split_name: str = "val",
) -> tuple[float, dict[str, float]]:
    """
    Evaluate model on a validation or test split.

    Returns
    -------
    (mean_loss, metrics_dict)
    """
    model.eval()
    total_loss = 0.0
    all_labels: list[int] = []
    all_preds: list[int] = []
    all_probs: list[float] = []

    pbar = tqdm(loader, desc=f"Epoch {epoch:03d} [{split_name}]", leave=False, unit="batch")

    with torch.no_grad():
        for inputs, targets in pbar:
            inputs = inputs.to(DEVICE, non_blocking=True)
            targets = targets.to(DEVICE, non_blocking=True)

            with autocast(device_type=DEVICE.type, enabled=DEVICE.type == "cuda"):
                logits = model(inputs)
                loss = criterion(logits, targets)

            total_loss += loss.item()

            probs = torch.softmax(logits, dim=1)[:, 1]  # P(genuine)
            preds = torch.argmax(logits, dim=1)

            all_labels.extend(targets.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())
            all_probs.extend(probs.cpu().tolist())

    mean_loss = total_loss / len(loader)
    metrics = compute_metrics(all_labels, all_preds, all_probs)
    return mean_loss, metrics


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN TRAINING ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def train() -> None:
    """
    Full training pipeline:
      1. Load and split dataset
      2. Build model
      3. Configure optimiser, scheduler, loss, scaler
      4. Train for N epochs with MLflow logging
      5. Save best checkpoint by validation AUC
      6. Final evaluation on test set
    """
    logger.info("═" * 60)
    logger.info("CertValidator — Training Pipeline")
    logger.info("═" * 60)

    # ── Dataset ───────────────────────────────────────────────────────────────
    paths, labels = load_dataset()
    train_ds, val_ds, test_ds = make_splits(paths, labels)

    train_loader = DataLoader(
        train_ds,
        batch_size=MODEL_CONFIG["batch_size"],
        shuffle=True,
        num_workers=min(4, os.cpu_count() or 1),
        pin_memory=DEVICE.type == "cuda",
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=MODEL_CONFIG["batch_size"],
        shuffle=False,
        num_workers=min(4, os.cpu_count() or 1),
        pin_memory=DEVICE.type == "cuda",
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=MODEL_CONFIG["batch_size"],
        shuffle=False,
        num_workers=min(4, os.cpu_count() or 1),
        pin_memory=DEVICE.type == "cuda",
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = build_model().to(DEVICE)
    model.train()

    # ── Optimiser & Scheduler ─────────────────────────────────────────────────
    optimiser = torch.optim.AdamW(
        model.parameters(),
        lr=MODEL_CONFIG["learning_rate"],
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser,
        T_max=MODEL_CONFIG["epochs"],
        eta_min=1e-6,
    )

    # ── Loss & Scaler ─────────────────────────────────────────────────────────
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    scaler = GradScaler(enabled=DEVICE.type == "cuda")

    # ── MLflow experiment ─────────────────────────────────────────────────────
    mlflow.set_experiment("certvalidator_forgery_detection")

    with mlflow.start_run(run_name=f"efficientnet_b4_{int(time.time())}"):
        # Log hyperparameters
        mlflow.log_params({
            "model": MODEL_CONFIG["model_name"],
            "epochs": MODEL_CONFIG["epochs"],
            "batch_size": MODEL_CONFIG["batch_size"],
            "learning_rate": MODEL_CONFIG["learning_rate"],
            "dropout": MODEL_CONFIG["dropout"],
            "image_size": f"{MODEL_CONFIG['image_width']}x{MODEL_CONFIG['image_height']}",
            "device": str(DEVICE),
            "train_samples": len(train_ds),
            "val_samples": len(val_ds),
            "test_samples": len(test_ds),
        })

        best_val_auc: float = 0.0
        epochs = MODEL_CONFIG["epochs"]

        # ── Training loop ─────────────────────────────────────────────────────
        for epoch in range(1, epochs + 1):
            epoch_start = time.time()

            # Train
            train_loss = train_one_epoch(
                model, train_loader, optimiser, criterion, scaler, epoch
            )

            # Validate
            val_loss, val_metrics = validate(model, val_loader, criterion, epoch, "val")

            # Step scheduler
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]

            epoch_time = time.time() - epoch_start

            # ── Console summary ───────────────────────────────────────────────
            logger.info(
                "Epoch %03d/%03d | "
                "train_loss=%.4f  val_loss=%.4f  "
                "val_acc=%.4f  val_auc=%.4f  val_f1=%.4f  "
                "lr=%.2e  time=%.1fs",
                epoch, epochs,
                train_loss, val_loss,
                val_metrics["accuracy"], val_metrics["roc_auc"], val_metrics["f1"],
                current_lr, epoch_time,
            )

            # ── MLflow logging ────────────────────────────────────────────────
            mlflow.log_metrics(
                {
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_accuracy": val_metrics["accuracy"],
                    "val_precision": val_metrics["precision"],
                    "val_recall": val_metrics["recall"],
                    "val_f1": val_metrics["f1"],
                    "val_auc": val_metrics["roc_auc"],
                    "learning_rate": current_lr,
                },
                step=epoch,
            )

            # ── Checkpoint: save best by val AUC ─────────────────────────────
            val_auc = val_metrics["roc_auc"]
            if not np.isnan(val_auc) and val_auc > best_val_auc:
                best_val_auc = val_auc
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimiser_state_dict": optimiser.state_dict(),
                        "val_auc": best_val_auc,
                        "val_metrics": val_metrics,
                        "model_config": MODEL_CONFIG,
                    },
                    BEST_MODEL_PATH,
                )
                logger.info("  ✓ New best model saved (val_auc=%.4f)", best_val_auc)
                mlflow.log_metric("best_val_auc", best_val_auc, step=epoch)

        # ── Final test evaluation ─────────────────────────────────────────────
        logger.info("Loading best checkpoint for test evaluation…")
        if BEST_MODEL_PATH.exists():
            ckpt = torch.load(BEST_MODEL_PATH, map_location=DEVICE)
            model.load_state_dict(ckpt["model_state_dict"])
            logger.info("  Loaded checkpoint from epoch %d (val_auc=%.4f)",
                        ckpt["epoch"], ckpt["val_auc"])

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

        mlflow.log_metrics(
            {
                "test_loss": test_loss,
                "test_accuracy": test_metrics["accuracy"],
                "test_precision": test_metrics["precision"],
                "test_recall": test_metrics["recall"],
                "test_f1": test_metrics["f1"],
                "test_auc": test_metrics["roc_auc"],
            }
        )

        # Log the best model artifact to MLflow
        if BEST_MODEL_PATH.exists():
            mlflow.log_artifact(str(BEST_MODEL_PATH), artifact_path="checkpoints")

    logger.info("Training complete. Best model → %s", BEST_MODEL_PATH)


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    train()
