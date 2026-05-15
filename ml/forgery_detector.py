"""
CertiProof — Forgery Detection Module
=========================================
Production-grade certificate forgery detection using:
  - EfficientNet-B4 (timm) as the backbone
  - Error Level Analysis (ELA) as a second input channel
  - GradCAM for visual explainability
  - OpenCV for image preprocessing and contour detection

Input:  certificate image (JPG / PNG / PDF-rendered page)
Output: forgery_score, gradcam_heatmap, tamper_regions
"""

import os
import logging
import tempfile
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("forgery_detector")

# ── Model configuration ───────────────────────────────────────────────────────
MODEL_CONFIG: dict = {
    "image_width": 224,   # reduced from 512 for CPU training speed
    "image_height": 224,  # reduced from 724 for CPU training speed
    "batch_size": 4,      # reduced for CPU memory
    "epochs": 15,         # reduced from 30; increase when GPU is available
    "learning_rate": 3e-4,
    "dropout": 0.3,
    "model_name": "efficientnet_b4",
    "num_classes": 2,
}

# Convenience aliases
IMG_W = MODEL_CONFIG["image_width"]   # 224
IMG_H = MODEL_CONFIG["image_height"]  # 224


# ═══════════════════════════════════════════════════════════════════════════════
# 1. PREPROCESSING
# ═══════════════════════════════════════════════════════════════════════════════

def preprocess(image_path: str | Path) -> np.ndarray:
    """
    Full preprocessing pipeline for a certificate image.

    Steps
    -----
    1. Load with OpenCV (BGR → RGB)
    2. Convert to grayscale for skew detection
    3. Detect skew angle via Hough line transform
    4. Rotate image to deskew
    5. Crop 2 % borders from all sides
    6. Apply CLAHE histogram equalisation (per channel)
    7. Resize while preserving aspect ratio
    8. Pad with white background to final size (IMG_W × IMG_H)
    9. Normalise to float32 in [0, 1]

    Parameters
    ----------
    image_path : str | Path
        Path to the certificate image file.

    Returns
    -------
    np.ndarray
        Shape (IMG_H, IMG_W, 3), dtype float32, values in [0, 1].
    """
    image_path = str(image_path)
    logger.debug("preprocess: loading %s", image_path)

    # ── 1. Load ───────────────────────────────────────────────────────────────
    bgr = cv2.imread(image_path)
    if bgr is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    # ── 2. Grayscale for skew detection ──────────────────────────────────────
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # ── 3. Detect skew angle via Hough lines ─────────────────────────────────
    skew_angle = _detect_skew(gray)
    logger.debug("preprocess: detected skew angle = %.2f°", skew_angle)

    # ── 4. Deskew ─────────────────────────────────────────────────────────────
    if abs(skew_angle) > 0.5:  # only rotate if skew is meaningful
        rgb = _rotate_image(rgb, skew_angle)

    # ── 5. Crop 2 % borders ───────────────────────────────────────────────────
    h, w = rgb.shape[:2]
    crop_y = max(1, int(h * 0.02))
    crop_x = max(1, int(w * 0.02))
    rgb = rgb[crop_y : h - crop_y, crop_x : w - crop_x]

    # ── 6. CLAHE per channel ──────────────────────────────────────────────────
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    channels = [clahe.apply(rgb[:, :, c]) for c in range(3)]
    rgb = np.stack(channels, axis=2)

    # ── 7. Resize preserving aspect ratio ────────────────────────────────────
    rgb = _resize_keep_aspect(rgb, IMG_W, IMG_H)

    # ── 8. Pad to exact target size with white background ────────────────────
    rgb = _pad_to_size(rgb, IMG_W, IMG_H, fill=255)

    # ── 9. Normalise to float32 [0, 1] ───────────────────────────────────────
    rgb = rgb.astype(np.float32) / 255.0

    logger.debug("preprocess: output shape %s", rgb.shape)
    return rgb  # (IMG_H, IMG_W, 3)


def _detect_skew(gray: np.ndarray) -> float:
    """
    Estimate document skew angle using Hough line transform.

    Returns the median angle (in degrees) of detected lines,
    clamped to [-45, 45] to avoid over-rotation.
    """
    # Edge detection
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # Probabilistic Hough transform
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=100,
        minLineLength=100,
        maxLineGap=10,
    )

    if lines is None or len(lines) == 0:
        return 0.0

    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 - x1 == 0:
            continue  # vertical line — skip
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        # Keep only near-horizontal lines (document text lines)
        if -45 < angle < 45:
            angles.append(angle)

    if not angles:
        return 0.0

    return float(np.median(angles))


def _rotate_image(image: np.ndarray, angle: float) -> np.ndarray:
    """Rotate image by `angle` degrees around its centre, filling with white."""
    h, w = image.shape[:2]
    cx, cy = w // 2, h // 2
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    rotated = cv2.warpAffine(
        image, M, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )
    return rotated


def _resize_keep_aspect(image: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    """Resize image so it fits within (target_w, target_h) without distortion."""
    h, w = image.shape[:2]
    scale = min(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)


def _pad_to_size(
    image: np.ndarray, target_w: int, target_h: int, fill: int = 255
) -> np.ndarray:
    """Pad image to exact (target_h, target_w) with a solid fill colour."""
    h, w = image.shape[:2]
    pad_top = (target_h - h) // 2
    pad_bottom = target_h - h - pad_top
    pad_left = (target_w - w) // 2
    pad_right = target_w - w - pad_left
    return cv2.copyMakeBorder(
        image,
        pad_top, pad_bottom, pad_left, pad_right,
        cv2.BORDER_CONSTANT,
        value=(fill, fill, fill),
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 2. ERROR LEVEL ANALYSIS (ELA)
# ═══════════════════════════════════════════════════════════════════════════════

def generate_ela(image_path: str | Path, quality: int = 90) -> np.ndarray:
    """
    Generate an Error Level Analysis (ELA) image.

    ELA reveals regions that have been re-saved at a different compression
    level — a strong indicator of localised tampering.

    Steps
    -----
    1. Open original image with Pillow
    2. Re-save as JPEG at `quality` into a temp file
    3. Reload the compressed version
    4. Compute absolute pixel difference (original − compressed)
    5. Amplify by 10× to make subtle differences visible
    6. Normalise to [0, 255] uint8

    Parameters
    ----------
    image_path : str | Path
        Path to the certificate image.
    quality : int
        JPEG re-save quality (default 90).

    Returns
    -------
    np.ndarray
        ELA image, shape (H, W, 3), dtype uint8.
    """
    image_path = str(image_path)
    logger.debug("generate_ela: processing %s at quality=%d", image_path, quality)

    # ── 1. Open original ──────────────────────────────────────────────────────
    original = Image.open(image_path).convert("RGB")
    orig_arr = np.array(original, dtype=np.float32)

    # ── 2. Re-save at reduced quality ─────────────────────────────────────────
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        original.save(tmp_path, "JPEG", quality=quality)

        # ── 3. Reload compressed version ─────────────────────────────────────
        compressed = Image.open(tmp_path).convert("RGB")
        comp_arr = np.array(compressed, dtype=np.float32)
    finally:
        os.unlink(tmp_path)

    # ── 4. Absolute difference ────────────────────────────────────────────────
    diff = np.abs(orig_arr - comp_arr)

    # ── 5. Amplify ────────────────────────────────────────────────────────────
    diff *= 10.0

    # ── 6. Normalise to uint8 ─────────────────────────────────────────────────
    diff = np.clip(diff, 0, 255).astype(np.uint8)

    logger.debug("generate_ela: output shape %s", diff.shape)
    return diff  # (H, W, 3)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. MODEL ARCHITECTURE
# ═══════════════════════════════════════════════════════════════════════════════

def build_model() -> nn.Module:
    """
    Build the 6-channel EfficientNet-B4 forgery detection model.

    Architecture
    ------------
    - Backbone : EfficientNet-B4 (pretrained on ImageNet via timm)
    - Input    : 6-channel tensor (3 RGB + 3 ELA channels)
    - Head     : Dropout → Linear(1792, 512) → GELU → Dropout → Linear(512, 2)

    The first conv layer is replaced to accept 6 channels.
    Pretrained RGB weights are averaged and copied into both halves of the
    new 6-channel weight tensor so the model starts from a sensible point.

    Returns
    -------
    nn.Module
        Model in eval mode, ready for inference or fine-tuning.
    """
    logger.info("build_model: loading EfficientNet-B4 backbone (pretrained=True)")

    # ── Load pretrained backbone ──────────────────────────────────────────────
    backbone = timm.create_model(
        MODEL_CONFIG["model_name"],
        pretrained=True,
        num_classes=0,   # remove default classifier
        global_pool="avg",
    )

    # ── Patch first conv to accept 6 channels ────────────────────────────────
    old_conv: nn.Conv2d = backbone.conv_stem  # (32, 3, 3, 3) for EfficientNet-B4
    old_weight = old_conv.weight.data         # shape: (out_ch, 3, kH, kW)

    new_conv = nn.Conv2d(
        in_channels=6,
        out_channels=old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=old_conv.bias is not None,
    )

    # Initialise new conv: copy pretrained weights into both 3-channel halves.
    # Divide by 2 so the initial activation magnitude is preserved.
    with torch.no_grad():
        new_conv.weight[:, :3, :, :] = old_weight * 0.5
        new_conv.weight[:, 3:, :, :] = old_weight * 0.5
        if old_conv.bias is not None:
            new_conv.bias.data = old_conv.bias.data.clone()

    backbone.conv_stem = new_conv
    logger.info("build_model: patched conv_stem to 6-channel input")

    # ── Custom classification head ────────────────────────────────────────────
    num_features = backbone.num_features  # 1792 for EfficientNet-B4

    head = nn.Sequential(
        nn.Dropout(p=MODEL_CONFIG["dropout"]),
        nn.Linear(num_features, 512),
        nn.GELU(),
        nn.Dropout(p=MODEL_CONFIG["dropout"]),
        nn.Linear(512, MODEL_CONFIG["num_classes"]),
    )

    # ── Assemble full model ───────────────────────────────────────────────────
    model = _ForgeryDetector(backbone=backbone, head=head)
    model.eval()

    logger.info(
        "build_model: model ready — %d trainable parameters",
        sum(p.numel() for p in model.parameters() if p.requires_grad),
    )
    return model


class _ForgeryDetector(nn.Module):
    """Thin wrapper combining the EfficientNet-B4 backbone with the custom head."""

    def __init__(self, backbone: nn.Module, head: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Shape (B, 6, H, W) — concatenated RGB + ELA channels.

        Returns
        -------
        torch.Tensor
            Shape (B, 2) — raw logits for [fake, genuine].
        """
        features = self.backbone(x)   # (B, 1792)
        return self.head(features)    # (B, 2)


# ═══════════════════════════════════════════════════════════════════════════════
# 4. GRADCAM
# ═══════════════════════════════════════════════════════════════════════════════

def generate_gradcam(
    model: nn.Module,
    tensor: torch.Tensor,
    target_class: int,
    original_image: np.ndarray,
) -> np.ndarray:
    """
    Generate a GradCAM visualisation for the given input tensor.

    The activation map highlights image regions that most influenced the
    model's prediction for `target_class`.

    Parameters
    ----------
    model : nn.Module
        The _ForgeryDetector model (or any model with a `.backbone` attribute
        whose last feature block is accessible as `backbone.blocks[-1]`).
    tensor : torch.Tensor
        Shape (1, 6, H, W) — the 6-channel input tensor.
    target_class : int
        Class index to explain (0 = fake, 1 = genuine).
    original_image : np.ndarray
        The original RGB image (uint8, H×W×3) used for overlay.

    Returns
    -------
    np.ndarray
        Heatmap overlaid on `original_image`, shape (H, W, 3), dtype uint8.
    """
    model.eval()

    # Storage for hooks
    activations: list[torch.Tensor] = []
    gradients: list[torch.Tensor] = []

    # ── Identify target layer (last feature block of EfficientNet-B4) ─────────
    # timm's EfficientNet stores feature blocks in backbone.blocks
    target_layer: nn.Module = model.backbone.blocks[-1]

    # ── Register hooks ────────────────────────────────────────────────────────
    def _save_activation(module, input, output):  # noqa: ARG001
        activations.append(output.detach())

    def _save_gradient(module, grad_input, grad_output):  # noqa: ARG001
        gradients.append(grad_output[0].detach())

    fwd_hook = target_layer.register_forward_hook(_save_activation)
    bwd_hook = target_layer.register_full_backward_hook(_save_gradient)

    try:
        # ── Forward pass ──────────────────────────────────────────────────────
        tensor = tensor.requires_grad_(True)
        logits = model(tensor)                    # (1, 2)

        # ── Backward pass for target class ────────────────────────────────────
        model.zero_grad()
        score = logits[0, target_class]
        score.backward()

        # ── Compute GradCAM ───────────────────────────────────────────────────
        act = activations[0]   # (1, C, h, w)
        grad = gradients[0]    # (1, C, h, w)

        # Global average pool the gradients → channel weights
        weights = grad.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)

        # Weighted sum of activation maps
        cam = (weights * act).sum(dim=1, keepdim=True)  # (1, 1, h, w)
        cam = F.relu(cam)                                # keep positive contributions

        # ── Resize to original image dimensions ───────────────────────────────
        orig_h, orig_w = original_image.shape[:2]
        cam_np = cam.squeeze().cpu().numpy()             # (h, w)
        cam_resized = cv2.resize(cam_np, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

        # ── Normalise to uint8 [0, 255] ───────────────────────────────────────
        cam_min, cam_max = cam_resized.min(), cam_resized.max()
        if cam_max - cam_min > 1e-8:
            cam_norm = (cam_resized - cam_min) / (cam_max - cam_min)
        else:
            cam_norm = np.zeros_like(cam_resized)

        cam_uint8 = (cam_norm * 255).astype(np.uint8)

        # ── Apply JET colormap ────────────────────────────────────────────────
        heatmap_bgr = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
        heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

        # ── Overlay on original image (alpha blend) ───────────────────────────
        # Ensure original_image is uint8
        if original_image.dtype != np.uint8:
            base = (original_image * 255).clip(0, 255).astype(np.uint8)
        else:
            base = original_image.copy()

        overlay = cv2.addWeighted(base, 0.6, heatmap_rgb, 0.4, 0)

    finally:
        fwd_hook.remove()
        bwd_hook.remove()

    return overlay  # (H, W, 3) uint8


# ═══════════════════════════════════════════════════════════════════════════════
# 5. PREDICT
# ═══════════════════════════════════════════════════════════════════════════════

def predict(image_path: str | Path, model: nn.Module) -> dict:
    """
    Full inference pipeline for a single certificate image.

    Steps
    -----
    1. Preprocess image → float32 ndarray (H, W, 3)
    2. Generate ELA image → uint8 ndarray (H, W, 3)
    3. Resize ELA to match preprocessed dimensions
    4. Concatenate RGB + ELA → 6-channel tensor (1, 6, H, W)
    5. Forward pass → softmax probabilities
    6. Compute forgery_score = 1 − P(genuine)
    7. Generate GradCAM heatmap for the predicted class
    8. Detect suspicious regions via contour analysis on the ELA map

    Parameters
    ----------
    image_path : str | Path
        Path to the certificate image.
    model : nn.Module
        Loaded _ForgeryDetector model (eval mode).

    Returns
    -------
    dict with keys:
        forgery_score  : float in [0, 1]  — higher = more likely forged
        gradcam_heatmap: np.ndarray (H, W, 3) uint8
        tamper_regions : list of dicts {x, y, w, h, confidence}
    """
    image_path = Path(image_path)
    logger.info("predict: running inference on %s", image_path.name)

    device = next(model.parameters()).device

    # ── 1. Preprocess ─────────────────────────────────────────────────────────
    rgb_float = preprocess(image_path)          # (IMG_H, IMG_W, 3) float32 [0,1]
    orig_uint8 = (rgb_float * 255).astype(np.uint8)

    # ── 2. ELA ────────────────────────────────────────────────────────────────
    ela_uint8 = generate_ela(image_path)        # (orig_H, orig_W, 3) uint8

    # ── 3. Resize ELA to match preprocessed size ─────────────────────────────
    ela_resized = cv2.resize(ela_uint8, (IMG_W, IMG_H), interpolation=cv2.INTER_AREA)
    ela_float = ela_resized.astype(np.float32) / 255.0  # (IMG_H, IMG_W, 3) [0,1]

    # ── 4. Build 6-channel tensor ─────────────────────────────────────────────
    # Stack along channel axis: (H, W, 6) → (1, 6, H, W)
    combined = np.concatenate([rgb_float, ela_float], axis=2)  # (H, W, 6)
    tensor = torch.from_numpy(combined).permute(2, 0, 1).unsqueeze(0).float()
    tensor = tensor.to(device)                                  # (1, 6, H, W)

    # ── 5. Forward pass ───────────────────────────────────────────────────────
    model.eval()
    with torch.no_grad():
        logits = model(tensor)                  # (1, 2)
        probs = F.softmax(logits, dim=1)        # (1, 2)

    prob_genuine = probs[0, 1].item()           # P(genuine)
    prob_fake = probs[0, 0].item()              # P(fake)

    # ── 6. Forgery score ──────────────────────────────────────────────────────
    forgery_score = float(1.0 - prob_genuine)   # high = likely forged

    # ── 7. GradCAM ────────────────────────────────────────────────────────────
    # Explain the predicted class
    predicted_class = int(torch.argmax(probs, dim=1).item())
    # Re-run with grad enabled (generate_gradcam needs backward pass)
    tensor_grad = tensor.clone().detach().requires_grad_(True)
    gradcam_overlay = generate_gradcam(model, tensor_grad, predicted_class, orig_uint8)

    # ── 8. Tamper region detection via ELA contours ───────────────────────────
    tamper_regions = _detect_tamper_regions(ela_resized, forgery_score)

    logger.info(
        "predict: forgery_score=%.4f  P(genuine)=%.4f  P(fake)=%.4f  regions=%d",
        forgery_score, prob_genuine, prob_fake, len(tamper_regions),
    )

    return {
        "forgery_score": forgery_score,
        "gradcam_heatmap": gradcam_overlay,
        "tamper_regions": tamper_regions,
    }


def _detect_tamper_regions(
    ela_image: np.ndarray,
    forgery_score: float,
    min_area: int = 500,
) -> list[dict]:
    """
    Detect suspicious regions in the ELA image using contour analysis.

    High-intensity regions in the ELA map correspond to areas that were
    re-compressed at a different quality level — a hallmark of tampering.

    Parameters
    ----------
    ela_image : np.ndarray
        ELA image (uint8, H×W×3) at the model's input resolution.
    forgery_score : float
        Overall forgery score; used to scale the detection threshold.
    min_area : int
        Minimum contour area (pixels²) to report as a region.

    Returns
    -------
    list of dicts, each with keys: x, y, w, h, confidence
    """
    # Convert to grayscale and threshold
    gray = cv2.cvtColor(ela_image, cv2.COLOR_RGB2GRAY)

    # Adaptive threshold: stricter when forgery_score is low
    threshold_val = max(30, int(255 * (1.0 - forgery_score) * 0.5))
    _, binary = cv2.threshold(gray, threshold_val, 255, cv2.THRESH_BINARY)

    # Morphological closing to merge nearby blobs
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # Find contours
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    regions = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue

        x, y, w, h = cv2.boundingRect(cnt)

        # Confidence: mean ELA intensity in the bounding box, normalised to [0,1]
        roi = gray[y : y + h, x : x + w]
        confidence = float(roi.mean() / 255.0)

        regions.append({
            "x": int(x),
            "y": int(y),
            "w": int(w),
            "h": int(h),
            "confidence": round(confidence, 4),
        })

    # Sort by confidence descending
    regions.sort(key=lambda r: r["confidence"], reverse=True)
    return regions


# ═══════════════════════════════════════════════════════════════════════════════
# EXAMPLE USAGE
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python forgery_detector.py <image_path> [checkpoint_path]")
        sys.exit(1)

    img_path = sys.argv[1]
    ckpt_path = sys.argv[2] if len(sys.argv) > 2 else None

    # Build model
    detector = build_model()

    # Optionally load trained weights
    if ckpt_path and Path(ckpt_path).exists():
        state = torch.load(ckpt_path, map_location="cpu")
        detector.load_state_dict(state["model_state_dict"])
        logger.info("Loaded checkpoint from %s", ckpt_path)
    else:
        logger.warning("No checkpoint provided — using random weights (for testing only)")

    detector.eval()

    # Run prediction
    result = predict(img_path, detector)

    print("\n── Prediction Result ──────────────────────────────")
    print(f"  Forgery Score : {result['forgery_score']:.4f}")
    print(f"  Tamper Regions: {len(result['tamper_regions'])}")
    for i, region in enumerate(result["tamper_regions"], 1):
        print(f"    [{i}] x={region['x']} y={region['y']} "
              f"w={region['w']} h={region['h']} "
              f"confidence={region['confidence']:.4f}")

    # Save GradCAM overlay
    out_path = Path(img_path).stem + "_gradcam.jpg"
    overlay_bgr = cv2.cvtColor(result["gradcam_heatmap"], cv2.COLOR_RGB2BGR)
    cv2.imwrite(out_path, overlay_bgr)
    print(f"\n  GradCAM saved → {out_path}")
