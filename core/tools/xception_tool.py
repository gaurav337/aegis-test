"""Face-Swap / Reenactment Detector using XceptionNet.
Leverages the timm library for the architecture and uses models
pre-trained on FaceForensics++.

Patch-Wise Consistency Scoring:
    Real phone selfies are processed uniformly by the ISP, producing low
    variance across face patches. Deepfakes have localized blending artifacts,
    producing high variance. We use patch std to dampen uniform false positives.
"""

import os
import time
import logging
from typing import Any, Dict, List, Tuple
import numpy as np
import cv2
from PIL import Image
import torch.nn.functional as F
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import timm

from core.base_tool import BaseForensicTool
from core.data_types import ToolResult
from core.config import AegisConfig
from utils.vram_manager import VRAMLifecycleManager

logger = logging.getLogger(__name__)


# ─── Patch Consistency Scoring Configuration ───
PATCH_GRID_SIZE = 3  # 3x3 grid = 9 overlapping patches
PATCH_COVER_RATIO = 0.65  # Each patch covers 65% of face width
PATCH_STD_THRESHOLD = 0.08  # Below this → uniform processing (suspect)
PATCH_DAMPENING_FACTOR = 0.45  # Max dampening multiplier for uniform scores
PATCH_ACTIVATION_SCORE = 0.40  # Only apply consistency check above this score


class XceptionTool(BaseForensicTool):
    """
    Detects face-swap and reenactment deepfakes using an XceptionNet backbone.
    Uses patch-wise consistency scoring to reject phone ISP false positives.
    """

    def __init__(self):
        super().__init__()
        self._tool_name = "run_xception"
        self.requires_gpu = True

        # Load from config
        self.config = AegisConfig()
        try:
            from utils.thresholds import XCEPTION_FAKE_THRESHOLD

            self.fake_threshold = XCEPTION_FAKE_THRESHOLD
        except ImportError:
            self.fake_threshold = 0.50

        self.xception_cfg = self.config.xception

        # ImageNet normalization for XceptionNet
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        self._weights_loaded_ok = False

    @property
    def tool_name(self) -> str:
        return self._tool_name

    def setup(self):
        logger.info("XceptionTool setup complete.")
        return True

    def _remap_keys(self, ckpt: Dict[str, Any]) -> Dict[str, Any]:
        """
        Remaps weights from FaceForensics++ (HongguLiu) format to timm format.
        - Strips 'module.' or 'model.' prefixes.
        - Renames 'last_linear' to 'fc' to match timm's classifier name.
        """
        remapped = {}
        for k, v in ckpt.items():
            new_k = k
            if new_k.startswith("module."):
                new_k = new_k.replace("module.", "", 1)
            elif new_k.startswith("model."):
                new_k = new_k.replace("model.", "", 1)

            if "last_linear" in new_k:
                new_k = new_k.replace("last_linear", "fc")
            elif "classifier" in new_k:
                new_k = new_k.replace("classifier", "fc")

            remapped[new_k] = v
        return remapped

    def _load_model(self) -> nn.Module:
        logger.info("Loading XceptionNet model...")
        model = timm.create_model("xception", pretrained=False, num_classes=2)

        weight_path = getattr(
            self.config.models,
            "xception_weights",
            "models/xception/xception_deepfake.pth",
        )

        if os.path.exists(weight_path):
            try:
                ckpt = torch.load(weight_path, map_location="cpu", weights_only=False)
                if "state_dict" in ckpt:
                    ckpt = ckpt["state_dict"]

                ckpt_remapped = self._remap_keys(ckpt)

                model_keys = set(model.state_dict().keys())
                ckpt_keys = set(ckpt_remapped.keys())
                matched_keys = model_keys.intersection(ckpt_keys)

                match_ratio = (
                    len(matched_keys) / len(model_keys) if len(model_keys) > 0 else 0
                )
                self._weights_loaded_ok = (
                    match_ratio >= self.xception_cfg.match_ratio_min
                )

                if self._weights_loaded_ok:
                    model.load_state_dict(ckpt_remapped, strict=False)
                    logger.info(
                        f"Xception weights loaded successfully ({match_ratio:.1%} match)."
                    )
                else:
                    logger.warning(
                        f"Xception weight match too low ({match_ratio:.1%}). Using random init."
                    )

            except Exception as e:
                logger.error(f"Failed to load Xception weights from {weight_path}: {e}")
                self._weights_loaded_ok = False
        else:
            logger.warning(
                f"Xception weights not found at {weight_path}. Using random initialization!"
            )
            self._weights_loaded_ok = False

        model.eval()
        return model

    def _prepare_tensor(
        self, img_array: np.ndarray, device: torch.device
    ) -> torch.Tensor:
        """Resizes to 299x299 (Lanczos4) and normalizes."""
        # Check input type
        if isinstance(img_array, Image.Image):
            img_array = np.array(img_array)

        if img_array.dtype != np.uint8:
            img_array = img_array.astype(np.uint8)

        # Resize
        resized = cv2.resize(img_array, (299, 299), interpolation=cv2.INTER_LANCZOS4)

        # To tensor
        tensor = torch.from_numpy(resized).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        tensor = self.normalize(tensor)
        return tensor.to(device)

    def _extract_overlapping_patches(
        self,
        img: np.ndarray,
        grid_size: int = PATCH_GRID_SIZE,
        cover: float = PATCH_COVER_RATIO,
    ) -> List[np.ndarray]:
        """Extract overlapping patches from an image in a grid layout.

        Each patch covers `cover` fraction of the image dimension, creating
        significant overlap between adjacent patches. This ensures every
        region of the face is covered by multiple patches.
        """
        h, w = img.shape[:2]
        patch_h = int(h * cover)
        patch_w = int(w * cover)

        # Calculate step size so patches span the full image
        step_h = (h - patch_h) // max(grid_size - 1, 1) if grid_size > 1 else 0
        step_w = (w - patch_w) // max(grid_size - 1, 1) if grid_size > 1 else 0

        patches = []
        for i in range(grid_size):
            for j in range(grid_size):
                y1 = i * step_h
                x1 = j * step_w
                y2 = min(y1 + patch_h, h)
                x2 = min(x1 + patch_w, w)

                # Ensure minimum patch size
                if (y2 - y1) < 32 or (x2 - x1) < 32:
                    continue

                patch = img[y1:y2, x1:x2].copy()
                patches.append(patch)

        return patches if patches else [img]

    @torch.no_grad()
    def _batch_score_patches(
        self, model: nn.Module, patches: List[np.ndarray], device: torch.device
    ) -> Tuple[float, float]:
        """Score all patches and return (mean_score, std_score).

        Also includes TTA (horizontal flip) for each patch.
        """
        all_scores = []

        for patch in patches:
            # Original
            tensor = self._prepare_tensor(patch, device)
            score = self._score_tensor(model, tensor)

            # TTA: horizontal flip
            flipped = cv2.flip(patch, 1)
            tensor_flip = self._prepare_tensor(flipped, device)
            flip_score = self._score_tensor(model, tensor_flip)

            # Use max of original and flipped (conservative)
            all_scores.append(max(score, flip_score))

        scores_arr = np.array(all_scores, dtype=np.float64)
        return float(np.mean(scores_arr)), float(np.std(scores_arr))

    def _apply_consistency_dampening(
        self, mean_score: float, std_score: float, is_grayscale: bool = False
    ) -> Tuple[float, str]:
        """Apply patch-consistency dampening to reduce ISP false positives.

        Real phone selfies: uniform ISP processing → low patch std → dampen
        Real deepfakes: localized blending → high patch std → no dampening
        """
        if mean_score < PATCH_ACTIVATION_SCORE:
            # Score too low to warrant consistency check
            return mean_score, "Score below activation threshold"

        # FIX: Grayscale evasion (destroying variance) should NOT trigger the safety dampening
        if is_grayscale:
            if std_score < PATCH_STD_THRESHOLD:
                return mean_score, f"Patch std={std_score:.4f} < {PATCH_STD_THRESHOLD}: Dampening blocked due to GRAYSCALE evasion"
            return mean_score, f"Patch std={std_score:.3f} >= {PATCH_STD_THRESHOLD}: localized artifacts detected in grayscale"

        if std_score >= PATCH_STD_THRESHOLD:
            # High variance → localized artifacts → likely real deepfake
            return (
                mean_score,
                f"Patch std={std_score:.3f} ≥ {PATCH_STD_THRESHOLD}: localized artifacts detected",
            )

        # Low variance → uniform processing → likely ISP false positive
        # Compute dampening factor: lower std → stronger dampening
        std_ratio = std_score / PATCH_STD_THRESHOLD  # 0.0 to 1.0
        dampening = 1.0 - (PATCH_DAMPENING_FACTOR * (1.0 - std_ratio))
        adjusted = mean_score * dampening

        logger.info(
            f"Xception patch consistency: mean={mean_score:.3f}, std={std_score:.4f}, "
            f"dampening={dampening:.2f}, adjusted={adjusted:.3f}"
        )

        return adjusted, (
            f"Patch std={std_score:.4f} < {PATCH_STD_THRESHOLD}: uniform processing detected, "
            f"score dampened {dampening:.2f}x ({mean_score:.3f}→{adjusted:.3f})"
        )

    @torch.no_grad()
    def _score_tensor(self, model: nn.Module, tensor: torch.Tensor) -> float:
        """Returns the fake probability."""
        outputs = model(tensor)  # (1, 2)
        probs = F.softmax(outputs, dim=1)
        # Class 1 is usually the fake class in standard FF++ pretraining
        return float(probs[0, 1].item())

    def _run_inference(self, input_data: dict) -> ToolResult:
        start_time = time.time()
        tracked_faces = input_data.get("tracked_faces", [])
        media_path = input_data.get("media_path", None)
        first_frame = input_data.get("first_frame", None)
        heuristic_flags = input_data.get("heuristic_flags", [])
        is_grayscale = "GRAYSCALE" in heuristic_flags

        # Build list of numpy crops (RGB, uint8) to analyze
        np_crops = []

        if tracked_faces:
            for face in tracked_faces:
                # Xception uses higher-res crops when available, fallback to 224
                face_crop = getattr(
                    face, "face_crop_380", getattr(face, "face_crop_224", None)
                )
                if face_crop is None:
                    continue
                if isinstance(face_crop, Image.Image):
                    face_crop = np.array(face_crop)
                if face_crop.dtype != np.uint8:
                    face_crop = face_crop.astype(np.uint8)
                np_crops.append(face_crop)

        # No-face fallback: load raw image (Xception will resize to 299x299 in _prepare_tensor)
        if not np_crops and first_frame is not None:
            np_crops.append(first_frame)
            logger.info(
                "Xception: No faces found, falling back to raw video frame analysis."
            )
        elif not np_crops and media_path:
            try:
                raw_img = Image.open(media_path).convert("RGB")
                np_crops.append(np.array(raw_img))
                logger.info(
                    "Xception: No faces found, falling back to raw image analysis."
                )
            except Exception as e:
                logger.warning(
                    f"Xception: Failed to load raw image from {media_path}: {e}"
                )

        if not np_crops:
            return ToolResult(
                tool_name=self.tool_name,
                success=False,
                score=0.0,
                confidence=0.0,
                details={},
                error=True,
                error_msg="No image data available for analysis.",
                execution_time=time.time() - start_time,
                evidence_summary="Xception detector: No image data available.",
            )

        worst_face_score = 0.0

        with VRAMLifecycleManager(self._load_model) as model:
            if getattr(self, "_weights_loaded_ok", False) is False:
                return ToolResult(
                    tool_name=self.tool_name,
                    success=True,
                    score=0.0,
                    confidence=0.0,
                    details={"execution_time": time.time() - start_time},
                    error=False,
                    error_msg=None,
                    execution_time=time.time() - start_time,
                    evidence_summary="Model weights missing, analysis skipped.",
                )

            device = (
                next(model.parameters()).device
                if list(model.parameters())
                else torch.device("cpu")
            )

            for face_crop in np_crops:
                # Original full-crop score (with TTA flip)
                tensor_norm = self._prepare_tensor(face_crop, device)
                score = self._score_tensor(model, tensor_norm)

                # TTA: Horizontal Flip
                flipped = cv2.flip(face_crop, 1)
                tensor_flip = self._prepare_tensor(flipped, device)
                flip_score = self._score_tensor(model, tensor_flip)

                full_crop_score = max(score, flip_score)

                # ─── Patch-Wise Consistency Scoring ───
                patches = self._extract_overlapping_patches(face_crop)
                patch_mean, patch_std = self._batch_score_patches(
                    model, patches, device
                )

                # The dampening logic expects the highest base suspicion before dampening is applied.
                # If the full crop is highly suspicious (e.g. 0.50) but perfectly uniform (std=0.002),
                # it is an ISP false positive and MUST be dampened.
                base_score_to_dampen = max(full_crop_score, patch_mean)

                # Apply consistency dampening
                adjusted_score, consistency_note = self._apply_consistency_dampening(
                    base_score_to_dampen, patch_std, is_grayscale
                )

                final_score = adjusted_score

                worst_face_score = max(worst_face_score, final_score)

                # Store per-face details for reporting
                if not hasattr(self, "_last_face_details"):
                    self._last_face_details = {}
                self._last_face_details = {
                    "full_crop_score": round(full_crop_score, 4),
                    "patch_mean_score": round(patch_mean, 4),
                    "patch_std_score": round(patch_std, 4),
                    "adjusted_score": round(adjusted_score, 4),
                    "final_score": round(final_score, 4),
                    "num_patches": len(patches),
                    "consistency_note": consistency_note,
                }

        execution_time = time.time() - start_time

        # Calculate confidence using config params
        raw_confidence = (
            self.xception_cfg.confidence_base
            + abs(worst_face_score - 0.5) * self.xception_cfg.confidence_multiplier
        )
        if not self._weights_loaded_ok:
            raw_confidence = min(raw_confidence, self.xception_cfg.partial_load_cap)

        confidence = max(0.0, min(1.0, raw_confidence))

        if worst_face_score > self.fake_threshold:
            summary = f"XceptionNet flagged subtle facial blending anomalies (Authenticity: {1.0 - worst_face_score:.2f}). Consistent with face-swap manipulation."
        else:
            summary = f"XceptionNet found natural facial blending (Authenticity: {1.0 - worst_face_score:.2f})."

        # Append consistency note to summary if available
        consistency_note = getattr(self, "_last_face_details", {}).get(
            "consistency_note", ""
        )
        if consistency_note and worst_face_score < self.fake_threshold:
            summary += f" Patch consistency: {consistency_note}"

        return ToolResult(
            tool_name=self.tool_name,
            success=True,
            score=worst_face_score,
            confidence=confidence,
            details={
                "execution_time": execution_time,
                "weights_loaded_ok": self._weights_loaded_ok,
                **getattr(self, "_last_face_details", {}),
            },
            error=False,
            error_msg=None,
            execution_time=execution_time,
            evidence_summary=summary,
        )
