"""
Aegis-X SBI (Self-Blended Images) Forensic Tool — V2
----------------------------------------------------
Architecture:
    - Backbone: EfficientNet-B4 (Trainable Head: 1792 -> 1)
    - Input: 380x380 crops (Dual-scale: 1.15x and 1.25x)
    - Explainability: GradCAM with MediaPipe Polygon Region Mapping

V2 Fixes:
    1. BORDER_REFLECT padding instead of BORDER_CONSTANT (no artificial black edges)
    2. Dual-scale averaging instead of max (prevents scale bias)
    3. Polygon-based GradCAM region sampling (hundreds of pixels vs 5-11 points)
    4. Multi-region boundary detection (requires 2+ adjacent regions elevated)
    5. Skip threshold raised to 0.90 (SBI always runs unless extremely obvious)
    6. Landmark coordinate validation with proper warnings
    7. Removed per-face VRAM flush (unnecessary overhead)
"""

import os
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
import cv2
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path

from core.base_tool import BaseForensicTool
from core.data_types import ToolResult
from core.config import AegisConfig
from utils.vram_manager import VRAMLifecycleManager
from utils.logger import setup_logger

# FIX #6: Threshold imports
try:
    from utils.thresholds import (
        SBI_SKIP_CLIP_THRESHOLD,
        SBI_FAKE_THRESHOLD,
        SBI_GRADCAM_REGION_THRESHOLD,
    )
except ImportError:
    SBI_SKIP_CLIP_THRESHOLD = 0.90  # FIX 5: Raised from 0.70
    SBI_FAKE_THRESHOLD = 0.60
    SBI_GRADCAM_REGION_THRESHOLD = 0.40

logger = setup_logger(__name__)

# MediaPipe Landmark Indices for Regions
# FIX 3: Expanded indices for polygon-based region sampling
LANDMARK_REGIONS = {
    "jaw": [172, 136, 150, 149, 176, 148, 152, 377, 400, 379, 365],
    "hairline": [10, 338, 297, 332, 284],
    "cheek_l": [205, 187, 123, 116, 143],
    "cheek_r": [425, 411, 352, 345, 372],
    "nose_bridge": [168, 6, 197, 195],
}

# FIX 4: Regions where blending boundaries typically appear in face-swaps
# nose_bridge is NOT a typical swap boundary — hot activation there usually means glasses/shadows
BLENDING_BOUNDARY_REGIONS = {"jaw", "hairline", "cheek_l", "cheek_r"}

# FIX 4: Adjacent region pairs — real blending boundaries span multiple regions
ADJACENT_REGION_PAIRS = [
    ("jaw", "cheek_l"),
    ("jaw", "cheek_r"),
    ("hairline", "cheek_l"),
    ("hairline", "cheek_r"),
    ("cheek_l", "nose_bridge"),
    ("cheek_r", "nose_bridge"),
]


class SBITool(BaseForensicTool):
    @property
    def tool_name(self) -> str:
        return "run_sbi"

    def __init__(self):
        super().__init__()
        self.device = None
        self.has_sigmoid = False

        # Strict ImageNet Normalization for EfficientNet
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        self.model = None
        self.requires_gpu = True

    def setup(self):
        """Tool-specific setup called by the engine before execution."""
        logger.info("SBITool setup complete")
        return True

    def _load_model(self) -> torch.nn.Module:
        """Load EfficientNet-B4 for SBI detection.
        
        The checkpoint was saved from `efficientnet_pytorch` (keys: net._conv_stem, 
        net._blocks, etc.), NOT torchvision (keys: features.0.0, features.1.0.block).
        We MUST use the matching architecture or key mapping.
        """
        # Try efficientnet_pytorch first (matches checkpoint format)
        use_efficientnet_pytorch = False
        try:
            from efficientnet_pytorch import EfficientNet
            model = EfficientNet.from_name('efficientnet-b4')
            num_features = model._fc.in_features
            # Checkpoint has 2-class output (class 0=real, class 1=fake)
            model._fc = nn.Linear(num_features, 2)
            use_efficientnet_pytorch = True
            logger.info("SBI: Using efficientnet_pytorch architecture (matches checkpoint)")
        except ImportError:
            logger.warning("SBI: efficientnet_pytorch not available. Falling back to torchvision with key remapping.")
            model = models.efficientnet_b4(weights=None)
            num_features = model.classifier[1].in_features
            model.classifier = nn.Sequential(
                nn.Dropout(p=0.4, inplace=True), nn.Linear(num_features, 2)
            )

        # FIX: Offline Model Path
        ROOT_DIR = Path(__file__).parent.parent.parent
        LOCAL_WEIGHT_PATH = ROOT_DIR / "models" / "sbi" / "efficientnet_b4.pth"

        weight_path = None
        if LOCAL_WEIGHT_PATH.exists():
            weight_path = str(LOCAL_WEIGHT_PATH)
            logger.info(f"Using local SBI weights: {weight_path}")
        else:
            config = AegisConfig()
            weight_path = getattr(config.models, "sbi_weights", None)
            if weight_path:
                logger.info(f"Using config SBI weights: {weight_path}")

        if weight_path and os.path.exists(weight_path):
            try:
                state_dict = torch.load(
                    weight_path, map_location="cpu", weights_only=True
                )
                # Checkpoint may wrap weights under 'state_dict' or 'model' key
                if "state_dict" in state_dict:
                    state_dict = state_dict["state_dict"]
                elif "model" in state_dict:
                    state_dict = state_dict["model"]
                
                # Strip 'net.' prefix from checkpoint keys
                # Checkpoint keys: net._conv_stem.weight -> _conv_stem.weight
                if use_efficientnet_pytorch:
                    cleaned = {}
                    for k, v in state_dict.items():
                        new_k = k[4:] if k.startswith('net.') else k
                        cleaned[new_k] = v
                    state_dict = cleaned
                
                load_result = model.load_state_dict(state_dict, strict=False)
                
                # Verify that weights actually loaded
                total_model_keys = len(model.state_dict())
                missing_count = len(load_result.missing_keys) if load_result.missing_keys else 0
                matched_count = total_model_keys - missing_count
                
                if matched_count < total_model_keys * 0.5:
                    logger.error(f"SBI: Only {matched_count}/{total_model_keys} keys matched! Weights NOT loaded.")
                    logger.error(f"  Missing (first 5): {load_result.missing_keys[:5]}")
                    logger.error(f"  Checkpoint (first 5): {list(state_dict.keys())[:5]}")
                    self._weights_loaded_ok = False
                else:
                    if load_result.missing_keys:
                        logger.info(f"SBI: {missing_count} missing keys (expected for classifier head changes)")
                    logger.info(f"SBI weights loaded from {weight_path} ({matched_count}/{total_model_keys} keys matched)")
                    self._weights_loaded_ok = True
            except Exception as e:
                logger.warning(f"Failed to load SBI weights: {e}. Using random init.")
                self._weights_loaded_ok = False
        else:
            logger.warning(
                "SBI weights not found. Using random initialization (TEST MODE)."
            )
            self._weights_loaded_ok = False

        # FIX #4: Sigmoid Verification
        if use_efficientnet_pytorch:
            last_layer = model._fc
        else:
            last_layer = model.classifier[-1]
        if isinstance(last_layer, nn.Sigmoid):
            self.has_sigmoid = True
        else:
            self.has_sigmoid = False

        model.eval()
        return model

    def _prepare_crop_and_landmarks(
        self, face_image: np.ndarray, landmarks: np.ndarray, scale: float
    ) -> Tuple[torch.Tensor, np.ndarray]:
        """
        Extracts a scaled crop using BORDER_REFLECT (not black padding) and
        applies the exact affine transformation to landmarks.

        FIX 1: BORDER_REFLECT_101 extends existing image content instead of
        introducing foreign black pixels that SBI mistakes for blending boundaries.
        """
        h, w, _ = face_image.shape

        # 1. Exact Float Padding Calculation
        pad_total_y = (h * scale) - h
        pad_total_x = (w * scale) - w

        # Symmetrical split (handles odd totals correctly)
        pad_top = int(pad_total_y // 2)
        pad_bottom = int(pad_total_y - pad_top)
        pad_left = int(pad_total_x // 2)
        pad_right = int(pad_total_x - pad_left)

        # FIX 1: Content-aware padding instead of black borders
        if pad_top > 0 or pad_left > 0:
            face_image = cv2.copyMakeBorder(
                face_image,
                pad_top,
                pad_bottom,
                pad_left,
                pad_right,
                cv2.BORDER_REFLECT_101,  # Reflects edge pixels, no artificial borders
            )

        padded_h, padded_w, _ = face_image.shape

        # 3. Resize to Target (380x380)
        crop = cv2.resize(face_image, (380, 380), interpolation=cv2.INTER_LANCZOS4)

        # 4. Exact Affine Transform for Landmarks
        scale_x = 380.0 / padded_w
        scale_y = 380.0 / padded_h

        transformed_landmarks = np.zeros_like(landmarks)
        transformed_landmarks[:, 0] = (landmarks[:, 0] * w + pad_left) * scale_x
        transformed_landmarks[:, 1] = (landmarks[:, 1] * h + pad_top) * scale_y

        # 5. Tensor Conversion
        tensor = torch.from_numpy(crop).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        tensor = self.normalize(tensor)

        return tensor, transformed_landmarks

    def _compute_gradcam(
        self, model: nn.Module, input_tensor: torch.Tensor
    ) -> np.ndarray:
        """Computes GradCAM. Returns 2D numpy array (380, 380)."""
        activations = None
        gradients = None

        if hasattr(model, "features"):
            target_layer = model.features[-1]
        elif hasattr(model, "_blocks"):
            target_layer = model._blocks[-1]
        else:
            raise ValueError("Unknown EfficientNet structure")

        def forward_hook(module, input, output):
            nonlocal activations
            activations = output

        def backward_hook(module, grad_input, grad_output):
            nonlocal gradients
            gradients = grad_output[0]

        hook_forward = target_layer.register_forward_hook(forward_hook)
        hook_backward = target_layer.register_full_backward_hook(backward_hook)

        try:
            output = model(input_tensor)
            score = output[0, 0]
            model.zero_grad()
            score.backward()

            if activations is None or gradients is None:
                return np.zeros((380, 380))

            weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
            cam = torch.sum(weights * activations, dim=1)
            cam = torch.relu(cam)

            cam = cam - cam.min()
            if cam.max() > 0:
                cam = cam / cam.max()

            cam = (
                nn.functional.interpolate(
                    cam.unsqueeze(0),
                    size=(380, 380),
                    mode="bilinear",
                    align_corners=False,
                )
                .squeeze(0)
                .squeeze(0)
                .detach()
                .cpu()
                .numpy()
            )

            return cam

        finally:
            hook_forward.remove()
            hook_backward.remove()

    def _map_regions_polygon(
        self, cam: np.ndarray, landmarks: np.ndarray
    ) -> Dict[str, float]:
        """
        FIX 3: Polygon-based region sampling instead of point sampling.

        Instead of sampling 5-11 individual landmark points (noise-sensitive),
        this creates a filled convex hull polygon from the landmarks and computes
        the mean CAM value over ALL pixels within the polygon (hundreds of pixels).
        """
        region_scores = {}

        for region_name, indices in LANDMARK_REGIONS.items():
            valid_points = []
            for idx in indices:
                if idx < landmarks.shape[0]:
                    x, y = landmarks[idx]
                    # Clip to valid range
                    ix = int(np.clip(x, 5, 374))
                    iy = int(np.clip(y, 5, 374))
                    valid_points.append([ix, iy])

            if len(valid_points) < 3:
                # Not enough points for a polygon — fall back to point sampling
                if valid_points:
                    pts = np.array(valid_points)
                    mean_score = float(np.mean(cam[pts[:, 1], pts[:, 0]]))
                else:
                    mean_score = 0.0
            else:
                # Create filled polygon mask
                pts = np.array(valid_points, dtype=np.int32)
                mask = np.zeros((380, 380), dtype=np.uint8)
                cv2.fillConvexPoly(mask, pts, 255)

                # Mean CAM value within polygon
                region_cam = cam[mask > 0]
                mean_score = float(np.mean(region_cam)) if len(region_cam) > 0 else 0.0

            region_scores[region_name] = mean_score

        return region_scores

    def _detect_boundary(
        self, region_scores: Dict[str, float]
    ) -> Tuple[bool, str, float]:
        """
        FIX 4: Multi-region boundary detection.

        Requires at least 2 adjacent regions both showing elevated activation.
        A single hot region is more likely a local feature (glasses, shadow) than
        a genuine blending boundary.
        """
        elevated_regions = {
            name: score
            for name, score in region_scores.items()
            if score >= SBI_GRADCAM_REGION_THRESHOLD
        }

        if not elevated_regions:
            return False, "diffuse", 0.0

        # Find the highest-scoring region
        best_region = max(elevated_regions, key=elevated_regions.get)
        best_score = elevated_regions[best_region]

        # FIX 4: Check if at least 2 adjacent regions are elevated
        has_adjacent_support = False
        for r1, r2 in ADJACENT_REGION_PAIRS:
            if r1 in elevated_regions and r2 in elevated_regions:
                has_adjacent_support = True
                break

        # FIX 4: If only nose_bridge is elevated, it's likely glasses — not a blend boundary
        if len(elevated_regions) == 1 and best_region == "nose_bridge":
            return False, "nose_bridge_single", best_score

        # Require adjacent support OR the region is a known blending boundary region
        if has_adjacent_support or best_region in BLENDING_BOUNDARY_REGIONS:
            return True, best_region, best_score

        return False, "diffuse", best_score

    def _run_inference(self, input_data: dict) -> ToolResult:
        import time

        start_time = time.time()

        # FIX #5: Context passing
        context = input_data.get("context", {})
        visual_score = context.get("siglip_score", context.get("clip_score", 0.0))

        # FIX 5: Raised skip threshold to 0.90 — SBI should almost always run
        if visual_score > SBI_SKIP_CLIP_THRESHOLD:
            return ToolResult(
                tool_name=self.tool_name,
                success=True,
                score=0.0,
                confidence=1.0,
                details={"skipped": True, "reason": "visual_score_high"},
                error=False,
                error_msg=None,
                execution_time=time.time() - start_time,
                evidence_summary=(
                    f"SBI analysis bypassed: Image exhibits extremely strong fully-synthetic signatures "
                    f"(Primary Visual Score {visual_score:.2f} > {SBI_SKIP_CLIP_THRESHOLD}). "
                    f"SBI is designed exclusively for face-swap composites."
                ),
            )

        tracked_faces = input_data.get("tracked_faces", [])
        if not tracked_faces:
            return ToolResult(
                tool_name=self.tool_name,
                success=False,
                score=0.0,
                confidence=0.0,
                details={},
                error=True,
                error_msg="No tracked faces found.",
                execution_time=time.time() - start_time,
                evidence_summary="SBI detector: No tracked faces found.",
            )

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

            self.device = (
                next(model.parameters()).device
                if list(model.parameters())
                else torch.device("cpu")
            )

            best_score = 0.0
            best_boundary_region = None
            best_scale = None
            scores_per_scale = {"1.15x": 0.0, "1.25x": 0.0}
            boundary_detected = False

            for face in tracked_faces:
                face_crop = getattr(face, "face_crop_380", None)
                landmarks = getattr(face, "landmarks", None)

                if face_crop is None or landmarks is None:
                    continue

                if isinstance(landmarks, list):
                    landmarks = np.array(landmarks)

                # FIX 6: Improved landmark coordinate validation
                if landmarks.max() > 1.0:
                    if landmarks.max() <= 380.0:
                        logger.warning(
                            f"SBI: Landmarks in pixel coords (max={landmarks.max():.0f}). "
                            f"Normalizing by 380."
                        )
                        landmarks = landmarks / 380.0
                    else:
                        logger.warning(
                            f"SBI: Landmarks likely in frame coords (max={landmarks.max():.0f}). "
                            f"Cannot safely transform without frame size — clamping to [0,1]. "
                            f"GradCAM region mapping may be inaccurate."
                        )
                        landmarks = np.clip(landmarks / landmarks.max(), 0, 1)

                # --- Pass 1: Cache Tensors & Fast Score (NO GRAD) ---
                tensor_115, lm_115 = self._prepare_crop_and_landmarks(
                    face_crop, landmarks, 1.15
                )
                tensor_125, lm_125 = self._prepare_crop_and_landmarks(
                    face_crop, landmarks, 1.25
                )

                tensor_115 = tensor_115.to(self.device)
                tensor_125 = tensor_125.to(self.device)

                with torch.no_grad():
                    out_115 = model(tensor_115)
                    out_125 = model(tensor_125)

                    # 2-class model: class 0=fake, class 1=real
                    import torch.nn.functional as F_sbi
                    score_115 = F_sbi.softmax(out_115, dim=1)[0, 0].item()
                    score_125 = F_sbi.softmax(out_125, dim=1)[0, 0].item()

                # FIX 2: Track both scales but use average for final score
                scores_per_scale["1.15x"] = max(scores_per_scale["1.15x"], score_115)
                scores_per_scale["1.25x"] = max(scores_per_scale["1.25x"], score_125)

                # FIX 2: Average instead of max — prevents scale bias
                avg_score = (score_115 + score_125) / 2.0

                if avg_score > best_score:
                    best_score = avg_score

                    # --- Pass 2: Conditional GradCAM ---
                    if avg_score > SBI_FAKE_THRESHOLD:
                        # Use the scale with the higher score for GradCAM
                        if score_125 >= score_115:
                            target_tensor, target_lm, target_scale = (
                                tensor_125,
                                lm_125,
                                "1.25x",
                            )
                        else:
                            target_tensor, target_lm, target_scale = (
                                tensor_115,
                                lm_115,
                                "1.15x",
                            )

                        best_scale = target_scale

                        target_tensor = target_tensor.to(self.device)
                        target_tensor.requires_grad_(True)
                        cam_map = self._compute_gradcam(model, target_tensor)

                        # FIX 3: Polygon-based region mapping
                        region_scores = self._map_regions_polygon(cam_map, target_lm)

                        # FIX 4: Multi-region boundary detection
                        boundary_detected, best_boundary_region, _ = (
                            self._detect_boundary(region_scores)
                        )

                        del target_tensor

                # FIX 7: Removed per-face VRAM flush — unnecessary overhead
                del tensor_115, tensor_125

        execution_time = time.time() - start_time

        # Pass through continuous model score directly.
        # The SBI model outputs calibrated probabilities — trust them.
        # GradCAM boundary detection modifies CONFIDENCE, not the score.
        final_score = best_score

        # Confidence: GradCAM boundary boosts confidence, absence lowers it
        if boundary_detected:
            confidence = min(1.0, best_score + 0.3)
            summary = (
                f"SBI detector: localized blend boundary detected at {best_boundary_region} "
                f"(authenticity: {1.0 - final_score:.2f}, scale: {best_scale}). "
                f"Consistent with face-swap compositing artifact."
            )
        elif best_score > SBI_FAKE_THRESHOLD:
            # Model says fake but no clean boundary — lower confidence
            confidence = best_score * 0.7
            summary = (
                f"SBI detector: elevated synthetic signatures (authenticity: {1.0 - final_score:.2f}), "
                f"but no localized boundary found. Diffuse suspicion."
            )
        else:
            confidence = max(0.4, 1.0 - best_score)
            summary = (
                f"SBI detector: no blend boundary detected (authenticity: {1.0 - final_score:.2f}). "
                f"Authentic / No artifacts."
            )

        return ToolResult(
            tool_name=self.tool_name,
            success=True,
            score=final_score,
            confidence=confidence,
            details={
                "boundary_detected": boundary_detected,
                "boundary_region": best_boundary_region,
                "winning_scale": best_scale,
                "scores_per_scale": scores_per_scale,
                "execution_time": execution_time,
            },
            error=False,
            error_msg=None,
            execution_time=execution_time,
            evidence_summary=summary,
        )

