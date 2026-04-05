"""
Aegis-X SBI (Self-Blended Images) Forensic Tool — V3 (Audit Corrected)
Architecture:
- Backbone: EfficientNet-B4 (Trainable Head: 1792 -> 1)
- Input: 380x380 crops (Dual-scale: 1.15x and 1.25x)
- Explainability: GradCAM with MediaPipe Polygon Region Mapping
- Padding: BORDER_REFLECT_101 (no artificial black edges)

Key Fixes:
1. m-05: BORDER_REFLECT_101 padding instead of BORDER_CONSTANT (no artificial edges)
2. S-09: Skip threshold raised to 0.90 (SBI runs unless extremely obvious fake)
3. M-06: Zero persistent state — all per-face data returned in ToolResult.details
4. FIX 3: Polygon-based GradCAM region sampling (hundreds of pixels vs 5-11 points)
5. FIX 4: Multi-region boundary detection (requires 2+ adjacent regions elevated)
6. FIX 6: Landmark coordinate validation with proper warnings
7. FIX 7: Removed per-face VRAM flush (unnecessary overhead)
8. C-06: Calibrated confidence based on GradCAM boundary clarity
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

logger = setup_logger(__name__)

# ──────────────────────────────────────────────────────────────
# Threshold Imports with Fallbacks
# ──────────────────────────────────────────────────────────────
try:
    from utils.thresholds import (
        SBI_SKIP_UNIVFD_THRESHOLD,  # FIX S-09: Raised to 0.90
        SBI_FAKE_THRESHOLD,
        SBI_GRADCAM_REGION_THRESHOLD,
        SBI_COMPRESSION_DISCOUNT,
    )
except ImportError:
    SBI_SKIP_UNIVFD_THRESHOLD = 0.90
    SBI_FAKE_THRESHOLD = 0.60
    SBI_GRADCAM_REGION_THRESHOLD = 0.40
    SBI_COMPRESSION_DISCOUNT = 0.40

# ──────────────────────────────────────────────────────────────
# MediaPipe Landmark Indices for Regions (FIX 3: Expanded for polygons)
# ──────────────────────────────────────────────────────────────
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
    """Detects face-swap & reenactment deepfakes via EfficientNet-B4 + patch consistency."""

    @property
    def tool_name(self) -> str:
        return "run_sbi"

    def __init__(self):
        super().__init__()
        self.device = None
        self.has_sigmoid = False
        self._weights_loaded_ok = False

        # Strict ImageNet Normalization for EfficientNet
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        self.model = None
        self.requires_gpu = True

    def setup(self):
        """Tool-specific setup called by the engine before execution."""
        logger.info("SBITool setup complete.")
        return True

    # ──────────────────────────────────────────────────────────────
    # M-06 Fix: Explicit state reset to prevent cross-inference contamination
    # ──────────────────────────────────────────────────────────────
    def reset_state(self) -> None:
        """Clear any cached state between inferences."""
        self._weights_loaded_ok = False

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

            model = EfficientNet.from_name("efficientnet-b4")
            num_features = model._fc.in_features
            # Checkpoint has 2-class output (class 0=real, class 1=fake)
            model._fc = nn.Linear(num_features, 2)
            use_efficientnet_pytorch = True
            logger.info(
                "SBI: Using efficientnet_pytorch architecture (matches checkpoint)"
            )
        except ImportError:
            logger.warning(
                "SBI: efficientnet_pytorch not available. Falling back to torchvision with key remapping."
            )
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
                        new_k = k[4:] if k.startswith("net.") else k
                        cleaned[new_k] = v
                    state_dict = cleaned

                load_result = model.load_state_dict(state_dict, strict=False)

                # Verify that weights actually loaded
                total_model_keys = len(model.state_dict())
                missing_count = (
                    len(load_result.missing_keys) if load_result.missing_keys else 0
                )
                matched_count = total_model_keys - missing_count

                if matched_count < total_model_keys * 0.5:
                    logger.error(
                        f"SBI: Only {matched_count}/{total_model_keys} keys matched! Weights NOT loaded."
                    )
                    logger.error(f"  Missing (first 5): {load_result.missing_keys[:5]}")
                    logger.error(
                        f"  Checkpoint (first 5): {list(state_dict.keys())[:5]}"
                    )
                    self._weights_loaded_ok = False
                else:
                    if load_result.missing_keys:
                        logger.info(
                            f"SBI: {missing_count} missing keys (expected for classifier head changes)"
                        )
                    logger.info(
                        f"SBI weights loaded from {weight_path} ({matched_count}/{total_model_keys} keys matched)"
                    )
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
        self, face_image: np.ndarray, landmarks: np.ndarray, bbox: Tuple[int, int, int, int], orig_size: Tuple[int, int], scale: float
    ) -> Tuple[torch.Tensor, np.ndarray]:
        """
        Extracts a scaled crop and maps global normalized landmarks to it.
        bbox: (x1, y1, x2, y2) in original frame pixels.
        orig_size: (width, height) of original frame.
        """
        h, w, _ = face_image.shape  # This is the 380x380 crop
        orig_w, orig_h = orig_size
        bx1, by1, bx2, by2 = bbox
        bw = bx2 - bx1
        bh = by2 - by1

        # 1. Map Global Normalized -> Local 380x380 Pixels
        # lx_px = landmarks[:, 0] * orig_w
        # lx_local = (lx_px - bx1) * (380.0 / bw)
        local_lms = np.zeros_like(landmarks)
        local_lms[:, 0] = (landmarks[:, 0] * orig_w - bx1) * (380.0 / bw)
        local_lms[:, 1] = (landmarks[:, 1] * orig_h - by1) * (380.0 / bh)

        # 2. Apply Scale Padding (the 1.15x / 1.25x logic)
        pad_total_y = (h * scale) - h
        pad_total_x = (w * scale) - w
        pad_top = int(pad_total_y // 2)
        pad_left = int(pad_total_x // 2)

        # FIX 1: Content-aware padding
        if pad_top > 0 or pad_left > 0:
            face_image = cv2.copyMakeBorder(
                face_image,
                pad_top,
                int(pad_total_y - pad_top),
                pad_left,
                int(pad_total_x - pad_left),
                cv2.BORDER_REFLECT_101,
            )

        padded_h, padded_w, _ = face_image.shape
        crop = cv2.resize(face_image, (380, 380), interpolation=cv2.INTER_AREA)

        # 3. Final Coordinate Transform (due to padding + resize)
        scale_x = 380.0 / padded_w
        scale_y = 380.0 / padded_h
        
        transformed_landmarks = np.zeros_like(local_lms)
        transformed_landmarks[:, 0] = (local_lms[:, 0] + pad_left) * scale_x
        transformed_landmarks[:, 1] = (local_lms[:, 1] + pad_top) * scale_y

        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(crop_rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
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
            score = output[0, 1]  # Class 1 is FAKE in this checkpoint
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
            # FIX m-08: Ensure hooks are always removed even if exception occurs
            try:
                hook_forward.remove()
            except:
                pass
            try:
                hook_backward.remove()
            except:
                pass

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

        # FIX 4: If nose_bridge is the primary elevated region, it's almost always glasses/shadows — not a blend boundary.
        # We explicitly reject nose_bridge as a valid face-swap perimeter. Deepfakes blend at the edges (jaw, hairline).
        if best_region == "nose_bridge":
            return False, "nose_bridge_glasses_reflection", best_score

        # FIX 4: Real face-swaps have continuous blending boundaries spanning multiple regions.
        # An isolated hot spot on a single cheek is usually a lighting shadow, mole, or blush.
        # A true blend mask spans across adjacent regions to conceal the entire pasted face.
        
        if len(elevated_regions) == 1:
            # If only a SINGLE massive boundary region (like the entire sweeping jaw or the entire hairline) 
            # is absolutely on fire (very high score), we can consider it a valid boundary.
            # But an isolated small region like a single cheek ('cheek_r') is a definitive false positive.
            if best_region in ["jaw", "hairline"] and best_score > (SBI_GRADCAM_REGION_THRESHOLD + 0.15):
                return True, best_region, best_score
            return False, f"isolated_{best_region}_false_positive", best_score

        if has_adjacent_support:
            return True, best_region, best_score

        return False, "diffuse", best_score

    def _calculate_confidence(self, score: float, boundary_detected: bool) -> float:
        """C-06 Fix: Calibrated confidence based on boundary clarity + score margin."""
        base_conf = 0.4
        if boundary_detected:
            # Strong boundary = high confidence
            base_conf += 0.3
        # Boost confidence as score moves away from threshold
        margin = abs(score - 0.5) * 2.0
        return min(0.95, max(0.4, base_conf + 0.3 * margin))

    def _run_inference(self, input_data: dict) -> ToolResult:
        import time

        start_time = time.time()

        # FIX #5: Context passing for compression/grayscale detection
        context = input_data.get("context", {})
        visual_score = context.get("univfd_score", context.get("clip_score", 0.0))
        is_grayscale = context.get("is_grayscale", False)
        compression_detected = context.get("compression_detected", False)

        # FIX S-09: Raised skip threshold to 0.90 — SBI should almost always run
        if visual_score > SBI_SKIP_UNIVFD_THRESHOLD:
            return ToolResult(
                tool_name=self.tool_name,
                success=True,
                real_prob=0.0,
                confidence=1.0,
                details={"skipped": True, "reason": "visual_score_high"},
                error=False,
                error_msg=None,
                execution_time=time.time() - start_time,
                evidence_summary=(
                    f"SBI analysis bypassed: Image exhibits extremely strong fully-synthetic signatures "
                    f"(Primary Visual Score {visual_score:.2f} > {SBI_SKIP_UNIVFD_THRESHOLD}). "
                    f"SBI is designed exclusively for face-swap composites."
                ),
            )

        tracked_faces = input_data.get("tracked_faces", [])
        if not tracked_faces:
            return ToolResult(
                tool_name=self.tool_name,
                success=False,
                real_prob=0.5,
                confidence=0.0,
                details={},
                error=True,
                error_msg="No tracked faces found.",
                execution_time=time.time() - start_time,
                evidence_summary="SBI detector: No tracked faces found.",
            )

        with VRAMLifecycleManager(self._load_model) as model:
            if not self._weights_loaded_ok:
                return ToolResult(
                    tool_name=self.tool_name,
                    success=True,
                    real_prob=0.5,
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

            worst_real_prob = 1.0
            best_boundary_region = None
            best_scale = None
            scores_per_scale = {"1.15x": 1.0, "1.25x": 1.0}
            boundary_detected = False
            per_face_details = []  # FIX M-06: Collect details per face, return in result

            for face in tracked_faces:
                face_crop = getattr(face, "face_crop_380", None)
                landmarks = getattr(face, "landmarks", None)

                if face_crop is None or landmarks is None:
                    continue

                if isinstance(landmarks, list):
                    landmarks = np.array(landmarks)

                # Get original frame context for coordinate mapping
                best_idx = getattr(face, "best_frame_idx", 0)
                bbox = face.trajectory_bboxes.get(best_idx)
                if not bbox:
                    # Fallback to any available bbox if best_idx missing
                    bbox = next(iter(face.trajectory_bboxes.values()))
                
                first_frame = input_data.get("first_frame")
                orig_h, orig_w = first_frame.shape[:2] if first_frame is not None else (380, 380)

                # --- Pass 1: Cache Tensors & Fast Score (NO GRAD) ---
                tensor_115, lm_115 = self._prepare_crop_and_landmarks(
                    face_crop, landmarks, bbox, (orig_w, orig_h), 1.15
                )
                tensor_125, lm_125 = self._prepare_crop_and_landmarks(
                    face_crop, landmarks, bbox, (orig_w, orig_h), 1.25
                )

                tensor_115 = tensor_115.to(self.device)
                tensor_125 = tensor_125.to(self.device)

                with torch.no_grad():
                    out_115 = model(tensor_115)
                    out_125 = model(tensor_125)

                    # 2-class model: class 0=real, class 1=fake (checkpoint convention)
                    import torch.nn.functional as F_sbi

                    SBI_TEMPERATURE = (
                        3.0  # Desaturates extreme logits for better calibration
                    )
                    real_prob_115 = F_sbi.softmax(out_115 / SBI_TEMPERATURE, dim=1)[
                        0, 0
                    ].item()
                    real_prob_125 = F_sbi.softmax(out_125 / SBI_TEMPERATURE, dim=1)[
                        0, 0
                    ].item()

                    fake_prob_115 = 1.0 - real_prob_115
                    fake_prob_125 = 1.0 - real_prob_125

                # FIX 2: Track both scales but use average for final score
                scores_per_scale["1.15x"] = min(scores_per_scale["1.15x"], real_prob_115)
                scores_per_scale["1.25x"] = min(scores_per_scale["1.25x"], real_prob_125)

                # FIX 2: Average instead of max — prevents scale bias
                avg_real_prob = (real_prob_115 + real_prob_125) / 2.0
                avg_fake_prob = 1.0 - avg_real_prob

                if avg_real_prob < worst_real_prob:
                    worst_real_prob = avg_real_prob

                    # --- Pass 2: Conditional GradCAM ---
                    if avg_fake_prob > SBI_FAKE_THRESHOLD:
                        # Use the scale with the higher score for GradCAM
                        if fake_prob_125 >= fake_prob_115:
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

                # FIX M-06: Collect per-face details for transparent reporting
                per_face_details.append(
                    {
                        "scale_115_real_prob": round(real_prob_115, 4),
                        "scale_125_real_prob": round(real_prob_125, 4),
                        "scale_115_fake_prob": round(fake_prob_115, 4),
                        "scale_125_fake_prob": round(fake_prob_125, 4),
                        "avg_real_prob": round(avg_real_prob, 4),
                        "avg_fake_prob": round(avg_fake_prob, 4),
                        "boundary_detected": boundary_detected
                        if avg_fake_prob > SBI_FAKE_THRESHOLD
                        else False,
                        "boundary_region": best_boundary_region
                        if avg_fake_prob > SBI_FAKE_THRESHOLD
                        else None,
                    }
                )

        execution_time = time.time() - start_time

        # The SBI model detects face-swap blending. 
        # If GradCAM verifies there is NO active blend boundary, the EfficientNet 
        # score is a False Positive (e.g., reacting to glasses or intense studio shadows).
        # A face-swap detector reporting a deepfake with ZERO blend boundaries is logically contradictory.
        if (1.0 - worst_real_prob) > SBI_FAKE_THRESHOLD and not boundary_detected:
            # Neutralize the false positive logically:
            # Without physical boundary evidence from GradCAM, the EfficientNet fake probability is structurally unsupported.
            # We scale down the fake probability significantly (e.g. 0.35x), meaning a 97% fake logically retreats
            # down to ~34% fake (66% authentic) naturally without hard floors.
            fake_prob = (1.0 - worst_real_prob) * 0.35
            final_real_prob = 1.0 - fake_prob
            logger.info("SBI: High fake probability but NO boundary detected. Logically dampening unsupported fake score.")
            confidence = self._calculate_confidence(final_real_prob, boundary_detected)
            # Slash confidence as the network contradicts its own spatial evidence
            confidence *= 0.50
        else:
            final_real_prob = worst_real_prob
            # C-06 Fix: Calibrated confidence
            confidence = self._calculate_confidence(final_real_prob, boundary_detected)

        # Apply compression discount to fake probability if compression artifacts are present.
        if compression_detected:
            final_fake_prob = (1.0 - final_real_prob) * SBI_COMPRESSION_DISCOUNT
            final_real_prob = 1.0 - final_fake_prob
            logger.info(
                f"SBI: Compression detected — score dampened by {SBI_COMPRESSION_DISCOUNT}x"
            )

        final_fake_prob = 1.0 - final_real_prob

        # Evidence summary
        if boundary_detected:
            summary = (
                f"SBI detector: localized blend boundary detected at {best_boundary_region} "
                f"(authenticity: {final_real_prob:.2f}, scale: {best_scale}). "
                f"Consistent with face-swap compositing artifact."
            )
        elif final_fake_prob > SBI_FAKE_THRESHOLD:
            # Model says fake but no clean boundary — lower confidence
            summary = (
                f"SBI detector: elevated synthetic signatures (authenticity: {final_real_prob:.2f}), "
                f"but no localized boundary found. Diffuse suspicion."
            )
        else:
            summary = (
                f"SBI detector: no blend boundary detected (authenticity: {final_real_prob:.2f}). "
                f"Authentic / No artifacts."
            )

        return ToolResult(
            tool_name=self.tool_name,
            success=True,
            real_prob=round(final_real_prob, 4),
            confidence=round(confidence, 4),
            details={
                "boundary_detected": boundary_detected,
                "boundary_region": best_boundary_region,
                "winning_scale": best_scale,
                "scores_per_scale": scores_per_scale,
                "execution_time": execution_time,
                "weights_loaded_ok": self._weights_loaded_ok,
                "per_face_details": per_face_details,  # FIX M-06: Transparent, no state leakage
            },
            error=False,
            error_msg=None,
            execution_time=execution_time,
            evidence_summary=summary,
        )
