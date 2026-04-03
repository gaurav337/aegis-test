"""
Aegis-X SBI (Self-Blended Images) Forensic Tool
-----------------------------------------------
Architecture:
    - Backbone: EfficientNet-B4 (Trainable Head: 1792 -> 1)
    - Input: 380x380 crops (Dual-scale: 1.15x and 1.25x)
    - Explainability: GradCAM with MediaPipe Region Mapping
    
Constraints & Mitigations:
    - VRAM: Strict Two-Pass protocol (No-grad scoring -> Conditional GradCAM).
    - Drift: Exact Affine transformations for landmark mapping.
    - Artifacts: 5% boundary clipping to avoid BORDER_CONSTANT false positives.
    - Bias: Prioritizes 1.25x wide-context for GradCAM if both scales trigger.
    - API: Compliant with ToolResult schema (score, confidence, execution_time).
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
    from utils.thresholds import SBI_SKIP_CLIP_THRESHOLD, SBI_FAKE_THRESHOLD, SBI_GRADCAM_REGION_THRESHOLD
except ImportError:
    SBI_SKIP_CLIP_THRESHOLD = 0.70
    SBI_FAKE_THRESHOLD = 0.60
    SBI_GRADCAM_REGION_THRESHOLD = 0.40

logger = setup_logger(__name__)

# MediaPipe Landmark Indices for Regions
LANDMARK_REGIONS = {
    "jaw": [172, 136, 150, 149, 176, 148, 152, 377, 400, 379, 365],
    "hairline": [10, 338, 297, 332, 284],
    "cheek_l": [205, 187, 123, 116, 143],
    "cheek_r": [425, 411, 352, 345, 372],
    "nose_bridge": [168, 6, 197, 195]
}

class SBITool(BaseForensicTool):
    @property
    def tool_name(self) -> str:
        return "run_sbi"

    def __init__(self):
        super().__init__()
        # Device will be controlled by VRAMLifecycleManager context
        self.device = None
        self.has_sigmoid = False
        
        # Strict ImageNet Normalization for EfficientNet
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        self.model = None
        self.requires_gpu = True

    def setup(self):
        """Tool-specific setup called by the engine before execution."""
        logger.info(f"SBITool setup complete")
        return True

    def _load_model(self) -> torch.nn.Module:
        model = models.efficientnet_b4(weights=None)
        
        # B4 final feature map is 1792 channels
        num_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.4, inplace=True),
            nn.Linear(num_features, 1)
        )
        
        # FIX: Offline Model Path - Check local models/ directory first
        ROOT_DIR = Path(__file__).parent.parent.parent
        LOCAL_WEIGHT_PATH = ROOT_DIR / "models" / "sbi" / "efficientnet_b4.pth"
        
        weight_path = None
        if LOCAL_WEIGHT_PATH.exists():
            weight_path = str(LOCAL_WEIGHT_PATH)
            logger.info(f"Using local SBI weights: {weight_path}")
        else:
            config = AegisConfig()
            weight_path = getattr(config.models, 'sbi_weights', None)
            if weight_path:
                logger.info(f"Using config SBI weights: {weight_path}")
        
        if weight_path and os.path.exists(weight_path):
            try:
                state_dict = torch.load(weight_path, map_location='cpu', weights_only=True)
                if 'state_dict' in state_dict:
                    state_dict = state_dict['state_dict']
                model.load_state_dict(state_dict, strict=False)
                logger.info(f"SBI weights loaded from {weight_path}")
                self._weights_loaded_ok = True
            except Exception as e:
                logger.warning(f"Failed to load SBI weights: {e}. Using random init.")
                self._weights_loaded_ok = False
        else:
            logger.warning("SBI weights not found. Using random initialization (TEST MODE).")
            self._weights_loaded_ok = False

        # FIX #4: Sigmoid Verification
        last_layer = model.classifier[-1]
        if isinstance(last_layer, nn.Sigmoid):
            self.has_sigmoid = True
        else:
            self.has_sigmoid = False

        # Device handled by VRAM manager context - keep model on CPU initially
        model.eval()
        return model

    def _prepare_crop_and_landmarks(self, face_image: np.ndarray, landmarks: np.ndarray, scale: float) -> Tuple[torch.Tensor, np.ndarray]:
        """
        Extracts a scaled crop using BORDER_CONSTANT and applies the exact 
        affine transformation to landmarks to prevent coordinate drift.
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
        
        # 2. Forensic-Safe Padding (Black borders, no reflection)
        if pad_top > 0 or pad_left > 0:
            face_image = cv2.copyMakeBorder(
                face_image, pad_top, pad_bottom, pad_left, pad_right,
                cv2.BORDER_CONSTANT, value=[0, 0, 0]
            )
            
        padded_h, padded_w, _ = face_image.shape
        
        # 3. Resize to Target (380x380)
        crop = cv2.resize(face_image, (380, 380), interpolation=cv2.INTER_LANCZOS4)
        
        # 4. Exact Affine Transform for Landmarks
        # Assumes landmarks are normalized [0,1] relative to the ORIGINAL face_image
        scale_x = 380.0 / padded_w
        scale_y = 380.0 / padded_h
        
        transformed_landmarks = np.zeros_like(landmarks)
        # Convert normalized to absolute padded, then scale to 380
        transformed_landmarks[:, 0] = (landmarks[:, 0] * w + pad_left) * scale_x
        transformed_landmarks[:, 1] = (landmarks[:, 1] * h + pad_top) * scale_y
        
        # 5. Tensor Conversion (device handled by caller)
        tensor = torch.from_numpy(crop).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        tensor = self.normalize(tensor)
        
        return tensor, transformed_landmarks

    def _compute_gradcam(self, model: nn.Module, input_tensor: torch.Tensor) -> np.ndarray:
        """
        Computes GradCAM. Returns 2D numpy array (380, 380).
        Includes strict hook lifecycle management.
        """
        activations = None
        gradients = None
        
        # FIX #14: Version-safe EfficientNet block access
        if hasattr(model, 'features'):
            target_layer = model.features[-1]
        elif hasattr(model, '_blocks'):
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
            # Pass 2 executes here: Gradients enabled
            output = model(input_tensor)
            
            score = output[0, 0]
            model.zero_grad()
            score.backward()
            
            if activations is None or gradients is None:
                return np.zeros((380, 380))

            # Channel Dimension mathematically sound (Dynamic C)
            weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
            cam = torch.sum(weights * activations, dim=1)
            cam = torch.relu(cam)
            
            cam = cam - cam.min()
            if cam.max() > 0:
                cam = cam / cam.max()
            
            # FIX: Explicit squeeze(0).squeeze(0) guarantees (380, 380) 2D array
            cam = nn.functional.interpolate(
                cam.unsqueeze(0), size=(380, 380), mode='bilinear', align_corners=False
            ).squeeze(0).squeeze(0).detach().cpu().numpy()
            
            return cam
            
        finally:
            # FIX: Absolute guarantee of VRAM safety
            hook_forward.remove()
            hook_backward.remove()

    def _map_regions(self, cam: np.ndarray, landmarks: np.ndarray) -> Tuple[str, float]:
        """Maps CAM to regions. Safely ignores artificial black padding borders."""
        best_region = "diffuse"
        best_score = 0.0
        
        for region_name, indices in LANDMARK_REGIONS.items():
            region_scores = []
            for idx in indices:
                if idx < landmarks.shape[0]:
                    x, y = landmarks[idx]
                    
                    # FIX: 5% Border Clipping Mitigation (19px to 360px)
                    # Prevents sampling the BORDER_CONSTANT artifact zone
                    ix = int(np.clip(x, 19, 360))
                    iy = int(np.clip(y, 19, 360))
                    
                    region_scores.append(cam[iy, ix])
            
            if region_scores:
                mean_score = np.mean(region_scores)
                if mean_score > best_score:
                    best_score = mean_score
                    best_region = region_name
                    
        if best_score < SBI_GRADCAM_REGION_THRESHOLD:
            return "diffuse", best_score
        return best_region, best_score

    def _run_inference(self, input_data: dict) -> ToolResult:
        import time
        start_time = time.time()

        # FIX #5: Context passing
        context = input_data.get("context", {})
        # Support both siglip_score (Day 11) and clip_score (Day 12 Spec)
        visual_score = context.get("siglip_score", context.get("clip_score", 0.0))
        
        # CRITICAL Conditional Skip
        if visual_score > SBI_SKIP_CLIP_THRESHOLD:
            return ToolResult(
                tool_name=self.tool_name,
                success=True,
                score=0.0,  # FIX: Use 'score' not 'fake_score'
                confidence=1.0,  # High confidence in skip decision
                details={"skipped": True, "reason": "visual_score_high"},
                error=False,
                error_msg=None,
                execution_time=time.time() - start_time,
                evidence_summary=(
                    f"SBI analysis bypassed: Image exhibits strong fully-synthetic signatures "
                    f"(Primary Visual Score {visual_score:.2f} > {SBI_SKIP_CLIP_THRESHOLD}). "
                    f"SBI is designed exclusively for face-swap composites."
                )
            )

        tracked_faces = input_data.get("tracked_faces", [])
        if not tracked_faces:
            # FIX: ToolResult Consistency (Day 11 API Rules)
            return ToolResult(
                tool_name=self.tool_name, 
                success=False, 
                score=0.0,
                confidence=0.0,
                details={},
                error=True,
                error_msg="No tracked faces found.",
                execution_time=time.time() - start_time,
                evidence_summary="SBI detector: No tracked faces found."
            )

        with VRAMLifecycleManager(self._load_model) as model:
            if getattr(self, '_weights_loaded_ok', False) is False:
                return ToolResult(
                    tool_name=self.tool_name,
                    success=True,
                    score=0.0,
                    confidence=0.0,
                    details={"execution_time": time.time() - start_time},
                    error=False,
                    error_msg=None,
                    execution_time=time.time() - start_time,
                    evidence_summary="Model weights missing, analysis skipped."
                )

            # Get device from VRAM manager context
            self.device = next(model.parameters()).device if list(model.parameters()) else torch.device('cpu')
            
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
                
                # FIX: Landmark Coordinate Safety
                # Enforce normalized [0,1] input assumption relative to the crop
                if landmarks.max() > 1.0:
                    logger.warning("Landmarks > 1.0 detected. Assuming normalized relative to crop.")
                    # Safe fallback: if max is small (e.g. 380), normalize. If large (e.g. 1920), clamp.
                    if landmarks.max() <= 380.0:
                        landmarks = landmarks / 380.0
                    else:
                        # Cannot safely transform frame coords without frame size. Clamp to [0,1].
                        landmarks = np.clip(landmarks / landmarks.max(), 0, 1)
                
                # --- Pass 1: Cache Tensors & Fast Score (NO GRAD) ---
                tensor_115, lm_115 = self._prepare_crop_and_landmarks(face_crop, landmarks, 1.15)
                tensor_125, lm_125 = self._prepare_crop_and_landmarks(face_crop, landmarks, 1.25)
                
                # Move tensors to model device
                tensor_115 = tensor_115.to(self.device)
                tensor_125 = tensor_125.to(self.device)
                
                with torch.no_grad():
                    out_115 = model(tensor_115)
                    out_125 = model(tensor_125)
                    
                    if not self.has_sigmoid:
                        out_115 = torch.sigmoid(out_115)
                        out_125 = torch.sigmoid(out_125)
                        
                    score_115 = out_115.item()
                    score_125 = out_125.item()
                
                scores_per_scale["1.15x"] = max(scores_per_scale["1.15x"], score_115)
                scores_per_scale["1.25x"] = max(scores_per_scale["1.25x"], score_125)
                
                max_score = max(score_115, score_125)
                
                if max_score > best_score:
                    best_score = max_score
                    
                    # --- Pass 2: Conditional GradCAM ---
                    if max_score > SBI_FAKE_THRESHOLD:
                        # FIX: Bias Mitigation - Prioritize wider context if both trigger
                        if score_115 > SBI_FAKE_THRESHOLD and score_125 > SBI_FAKE_THRESHOLD:
                            target_tensor, target_lm, target_scale = tensor_125, lm_125, "1.25x"
                        elif score_125 > SBI_FAKE_THRESHOLD:
                            target_tensor, target_lm, target_scale = tensor_125, lm_125, "1.25x"
                        else:
                            target_tensor, target_lm, target_scale = tensor_115, lm_115, "1.15x"
                            
                        best_scale = target_scale
                        
                        target_tensor = target_tensor.to(self.device)
                        target_tensor.requires_grad_(True)
                        cam_map = self._compute_gradcam(model, target_tensor)
                        region, _ = self._map_regions(cam_map, target_lm)
                        
                        if region != "diffuse":
                            boundary_detected = True
                            best_boundary_region = region
                        else:
                            # Capture diffuse state correctly
                            boundary_detected = False
                            best_boundary_region = "diffuse"
                        
                        # FIX: VRAM Hygiene
                        del target_tensor
                
                # Per-face VRAM flush
                del tensor_115, tensor_125
                torch.cuda.empty_cache() if torch.cuda.is_available() else None

        execution_time = time.time() - start_time

        # FIX: Semantic Evidence Summary (Corrected Logic)
        if boundary_detected:
            summary = (f"SBI detector: localized blend boundary detected at {best_boundary_region} "
                       f"(authenticity: {1.0 - best_score:.2f}, scale: {best_scale}). "
                       f"Consistent with face-swap compositing artifact.")
        elif best_boundary_region == "diffuse" and best_score > SBI_FAKE_THRESHOLD:
            summary = (f"SBI detector: strong synthetic signatures detected globally (authenticity: {1.0 - best_score:.2f}), "
                       f"but lacking localized composite boundaries. Consistent with full-face synthesis.")
        else:
            summary = (f"SBI detector: no blend boundary detected (authenticity: {1.0 - best_score:.2f}). "
                       f"Authentic / No artifacts.")

        # FIX: Confidence based on score magnitude and boundary detection
        confidence = min(1.0, best_score + 0.2) if boundary_detected else max(0.5, 1.0 - abs(best_score - 0.5))

        # Clamp visually confusing baseline scores for users when completely authentic
        # If no boundary detected, and it's not a diffuse fake, return 0.0 to abstain.
        final_score = best_score if (boundary_detected or (best_boundary_region == "diffuse" and best_score > SBI_FAKE_THRESHOLD)) else 0.0

        return ToolResult(
            tool_name=self.tool_name,
            success=True,
            score=final_score,  # FIX: Use clamped final_score
            confidence=confidence,
            details={
                "boundary_detected": boundary_detected,
                "boundary_region": best_boundary_region,
                "winning_scale": best_scale,
                "scores_per_scale": scores_per_scale,
                "execution_time": execution_time
            },
            error=False,
            error_msg=None,
            execution_time=execution_time,
            evidence_summary=summary
        )