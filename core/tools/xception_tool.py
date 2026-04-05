"""
Aegis-X XceptionNet Forensic Tool (v3.1 - Audit Corrected)
Face-swap & reenactment detection using XceptionNet (FaceForensics++) + Patch Consistency.

Key Fixes:
1. S-08: Patch dampening ONLY applies to low-suspicion scores. Well-blended deepfakes (high score + uniform patches) are preserved.
2. M-06: Removed persistent `_last_face_details` instance variable. Zero state leakage.
3. C-06: Confidence is now epistemic (calibrated to score margin + weight load status).
4. Grayscale Evasion: Explicitly bypasses dampening when GRAYSCALE flag is present.
5. TTA & Batch Scoring: Maintained for stability; cleaned up tensor lifecycle.
"""
import os
import time
import logging
from typing import Any, Dict, List, Tuple
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import timm
from PIL import Image

from core.base_tool import BaseForensicTool
from core.data_types import ToolResult
from core.config import AegisConfig
from utils.vram_manager import VRAMLifecycleManager
from utils.logger import setup_logger

logger = setup_logger(__name__)

# ──────────────────────────────────────────────────────────────
# Patch Consistency Configuration
# ──────────────────────────────────────────────────────────────
PATCH_GRID_SIZE = 3
PATCH_COVER_RATIO = 0.65
PATCH_STD_THRESHOLD = 0.08
PATCH_DAMPENING_FACTOR = 0.45
PATCH_ACTIVATION_SCORE = 0.35  # Only evaluate consistency if base suspicion is meaningful

class XceptionTool(BaseForensicTool):
    """Detects face-swap & reenactment deepfakes via XceptionNet + patch variance."""
    
    def __init__(self):
        super().__init__()
        self._tool_name = "run_xception"
        self.requires_gpu = True
        self.config = AegisConfig()
        self._weights_loaded_ok = False
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        
        try:
            from utils.thresholds import XCEPTION_FAKE_THRESHOLD
            self.fake_threshold = XCEPTION_FAKE_THRESHOLD
        except ImportError:
            self.fake_threshold = 0.50

    @property
    def tool_name(self) -> str:
        return self._tool_name

    def setup(self) -> None:
        logger.info("XceptionTool setup complete.")
        return True

    def reset_state(self) -> None:
        """Explicit state reset to prevent cross-inference contamination (M-06)."""
        self._weights_loaded_ok = False

    def _remap_keys(self, ckpt: Dict[str, Any]) -> Dict[str, Any]:
        """Remaps FaceForensics++ checkpoint keys to timm Xception format."""
        remapped = {}
        for k, v in ckpt.items():
            new_k = k.replace("module.", "").replace("model.", "")
            if "last_linear" in new_k:
                new_k = new_k.replace("last_linear", "fc")
            elif "classifier" in new_k:
                new_k = new_k.replace("classifier", "fc")
            remapped[new_k] = v
        return remapped

    def _load_model(self) -> nn.Module:
        logger.info("Loading XceptionNet model...")
        model = timm.create_model("xception", pretrained=False, num_classes=2)
        weight_path = getattr(self.config.models, "xception_weights", "models/xception/xception_deepfake.pth")

        if os.path.exists(weight_path):
            try:
                ckpt = torch.load(weight_path, map_location="cpu", weights_only=False)
                if "state_dict" in ckpt:
                    ckpt = ckpt["state_dict"]
                
                remapped = self._remap_keys(ckpt)
                load_result = model.load_state_dict(remapped, strict=False)
                total_keys = len(model.state_dict())
                matched = total_keys - len(load_result.missing_keys or [])
                ratio = matched / total_keys if total_keys > 0 else 0.0
                
                cfg = self.config.xception
                if ratio >= cfg.match_ratio_min:
                    self._weights_loaded_ok = True
                    logger.info(f"Xception weights loaded ({ratio:.1%} match).")
                else:
                    logger.warning(f"Xception weight match too low ({ratio:.1%}). Using random init.")
            except Exception as e:
                logger.error(f"Failed to load Xception weights: {e}")
                self._weights_loaded_ok = False
        else:
            logger.warning("Xception weights not found. Random initialization.")
            self._weights_loaded_ok = False

        model.eval()
        return model

    def _prepare_tensor(self, img_array: np.ndarray, device: torch.device) -> torch.Tensor:
        """Resizes to 299x299, normalizes, and moves to device."""
        if isinstance(img_array, Image.Image):
            img_array = np.array(img_array)
        if img_array.dtype != np.uint8:
            img_array = img_array.astype(np.uint8)
            
        resized = cv2.resize(img_array, (299, 299), interpolation=cv2.INTER_LANCZOS4)
        tensor = torch.from_numpy(resized).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        tensor = self.normalize(tensor)
        return tensor.to(device)

    def _extract_overlapping_patches(self, img: np.ndarray, grid_size: int = PATCH_GRID_SIZE, cover: float = PATCH_COVER_RATIO) -> List[np.ndarray]:
        h, w = img.shape[:2]
        patch_h, patch_w = int(h * cover), int(w * cover)
        step_h = (h - patch_h) // max(grid_size - 1, 1) if grid_size > 1 else 0
        step_w = (w - patch_w) // max(grid_size - 1, 1) if grid_size > 1 else 0
        
        patches = []
        for i in range(grid_size):
            for j in range(grid_size):
                y1, x1 = i * step_h, j * step_w
                y2 = min(y1 + patch_h, h)
                x2 = min(x1 + patch_w, w)
                if (y2 - y1) >= 32 and (x2 - x1) >= 32:
                    patches.append(img[y1:y2, x1:x2].copy())
        return patches if patches else [img]

    @torch.no_grad()
    def _batch_score_patches(self, model: nn.Module, patches: List[np.ndarray], device: torch.device) -> Tuple[float, float]:
        all_scores = []
        for patch in patches:
            t_orig = self._prepare_tensor(patch, device)
            s_orig = self._score_tensor(model, t_orig)
            
            t_flip = self._prepare_tensor(cv2.flip(patch, 1), device)
            s_flip = self._score_tensor(model, t_flip)
            all_scores.append(max(s_orig, s_flip))
            
        arr = np.array(all_scores, dtype=np.float64)
        return float(np.mean(arr)), float(np.std(arr))

    @torch.no_grad()
    def _score_tensor(self, model: nn.Module, tensor: torch.Tensor) -> float:
        out = model(tensor)
        probs = F.softmax(out, dim=1)
        return float(probs[0, 1].item())

    def _apply_consistency_dampening(self, mean_score: float, std_score: float, is_grayscale: bool = False) -> Tuple[float, str]:
        """
        FIX S-08: Only dampen LOW suspicion scores. 
        High suspicion + uniform patches = well-blended deepfake (DO NOT DAMPEN).
        Low suspicion + uniform patches = ISP false positive (DAMPEN).
        """
        if mean_score < PATCH_ACTIVATION_SCORE:
            return mean_score, "Below activation threshold"
            
        if is_grayscale:
            return mean_score, "Grayscale evasion: dampening bypassed"
            
        if std_score >= PATCH_STD_THRESHOLD:
            return mean_score, f"Localized artifacts detected (std={std_score:.4f})"
            
        # Low variance branch
        if mean_score >= 0.55:
            # High suspicion + uniform = sophisticated blend. Preserve score.
            return mean_score, f"High suspicion, uniform processing (std={std_score:.4f}). Dampening bypassed."
        
        # Low suspicion + uniform = likely ISP artifact. Dampen.
        std_ratio = std_score / PATCH_STD_THRESHOLD
        dampening = 1.0 - (PATCH_DAMPENING_FACTOR * (1.0 - std_ratio))
        adjusted = mean_score * dampening
        return adjusted, f"Uniform ISP artifact. Score dampened {dampening:.2f}x ({mean_score:.3f}→{adjusted:.3f})"

    def _calculate_confidence(self, score: float, weights_ok: bool) -> float:
        """Epistemic confidence: probability that the reported score is reliable."""
        if not weights_ok:
            return 0.2
        # Parabolic: low near threshold (0.5), high at extremes
        margin = abs(score - 0.5) * 2.0  # 0.0 at 0.5, 1.0 at 0.0/1.0
        return min(0.95, max(0.4, 0.4 + 0.6 * margin))

    def _run_inference(self, input_data: Dict[str, Any]) -> ToolResult:
        start_time = time.time()
        tracked_faces = input_data.get("tracked_faces", [])
        heuristic_flags = input_data.get("heuristic_flags", [])
        is_grayscale = "GRAYSCALE" in heuristic_flags
        first_frame = input_data.get("first_frame", None)
        media_path = input_data.get("media_path", None)

        np_crops = []
        if tracked_faces:
            for face in tracked_faces:
                crop = getattr(face, "face_crop_380", getattr(face, "face_crop_224", None))
                if crop is None: continue
                if isinstance(crop, Image.Image): crop = np.array(crop)
                if crop.dtype != np.uint8: crop = crop.astype(np.uint8)
                np_crops.append(crop)
                
        if not np_crops and first_frame is not None:
            np_crops.append(first_frame)
        elif not np_crops and media_path:
            try: np_crops.append(np.array(Image.open(media_path).convert("RGB")))
            except: pass

        if not np_crops:
            return ToolResult(tool_name=self.tool_name, success=False, score=0.0, confidence=0.0,
                              error=True, error_msg="No image data", execution_time=0.0,
                              evidence_summary="Xception: No image data available.")

        worst_score = 0.0
        details_list = []

        with VRAMLifecycleManager(self._load_model) as model:
            if not self._weights_loaded_ok:
                return ToolResult(tool_name=self.tool_name, success=True, score=0.0, confidence=0.0,
                                  details={"weights_loaded_ok": False, "execution_time": time.time()-start_time},
                                  evidence_summary="Model weights missing.")

            device = next(model.parameters()).device if list(model.parameters()) else torch.device("cpu")

            for crop in np_crops:
                # Full crop TTA
                t_orig = self._prepare_tensor(crop, device)
                s_orig = self._score_tensor(model, t_orig)
                t_flip = self._prepare_tensor(cv2.flip(crop, 1), device)
                s_flip = self._score_tensor(model, t_flip)
                full_score = max(s_orig, s_flip)

                # Patches
                patches = self._extract_overlapping_patches(crop)
                patch_mean, patch_std = self._batch_score_patches(model, patches, device)

                # Consistency Dampening (FIX S-08)
                base_score = max(full_score, patch_mean)
                adjusted, note = self._apply_consistency_dampening(base_score, patch_std, is_grayscale)
                
                if adjusted > worst_score:
                    worst_score = adjusted

                details_list.append({
                    "full_crop_score": round(full_score, 4),
                    "patch_mean": round(patch_mean, 4),
                    "patch_std": round(patch_std, 4),
                    "adjusted_score": round(adjusted, 4),
                    "consistency_note": note
                })

        confidence = self._calculate_confidence(worst_score, self._weights_loaded_ok)
        execution_time = time.time() - start_time

        if worst_score > self.fake_threshold:
            summary = f"XceptionNet flagged facial blending anomalies (Authenticity: {1.0-worst_score:.2f})."
        else:
            summary = f"XceptionNet found natural facial blending (Authenticity: {1.0-worst_score:.2f})."

        if details_list:
            summary += f" Patch consistency: {details_list[-1]['consistency_note']}"

        return ToolResult(
            tool_name=self.tool_name, success=True, score=round(worst_score, 4),
            confidence=round(confidence, 4),
            details={
                "execution_time": execution_time,
                "weights_loaded_ok": self._weights_loaded_ok,
                "faces_analyzed": len(details_list),
                "per_face_details": details_list  # Transparent, no state leakage
            },
            error=False, error_msg=None, execution_time=execution_time,
            evidence_summary=summary
        )