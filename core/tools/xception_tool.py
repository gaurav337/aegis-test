"""Face-Swap / Reenactment Detector using XceptionNet.
Leverages the timm library for the architecture and uses models
pre-trained on FaceForensics++.
"""

import os
import time
import logging
from typing import Any, Dict
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

class XceptionTool(BaseForensicTool):
    """
    Detects face-swap and reenactment deepfakes using an XceptionNet backbone.
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
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
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
        model = timm.create_model('xception', pretrained=False, num_classes=2)
        
        weight_path = getattr(self.config.models, "xception_weights", "models/xception/xception_deepfake.pth")
        
        if os.path.exists(weight_path):
            try:
                ckpt = torch.load(weight_path, map_location="cpu", weights_only=False)
                if "state_dict" in ckpt:
                    ckpt = ckpt["state_dict"]
                
                ckpt_remapped = self._remap_keys(ckpt)
                
                model_keys = set(model.state_dict().keys())
                ckpt_keys = set(ckpt_remapped.keys())
                matched_keys = model_keys.intersection(ckpt_keys)
                
                match_ratio = len(matched_keys) / len(model_keys) if len(model_keys) > 0 else 0
                self._weights_loaded_ok = (match_ratio >= self.xception_cfg.match_ratio_min)
                
                if self._weights_loaded_ok:
                    model.load_state_dict(ckpt_remapped, strict=False)
                    logger.info(f"Xception weights loaded successfully ({match_ratio:.1%} match).")
                else:
                    logger.warning(f"Xception weight match too low ({match_ratio:.1%}). Using random init.")
                    
            except Exception as e:
                logger.error(f"Failed to load Xception weights from {weight_path}: {e}")
                self._weights_loaded_ok = False
        else:
            logger.warning(f"Xception weights not found at {weight_path}. Using random initialization!")
            self._weights_loaded_ok = False
            
        model.eval()
        return model

    def _prepare_tensor(self, img_array: np.ndarray, device: torch.device) -> torch.Tensor:
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

    @torch.no_grad()
    def _score_tensor(self, model: nn.Module, tensor: torch.Tensor) -> float:
        """Returns the fake probability."""
        outputs = model(tensor) # (1, 2)
        probs = F.softmax(outputs, dim=1)
        # Class 1 is usually the fake class in standard FF++ pretraining
        return float(probs[0, 1].item())

    def _run_inference(self, input_data: dict) -> ToolResult:
        start_time = time.time()
        tracked_faces = input_data.get("tracked_faces", [])
        media_path = input_data.get("media_path", None)
        first_frame = input_data.get("first_frame", None)
        
        # Build list of numpy crops (RGB, uint8) to analyze
        np_crops = []
        
        if tracked_faces:
            for face in tracked_faces:
                # Xception uses higher-res crops when available, fallback to 224
                face_crop = getattr(face, "face_crop_380", getattr(face, "face_crop_224", None))
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
            logger.info("Xception: No faces found, falling back to raw video frame analysis.")
        elif not np_crops and media_path:
            try:
                raw_img = Image.open(media_path).convert("RGB")
                np_crops.append(np.array(raw_img))
                logger.info("Xception: No faces found, falling back to raw image analysis.")
            except Exception as e:
                logger.warning(f"Xception: Failed to load raw image from {media_path}: {e}")
        
        if not np_crops:
            return ToolResult(
                tool_name=self.tool_name, 
                success=False, score=0.0, confidence=0.0, details={}, error=True,
                error_msg="No image data available for analysis.",
                execution_time=time.time() - start_time,
                evidence_summary="Xception detector: No image data available."
            )

        worst_face_score = 0.0
        
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
                
            device = next(model.parameters()).device if list(model.parameters()) else torch.device("cpu")
            
            for face_crop in np_crops:
                # Original orientation
                tensor_norm = self._prepare_tensor(face_crop, device)
                score = self._score_tensor(model, tensor_norm)
                
                # TTA: Horizontal Flip
                flipped = cv2.flip(face_crop, 1)
                
                tensor_flip = self._prepare_tensor(flipped, device)
                flip_score = self._score_tensor(model, tensor_flip)
                
                max_score = max(score, flip_score)
                worst_face_score = max(worst_face_score, max_score)

        execution_time = time.time() - start_time
        
        # Calculate confidence using config params
        raw_confidence = self.xception_cfg.confidence_base + abs(worst_face_score - 0.5) * self.xception_cfg.confidence_multiplier
        if not self._weights_loaded_ok:
            raw_confidence = min(raw_confidence, self.xception_cfg.partial_load_cap)
            
        confidence = max(0.0, min(1.0, raw_confidence))

        if worst_face_score > self.fake_threshold:
            summary = f"XceptionNet flagged subtle facial blending anomalies (Authenticity: {1.0 - worst_face_score:.2f}). Consistent with face-swap manipulation."
        else:
            summary = f"XceptionNet found natural facial blending (Authenticity: {1.0 - worst_face_score:.2f})."

        return ToolResult(
            tool_name=self.tool_name,
            success=True,
            score=worst_face_score,
            confidence=confidence,
            details={
                "execution_time": execution_time,
                "weights_loaded_ok": self._weights_loaded_ok
            },
            error=False,
            error_msg=None,
            execution_time=execution_time,
            evidence_summary=summary
        )
