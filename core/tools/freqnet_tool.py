"""
Aegis-X FreqNet Forensic Tool (v2)
----------------------------------
Replaces FreqNetDual with a fast ResNet-50 CNNDetect architecture + 
FADHook spectral analysis fusion stream.
"""

import os
import time
import logging
from typing import Any, Dict

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from PIL import Image

from core.base_tool import BaseForensicTool
from core.data_types import ToolResult
from core.config import AegisConfig
from utils.vram_manager import VRAMLifecycleManager

# Intact components from v1 FreqNet package
from core.tools.freqnet.preprocessor import DCTPreprocessor, SpatialPreprocessor
from core.tools.freqnet.fad_hook import FADHook
from core.tools.freqnet.calibration import CalibrationManager

logger = logging.getLogger(__name__)

class _CNNDetect(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = models.resnet50(weights=None)
        # Strip the last FC layer, keep features
        self.features = nn.Sequential(*list(backbone.children())[:-1])  # -> (B, 2048, 1, 1)
        self.classifier = nn.Linear(2048, 1, bias=True)

    def forward(self, x):  # x: (B, 3, 224, 224) 
        feat = self.features(x).flatten(start_dim=1)  # (B, 2048)
        return self.classifier(feat)                  # (B, 1) logit


class FreqNetTool(BaseForensicTool):
    @property
    def tool_name(self) -> str:
        return "run_freqnet"
    
    def __init__(self):
        super().__init__()
        self.device = None
        self.requires_gpu = True
        
        self.config = AegisConfig()
        self.fusion_cfg = self.config.freqnet_fusion
        
        self.calibration_manager = CalibrationManager()
        self._weights_loaded_ok = False

    def setup(self):
        """Tool-specific setup."""
        if hasattr(self.calibration_manager, 'load'):
            self.calibration_manager.load()
        logger.info("FreqNetTool setup complete.")
        return True

    def _remap_cnndetect_keys(self, ckpt: Dict[str, Any]) -> Dict[str, Any]:
        """Strips DDP prefixes and aligns fully connected layer names.
        (e.g., 'fc.weight' -> 'classifier.weight')
        """
        remapped = {}
        for k, v in ckpt.items():
            new_k = k
            if new_k.startswith("module."):
                new_k = new_k.replace("module.", "", 1)
            elif new_k.startswith("model."):
                new_k = new_k.replace("model.", "", 1)
                
            if "fc." in new_k:
                new_k = new_k.replace("fc.", "classifier.", 1)
                
            remapped[new_k] = v
        return remapped

    def _load_model(self) -> torch.nn.Module:
        logger.info("Loading CNNDetect ResNet-50 architecture...")
        model = _CNNDetect()
        
        weight_path = getattr(self.config.models, 'freqnet_weights', 'models/freqnet/cnndetect_resnet50.pth')
        
        if os.path.exists(weight_path):
            try:
                ckpt = torch.load(weight_path, map_location='cpu', weights_only=False)
                
                if 'state_dict' in ckpt:
                    ckpt = ckpt['state_dict']
                elif 'model' in ckpt:
                    ckpt = ckpt['model']
                    
                remapped = self._remap_cnndetect_keys(ckpt)
                model.load_state_dict(remapped, strict=False)
                self._weights_loaded_ok = True
                logger.info(f"FreqNet (CNNDetect) weights loaded from {weight_path}")
            except Exception as e:
                logger.warning(f"Failed to load CNNDetect weights: {e}")
                self._weights_loaded_ok = False
        else:
            logger.warning(f"CNNDetect weights absent at {weight_path}. Fallback to pure FAD mode.")
            self._weights_loaded_ok = False
            
        model.eval()
        return model

    def _run_inference(self, input_data: dict) -> ToolResult:
        start_time = time.time()
        
        tracked_faces = input_data.get("tracked_faces", [])
        media_path = input_data.get("media_path", None)
        first_frame = input_data.get("first_frame", None)
        
        # Build list of numpy crops (224x224, uint8, RGB) to analyze
        np_crops = []
        
        if tracked_faces:
            for face in tracked_faces:
                face_crop = getattr(face, "face_crop_224", None)
                if face_crop is None:
                    continue
                if isinstance(face_crop, Image.Image):
                    face_crop = np.array(face_crop)
                if face_crop.dtype != np.uint8:
                    face_crop = face_crop.astype(np.uint8)
                np_crops.append(face_crop)
        
        # ALWAYS analyze the full image context for spectral anomalies
        if first_frame is not None:
            raw_img = cv2.resize(first_frame, (224, 224), interpolation=cv2.INTER_LANCZOS4)
            np_crops.append(raw_img)
            logger.info("FreqNet: Attached full video frame context for frequency analysis.")
        elif media_path:
            try:
                raw_img = Image.open(media_path).convert("RGB").resize((224, 224), Image.LANCZOS)
                np_crops.append(np.array(raw_img))
                logger.info("FreqNet: Attached full raw image context for frequency analysis.")
            except Exception as e:
                logger.warning(f"FreqNet: Failed to load raw image from {media_path}: {e}")
        
        if not np_crops:
            return ToolResult(
                tool_name=self.tool_name,
                success=False, score=0.0, confidence=0.0, details={}, error=True,
                error_msg="No image data available for analysis.",
                execution_time=time.time() - start_time,
                evidence_summary="FreqNet detector: No image data available."
            )

        worst_face_score = 0.0
        best_band_analysis = None
        
        spatial_prep = SpatialPreprocessor()
        dct_prep = DCTPreprocessor()
        
        # Determine fallback threshold from general config
        try:
            from utils.thresholds import FREQNET_FAKE_THRESHOLD
            fake_threshold = FREQNET_FAKE_THRESHOLD
        except ImportError:
            fake_threshold = 0.60
            
        with VRAMLifecycleManager(self._load_model) as model:
            # Device pinning
            self.device = next(model.parameters()).device if list(model.parameters()) else torch.device("cpu")
            spatial_prep = spatial_prep.to(self.device)
            dct_prep = dct_prep.to(self.device)
            
            for face_crop in np_crops:

                # Prepare tensor: expects (B, 3, H, W) float
                tensor = torch.from_numpy(face_crop).permute(2, 0, 1).unsqueeze(0).float() / 255.0
                tensor = tensor.to(self.device)

                with torch.no_grad():
                    # --- Stream 1: Statistical FAD ---
                    # FADHook intercepts intermediate frequency tensors to extract energy maps
                    calib_data = self.calibration_manager.get_data() if hasattr(self.calibration_manager, 'get_data') else None
                    hook = FADHook(calib_data)
                    
                    # Temporarily register hook via DCT preprocessing pass
                    hook.register(dct_prep._dct_conv)
                    _ = dct_prep(tensor)
                    hook.remove()
                    
                    band = hook.analyze()
                    
                    # FAD Logic
                    fad_score = 0.0
                    if band and getattr(band, 'anomaly_detected', False):
                        mid_excess = max(0.0, getattr(band, 'mid_ratio', 0.0) - self.fusion_cfg.mid_excess_threshold)
                        high_excess = max(0.0, getattr(band, 'high_ratio', 0.0) - self.fusion_cfg.high_excess_threshold)
                        
                        fad_math = (mid_excess * self.fusion_cfg.mid_multiplier) + (high_excess * self.fusion_cfg.high_multiplier)
                        fad_score = float(min(1.0, fad_math))

                    # --- Stream 2: Neural ---
                    neural_score = 0.0  # FIX: Default to 0 (no detection), not 0.5 (neutral vote)
                    if self._weights_loaded_ok:
                        norm_tensor = spatial_prep(tensor)
                        logit = model(norm_tensor)
                        neural_score = float(torch.sigmoid(logit).item())
                        
                    # --- Dual Fusion ---
                    if self._weights_loaded_ok:
                        final_score = (self.fusion_cfg.neural_weight * neural_score) + (self.fusion_cfg.fad_weight * fad_score)
                    else:
                        final_score = fad_score  # 100% statistical fallback
                        
                    if final_score > worst_face_score:
                        worst_face_score = final_score
                        best_band_analysis = band

        execution_time = time.time() - start_time
        
        confidence = 0.9 if self._weights_loaded_ok else 0.5
        
        # Evidence generation
        if worst_face_score > fake_threshold:
            anomaly_info = getattr(best_band_analysis, 'interpretation', 'Spectral trace detected') if best_band_analysis else 'Pattern matched'
            summary = f"FreqNet detected high-frequency spatial artifacts ({anomaly_info}). Authenticity: {1.0 - worst_face_score:.2f}."
        else:
            summary = f"FreqNet analysis showed normal frequency energy distribution (Authenticity: {1.0 - worst_face_score:.2f})."
            
        return ToolResult(
            tool_name=self.tool_name,
            success=True,
            score=worst_face_score,
            confidence=confidence,
            details={
                "fad_score_contribution": fad_score if 'fad_score' in locals() else 0.0,
                "neural_score_contribution": neural_score if 'neural_score' in locals() else 0.5,
                "weights_loaded_ok": self._weights_loaded_ok,
                "anomaly_detected": getattr(best_band_analysis, 'anomaly_detected', False) if best_band_analysis else False,
                "execution_time": execution_time
            },
            error=False,
            error_msg=None,
            execution_time=execution_time,
            evidence_summary=summary
        )