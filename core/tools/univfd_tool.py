"""Generative AI Detector using UnivFD (CLIP-ViT-L/14 linear probe).
Replaces SigLIP with a faster, more robust pre-trained CVPR 2023 architecture.
"""

import os
import time
import logging
from typing import Any, Dict, List
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor

from core.base_tool import BaseForensicTool
from core.data_types import ToolResult
from core.config import AegisConfig
from utils.vram_manager import VRAMLifecycleManager

logger = logging.getLogger(__name__)

class _UnivFDWrapper(nn.Module):
    """Wrapper to encapsulate both the backbone and the probe so VRAMManager can handle it."""
    def __init__(self, backbone, probe):
        super().__init__()
        self.backbone = backbone
        self.probe = probe

class _LinearProbe(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(768, 1)

    def forward(self, x):
        return self.fc(x)

class UnivFDTool(BaseForensicTool):
    """
    Detects generative AI images (FLUX, Midjourney, Stable Diffusion, etc.)
    using a 4 KB linear probe on top of a frozen CLIP-ViT-L/14 backbone.
    """
    
    def __init__(self):
        super().__init__()
        self._tool_name = "run_univfd"
        self.requires_gpu = True
        
        self.config = AegisConfig()
        
        try:
            from utils.thresholds import UNIVFD_FAKE_THRESHOLD, UNIVFD_CONFIDENCE_MIN, UNIVFD_TTA_ENABLED
            self.fake_threshold = UNIVFD_FAKE_THRESHOLD
            self.conf_min = UNIVFD_CONFIDENCE_MIN
            self.tta_enabled = UNIVFD_TTA_ENABLED
        except ImportError:
            self.fake_threshold = 0.60
            self.conf_min = 0.50
            self.tta_enabled = True

        self._processor = None

    @property
    def tool_name(self) -> str:
        return self._tool_name

    def setup(self):
        """Pre-load non-VRAM components like the processor."""
        logger.info("UnivFDTool setup complete.")
        return True

    def _load_model(self) -> nn.Module:
        """Loads the CLIP backbone and the linear probe into RAM/VRAM."""
        config = AegisConfig()
        backbone_dir = getattr(config.models, 'univfd_backbone_dir', 'openai/clip-vit-large-patch14')
        probe_path = getattr(config.models, 'univfd_probe_path', 'models/univfd/probe.pth')

        logger.info(f"Loading CLIP backbone from: {backbone_dir}")
        try:
            # Load in FP16 to keep VRAM footprint ~1.8 GB
            backbone = CLIPVisionModelWithProjection.from_pretrained(
                backbone_dir, 
                torch_dtype=torch.float16,
                local_files_only=os.path.exists(backbone_dir)
            )
        except Exception as e:
            logger.warning(f"Could not load CLIP locally, falling back to HF Hub: {e}")
            backbone = CLIPVisionModelWithProjection.from_pretrained(
                "openai/clip-vit-large-patch14", 
                torch_dtype=torch.float16
            )
            
        self._processor = CLIPImageProcessor.from_pretrained(
            backbone_dir if os.path.exists(backbone_dir) else "openai/clip-vit-large-patch14"
        )
        
        probe = _LinearProbe()
        if os.path.exists(probe_path):
            try:
                ckpt = torch.load(probe_path, map_location="cpu", weights_only=False)
                # Multi-format probe loading
                if "fc.weight" in ckpt:
                    probe.load_state_dict(ckpt, strict=True)
                elif "coef" in ckpt and "intercept" in ckpt:
                    with torch.no_grad():
                        probe.fc.weight.copy_(torch.tensor(ckpt["coef"]).clone().detach().reshape(1,-1))
                        probe.fc.bias.copy_(torch.tensor(ckpt["intercept"]).clone().detach().reshape(1))
                elif "weight" in ckpt:
                    with torch.no_grad():
                        probe.fc.weight.copy_(ckpt["weight"].clone().detach().reshape(1,-1))
                        if "bias" in ckpt:
                            probe.fc.bias.copy_(ckpt["bias"].clone().detach().reshape(1))
                        else:
                            probe.fc.bias.copy_(torch.zeros(1))
                logger.info("UnivFD probe weights loaded successfully.")
                self._weights_loaded_ok = True
            except Exception as e:
                logger.error(f"Failed to load UnivFD probe from {probe_path}: {e}")
                self._weights_loaded_ok = False
        else:
            logger.warning(f"UnivFD probe not found at {probe_path}. Using random init!")
            self._weights_loaded_ok = False

        backbone.eval()
        probe.eval()
        return _UnivFDWrapper(backbone, probe)

    def _crop_to_tensor(self, crop_image: Image.Image, device: torch.device) -> torch.Tensor:
        """Uses CLIP image processor to match exact backbone training distribution."""
        inputs = self._processor(images=crop_image, return_tensors="pt")
        # Ensure input matches backbone dtype (float16)
        pixel_values = inputs.pixel_values.to(device, dtype=torch.float16)
        return pixel_values

    @torch.no_grad()
    def _score_single_crop(self, wrapper: _UnivFDWrapper, crop_rgb: Image.Image, device: torch.device) -> float:
        """Processes a single PIL crop and returns the scalar generative AI score."""
        pixel_values = self._crop_to_tensor(crop_rgb, device)
        
        # 1. Backbone forward pass (FP16)
        outputs = wrapper.backbone(pixel_values=pixel_values)
        embeds = outputs.image_embeds.float() # Cast to FP32 immediately 
        
        # 2. L2 normalize (FP32)
        embeds = F.normalize(embeds, p=2, dim=-1)
        
        # 3. Probe and Sigmoid (FP32)
        logit = wrapper.probe(embeds)
        score = torch.sigmoid(logit).item()
        return score

    def _run_inference(self, input_data: dict) -> ToolResult:
        start_time = time.time()
        tracked_faces = input_data.get("tracked_faces", [])
        
        if not tracked_faces:
            return ToolResult(
                tool_name=self.tool_name, 
                success=False, score=0.0, confidence=0.0, details={}, error=True,
                error_msg="No tracked faces found.",
                execution_time=time.time() - start_time,
                evidence_summary="UnivFD detector: No faces found."
            )

        worst_face_score = 0.0
        
        with VRAMLifecycleManager(self._load_model) as wrapper:
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
                
            device = next(wrapper.parameters()).device if list(wrapper.parameters()) else torch.device("cpu")
            
            for face in tracked_faces:
                face_crop = getattr(face, "face_crop_224", None)
                if face_crop is None:
                    continue
                    
                if isinstance(face_crop, np.ndarray):
                    # OpenCV uses BGR natively. If the pipeline gave us RGB or BGR, we must handle it.
                    # Usually tracked_faces crops are stored as RGB in Aegis-X preprocessing.
                    if face_crop.dtype != np.uint8:
                        face_crop = face_crop.astype(np.uint8)
                    pil_crop = Image.fromarray(face_crop)
                elif isinstance(face_crop, Image.Image):
                    pil_crop = face_crop
                else:
                    logger.warning("Unknown face crop type passed to UnivFD.")
                    continue
                
                # Original orientation
                score = self._score_single_crop(wrapper, pil_crop, device)
                
                # TTA: Horizontal Flip
                if self.tta_enabled:
                    flipped_crop = pil_crop.transpose(Image.FLIP_LEFT_RIGHT)
                    flip_score = self._score_single_crop(wrapper, flipped_crop, device)
                    score = max(score, flip_score)
                
                worst_face_score = max(worst_face_score, score)

        execution_time = time.time() - start_time
        
        confidence = max(self.conf_min, 0.5 + abs(worst_face_score - 0.5))

        if worst_face_score > self.fake_threshold:
            summary = f"UnivFD detected strong generative AI signatures (Authenticity: {1.0 - worst_face_score:.2f})."
        else:
            summary = f"UnivFD found no reliable generative AI signatures (Authenticity: {1.0 - worst_face_score:.2f})."

        return ToolResult(
            tool_name=self.tool_name,
            success=True,
            score=worst_face_score,
            confidence=confidence,
            details={
                "siglip_score": worst_face_score,  # Mandatory backward compatibility shim
                "execution_time": execution_time
            },
            error=False,
            error_msg=None,
            execution_time=execution_time,
            evidence_summary=summary
        )
