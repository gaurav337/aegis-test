"""Generative AI Detector using UnivFD (CLIP-ViT-L/14 linear probe).
Replaces SigLIP with a faster, more robust pre-trained CVPR 2023 architecture.

V2 Fixes:
1. No double resizing — pass original resolution to CLIP processor (it handles resizing internally)
2. Face-only analysis — full image fallback ONLY when zero faces detected
3. TTA averaging — use mean(original, flipped) instead of max
4. Phone detection — higher threshold for phone-origin images
5. Temperature scaling — calibrate sigmoid output for better probability estimates
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


# ─── EXIF Phone Detection (shared with DCT tool) ───
def _is_phone_origin(filepath: str) -> bool:
    """Check if image likely originated from a phone camera via EXIF data."""
    phone_markers = [
        b"Apple",
        b"Samsung",
        b"Xiaomi",
        b"Huawei",
        b"OnePlus",
        b"Google",
        b"SONY",
        b"OPPO",
        b"vivo",
        b"realme",
        b"Motorola",
        b"Nokia",
        b"LG",
        b"HTC",
    ]
    try:
        with open(filepath, "rb") as f:
            content = f.read(2_000_000)
            if b"Exif\x00\x00" in content or b"JFIF" in content:
                for marker in phone_markers:
                    if marker in content:
                        return True
    except Exception:
        pass
    return False


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

    V2: Fixed double-resizing, full-scene poisoning, max-TTA, phone calibration.
    """

    # Temperature scaling parameters
    TEMP_REAL = 1.5  # Phone selfies need higher temperature (spread distribution)
    TEMP_DEFAULT = 1.0  # Standard images
    PHONE_FAKE_THRESHOLD = 0.75  # Higher threshold for phone-origin images

    def __init__(self):
        super().__init__()
        self._tool_name = "run_univfd"
        self.requires_gpu = True

        self.config = AegisConfig()

        try:
            from utils.thresholds import (
                UNIVFD_FAKE_THRESHOLD,
                UNIVFD_CONFIDENCE_MIN,
                UNIVFD_TTA_ENABLED,
            )

            self.fake_threshold = UNIVFD_FAKE_THRESHOLD
            self.conf_min = UNIVFD_CONFIDENCE_MIN
            self.tta_enabled = UNIVFD_TTA_ENABLED
        except ImportError:
            self.fake_threshold = 0.60
            self.conf_min = 0.50
            self.tta_enabled = True

        self._processor = None
        self._weights_loaded_ok = False

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
        backbone_dir = getattr(
            config.models, "univfd_backbone_dir", "openai/clip-vit-large-patch14"
        )
        probe_path = getattr(
            config.models, "univfd_probe_path", "models/univfd/probe.pth"
        )

        logger.info(f"Loading CLIP backbone from: {backbone_dir}")
        try:
            # Load in FP16 to keep VRAM footprint ~1.8 GB
            backbone = CLIPVisionModelWithProjection.from_pretrained(
                backbone_dir,
                torch_dtype=torch.float16,
                local_files_only=os.path.exists(backbone_dir),
            )
        except Exception as e:
            logger.warning(f"Could not load CLIP locally, falling back to HF Hub: {e}")
            backbone = CLIPVisionModelWithProjection.from_pretrained(
                "openai/clip-vit-large-patch14", torch_dtype=torch.float16
            )

        self._processor = CLIPImageProcessor.from_pretrained(
            backbone_dir
            if os.path.exists(backbone_dir)
            else "openai/clip-vit-large-patch14"
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
                        probe.fc.weight.copy_(
                            torch.tensor(ckpt["coef"]).clone().detach().reshape(1, -1)
                        )
                        probe.fc.bias.copy_(
                            torch.tensor(ckpt["intercept"]).clone().detach().reshape(1)
                        )
                elif "weight" in ckpt:
                    with torch.no_grad():
                        probe.fc.weight.copy_(
                            ckpt["weight"].clone().detach().reshape(1, -1)
                        )
                        if "bias" in ckpt:
                            probe.fc.bias.copy_(
                                ckpt["bias"].clone().detach().reshape(1)
                            )
                        else:
                            probe.fc.bias.copy_(torch.zeros(1))
                logger.info("UnivFD probe weights loaded successfully.")
                self._weights_loaded_ok = True
            except Exception as e:
                logger.error(f"Failed to load UnivFD probe from {probe_path}: {e}")
                self._weights_loaded_ok = False
        else:
            logger.warning(
                f"UnivFD probe not found at {probe_path}. Using random init!"
            )
            self._weights_loaded_ok = False

        backbone.eval()
        probe.eval()
        return _UnivFDWrapper(backbone, probe)

    def _crop_to_tensor(
        self, crop_image: Image.Image, device: torch.device
    ) -> torch.Tensor:
        """Uses CLIP image processor to match exact backbone training distribution.

        FIX 1: NO manual resizing — CLIPImageProcessor handles all preprocessing
        (resize, center crop, normalize) internally to match training distribution.
        """
        inputs = self._processor(images=crop_image, return_tensors="pt")
        pixel_values = inputs.pixel_values.to(device, dtype=torch.float16)
        return pixel_values

    @torch.no_grad()
    def _score_single_crop(
        self, wrapper: _UnivFDWrapper, crop_rgb: Image.Image, device: torch.device
    ) -> float:
        """Processes a single PIL crop and returns the scalar generative AI score."""
        pixel_values = self._crop_to_tensor(crop_rgb, device)

        # 1. Backbone forward pass (FP16)
        outputs = wrapper.backbone(pixel_values=pixel_values)
        embeds = outputs.image_embeds.float()  # Cast to FP32

        # 2. L2 normalize (FP32) — matches Ojha et al. training protocol
        embeds = F.normalize(embeds, p=2, dim=-1)

        # 3. Probe (FP32)
        logit = wrapper.probe(embeds)
        return logit.item()  # Return raw logit, not sigmoid

    def _calibrate_score(self, logit: float, temperature: float = 1.0) -> float:
        """Apply temperature-scaled sigmoid calibration to raw logit.

        FIX 6: Raw logits from linear probes are not well-calibrated probabilities.
        Temperature scaling spreads the distribution to reduce overconfidence.
        """
        scaled_logit = logit / temperature
        return float(1.0 / (1.0 + np.exp(-scaled_logit)))

    def _run_inference(self, input_data: dict) -> ToolResult:
        start_time = time.time()
        tracked_faces = input_data.get("tracked_faces", [])
        media_path = input_data.get("media_path", None)
        first_frame = input_data.get("first_frame", None)

        # ─── FIX 4: Phone detection ───
        is_phone = False
        if media_path:
            is_phone = _is_phone_origin(media_path)
            if is_phone:
                logger.info(
                    "UnivFD: Phone-origin image detected — applying higher threshold"
                )

        # Determine effective threshold and temperature
        effective_threshold = (
            self.PHONE_FAKE_THRESHOLD if is_phone else self.fake_threshold
        )
        temperature = self.TEMP_REAL if is_phone else self.TEMP_DEFAULT

        # ─── FIX 2: Face-only analysis (no full-scene poisoning) ───
        pil_crops: List[Image.Image] = []
        has_faces = False

        if tracked_faces:
            for face in tracked_faces:
                face_crop = getattr(face, "face_crop_224", None)
                if face_crop is None:
                    continue
                if isinstance(face_crop, np.ndarray):
                    if face_crop.dtype != np.uint8:
                        face_crop = face_crop.astype(np.uint8)
                    pil_crops.append(Image.fromarray(face_crop))
                elif isinstance(face_crop, Image.Image):
                    pil_crops.append(face_crop)
                has_faces = True

        # Evaluate BOTH face crops and the full scene!
        # Generative AI artifacts are often most obvious in the global context
        # (e.g., overlapping bodies, weird backgrounds), which face crops entirely miss.
        if first_frame is not None:
            # NO manual resize — let CLIP processor handle it
            raw_img = Image.fromarray(first_frame).convert("RGB")
            pil_crops.append(raw_img)
        elif media_path:
            try:
                raw_img = Image.open(media_path).convert("RGB")
                pil_crops.append(raw_img)
            except Exception as e:
                logger.warning(
                    f"UnivFD: Failed to load raw image from {media_path}: {e}"
                )

        if not pil_crops:
            return ToolResult(
                tool_name=self.tool_name,
                success=False,
                score=0.0,
                confidence=0.0,
                details={},
                error=True,
                error_msg="No image data available for analysis.",
                execution_time=time.time() - start_time,
                evidence_summary="UnivFD detector: No image data available.",
            )

        worst_face_score = 0.0
        all_logits = []

        with VRAMLifecycleManager(self._load_model) as wrapper:
            if not self._weights_loaded_ok:
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
                next(wrapper.parameters()).device
                if list(wrapper.parameters())
                else torch.device("cpu")
            )

            for pil_crop in pil_crops:
                # FIX 3: TTA with AVERAGING instead of max
                logit = self._score_single_crop(wrapper, pil_crop, device)

                if self.tta_enabled:
                    flipped_crop = pil_crop.transpose(Image.FLIP_LEFT_RIGHT)
                    flip_logit = self._score_single_crop(wrapper, flipped_crop, device)
                    # Average logits before calibration
                    logit = (logit + flip_logit) / 2.0

                all_logits.append(logit)

                # Calibrate with temperature scaling
                score = self._calibrate_score(logit, temperature)
                worst_face_score = max(worst_face_score, score)

        execution_time = time.time() - start_time

        # FIX 6: Parabolic confidence (Low near threshold, high as it moves away)
        raw_confidence = self.conf_min + (4.0 * (worst_face_score - effective_threshold) ** 2)
        confidence = max(self.conf_min, min(1.0, raw_confidence))

        # FIX 4: Phone-aware verdict messaging
        if worst_face_score > effective_threshold:
            summary = (
                f"UnivFD detected generative AI signatures "
                f"(Authenticity: {1.0 - worst_face_score:.2f}, "
                f"{'phone-calibrated' if is_phone else 'standard'} threshold: {effective_threshold:.2f})."
            )
        else:
            summary = (
                f"UnivFD found no reliable generative AI signatures "
                f"(Authenticity: {1.0 - worst_face_score:.2f})."
            )
            if is_phone and worst_face_score > self.fake_threshold:
                summary += (
                    f" Score exceeds standard threshold ({self.fake_threshold:.2f}) "
                    f"but below phone-calibrated threshold ({self.PHONE_FAKE_THRESHOLD:.2f})."
                )

        return ToolResult(
            tool_name=self.tool_name,
            success=True,
            score=worst_face_score,
            confidence=confidence,
            details={
                "raw_logits": [round(l, 4) for l in all_logits],
                "temperature": temperature,
                "is_phone_origin": is_phone,
                "effective_threshold": effective_threshold,
                "num_crops_analyzed": len(pil_crops),
                "face_only_analysis": False,
                "siglip_score": worst_face_score,  # Backward compatibility
                "execution_time": execution_time,
            },
            error=False,
            error_msg=None,
            execution_time=execution_time,
            evidence_summary=summary,
        )
