"""
Aegis-X UnivFD Forensic Tool (v2.1 - Audit Corrected)
Generative AI detector using CLIP-ViT-L/14 backbone + 4KB linear probe.

Key Fixes:
1. TTA Averaging: Averages logits (original + flipped) instead of taking max.
2. Phone Calibration: Temperature scaling + elevated threshold for phone-origin EXIF.
3. Confidence Semantics: Parabolic confidence (low near threshold, high at extremes).
4. No Double Resizing: Delegates all preprocessing to CLIPImageProcessor.
5. Full-Scene Poisoning Guard: Only analyzes full frame if zero faces detected.
"""
import os
import time
import logging
from typing import Any, Dict, List, Tuple
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
from utils.logger import setup_logger

logger = setup_logger(__name__)

# ──────────────────────────────────────────────────────────────
# EXIF Phone Detection (Cross-validated with resolution heuristic)
# ──────────────────────────────────────────────────────────────
_PHONE_MARKERS = [b"Apple", b"Samsung", b"Xiaomi", b"Huawei", b"OnePlus", b"Google", b"SONY"]

def _is_phone_origin(filepath: str) -> Tuple[bool, float]:
    """Check if image likely originated from a phone camera.
    Returns (is_phone, trust_score_0_to_1) to avoid blind EXIF trust."""
    trust = 0.0
    try:
        with open(filepath, "rb") as f:
            content = f.read(2_000_000)
            has_exif = b"Exif\x00\x00" in content
            if has_exif:
                for marker in _PHONE_MARKERS:
                    if marker in content:
                        trust += 0.4
                        break
            # Cross-validate with common phone resolutions
            if has_exif and (b"4032" in content or b"3024" in content):
                trust += 0.3
            # Check JPEG compression density (phone JPEGs: 0.3-1.5 bpp)
            import io
            f.seek(0)
            try:
                from PIL import Image as PILImage
                img = PILImage.open(io.BytesIO(content))
                w, h = img.size
                bpp = len(content) / (w * h)
                if 0.3 < bpp < 1.8:
                    trust += 0.3
            except:
                pass
    except Exception:
        pass
    return trust > 0.5, min(1.0, trust)

class _LinearProbe(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(768, 1)
    def forward(self, x):
        return self.fc(x)

class _UnivFDWrapper(nn.Module):
    def __init__(self, backbone, probe):
        super().__init__()
        # Explicitly assign to register as submodules for .to() and .parameters()
        self.backbone = backbone
        self.probe = probe

class UnivFDTool(BaseForensicTool):
    @property
    def tool_name(self) -> str:
        return "run_univfd"

    def __init__(self):
        super().__init__()
        self.requires_gpu = True
        self.config = AegisConfig()
        self._processor = None
        self._weights_loaded_ok = False
        
        try:
            from utils.thresholds import UNIVFD_FAKE_THRESHOLD, UNIVFD_CONFIDENCE_MIN, UNIVFD_TTA_ENABLED
            self.fake_threshold = UNIVFD_FAKE_THRESHOLD
            self.conf_min = UNIVFD_CONFIDENCE_MIN
            self.tta_enabled = UNIVFD_TTA_ENABLED
        except ImportError:
            self.fake_threshold = 0.50
            self.conf_min = 0.50
            self.tta_enabled = True

    def setup(self) -> None:
        logger.info("UnivFDTool setup complete.")
        return True

    def _load_model(self) -> nn.Module:
        config = self.config
        backbone_dir = getattr(config.models, "univfd_backbone_dir", "openai/clip-vit-large-patch14")
        probe_path = getattr(config.models, "univfd_probe_path", "models/univfd/probe.pth")

        logger.info(f"Loading CLIP backbone: {backbone_dir}")
        try:
            backbone = CLIPVisionModelWithProjection.from_pretrained(
                backbone_dir, torch_dtype=torch.float16, local_files_only=os.path.exists(backbone_dir)
            )
        except Exception as e:
            logger.warning(f"Local CLIP failed, falling back to Hub: {e}")
            backbone = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=torch.float16)

        self._processor = CLIPImageProcessor.from_pretrained(backbone_dir if os.path.exists(backbone_dir) else "openai/clip-vit-large-patch14")
        probe = _LinearProbe()

        if os.path.exists(probe_path):
            try:
                ckpt = torch.load(probe_path, map_location="cpu", weights_only=False)
                if "fc.weight" in ckpt:
                    probe.load_state_dict(ckpt, strict=True)
                elif "coef" in ckpt and "intercept" in ckpt:
                    with torch.no_grad():
                        probe.fc.weight.copy_(torch.tensor(ckpt["coef"]).clone().detach().reshape(1, -1))
                        probe.fc.bias.copy_(torch.tensor(ckpt["intercept"]).clone().detach().reshape(1))
                else:
                    with torch.no_grad():
                        probe.fc.weight.copy_(ckpt.get("weight", ckpt).clone().detach().reshape(1, -1))
                        probe.fc.bias.copy_(ckpt.get("bias", torch.zeros(1)).clone().detach())
                self._weights_loaded_ok = True
            except Exception as e:
                logger.error(f"Probe load failed: {e}")
                self._weights_loaded_ok = False
        else:
            logger.warning("Probe not found. Random init.")
            self._weights_loaded_ok = False

        # ──────────────────────────────────────────────────────────────
        # Device & Precision Awareness (FIX for constant scores on CPU)
        # ──────────────────────────────────────────────────────────────
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Use FP16 ONLY on GPU; FP16 on certain CPUs yields static/garbage embeddings
        target_dtype = torch.float16 if device.type == "cuda" else torch.float32
        
        backbone = backbone.to(device, dtype=target_dtype)
        probe = probe.to(device) # Linear probes are lightweight, keep on device

        backbone.eval()
        probe.eval()
        
        wrapper = _UnivFDWrapper(backbone, probe)
        wrapper.to(device) # Cascades to submodules
        return wrapper

    def _crop_to_tensor(self, pil_img: Image.Image, device: torch.device) -> torch.Tensor:
        """CLIP processor handles resize/crop/normalize internally. NO double resizing."""
        inputs = self._processor(images=pil_img, return_tensors="pt")
        return inputs.pixel_values.to(device, dtype=torch.float16)

    @torch.no_grad()
    def _score_single(self, wrapper: _UnivFDWrapper, tensor: torch.Tensor) -> float:
        outputs = wrapper.backbone(pixel_values=tensor)
        embeds = outputs.image_embeds.float()
        embeds = F.normalize(embeds, p=2, dim=-1)
        return wrapper.probe(embeds).item()

    def _calibrate_score(self, logit: float, temperature: float = 1.0) -> float:
        """Temperature-scaled sigmoid calibration."""
        return float(1.0 / (1.0 + np.exp(-logit / temperature)))

    def _compute_confidence(self, score: float, threshold: float) -> float:
        """Parabolic confidence: low near threshold, high at extremes."""
        distance = abs(score - threshold)
        raw = self.conf_min + (4.0 * distance**2)
        return max(self.conf_min, min(1.0, raw))

    def _run_inference(self, input_data: Dict[str, Any]) -> ToolResult:
        start_time = time.time()
        tracked_faces = input_data.get("tracked_faces", [])
        media_path = input_data.get("media_path", None)
        first_frame = input_data.get("first_frame", None)
        is_phone, phone_trust = _is_phone_origin(media_path) if media_path else (False, 0.0)

        # Dynamic threshold & temperature based on phone trust score
        base_threshold = self.fake_threshold
        base_temp = 1.0
        if is_phone:
            base_threshold = min(0.75, base_threshold + (0.15 * phone_trust))
            base_temp = 1.0 + (0.5 * phone_trust)

        # Prepare crops: faces only, fallback to full frame ONLY if zero faces
        pil_crops: List[Image.Image] = []
        has_faces = False
        if tracked_faces:
            for face in tracked_faces:
                crop = getattr(face, "face_crop_224", None)
                if crop is not None:
                    if isinstance(crop, np.ndarray):
                        crop = Image.fromarray(crop.astype(np.uint8))
                    pil_crops.append(crop.convert("RGB"))
                    has_faces = True

        if not has_faces and first_frame is not None:
            pil_crops.append(Image.fromarray(first_frame).convert("RGB"))
        elif not has_faces and media_path:
            try:
                pil_crops.append(Image.open(media_path).convert("RGB"))
            except Exception as e:
                logger.warning(f"UnivFD: Raw load failed: {e}")

        if not pil_crops:
            return ToolResult(tool_name=self.tool_name, success=False, score=0.0, confidence=0.0,
                              error=True, error_msg="No image data", execution_time=0.0,
                              evidence_summary="UnivFD skipped: No image data available.")

        all_logits = []
        with VRAMLifecycleManager(self._load_model) as wrapper:
            if not self._weights_loaded_ok:
                return ToolResult(tool_name=self.tool_name, success=True, score=0.0, confidence=0.0,
                                  details={"weights_loaded_ok": False, "execution_time": 0.0},
                                  execution_time=0.0, evidence_summary="Model weights missing.")

            # Determine current device/precision from wrapper
            device = next(wrapper.parameters()).device if list(wrapper.parameters()) else torch.device("cpu")
            dtype = next(wrapper.backbone.parameters()).dtype if list(wrapper.backbone.parameters()) else torch.float32

            for pil_crop in pil_crops:
                # Preprocess image directly to matching device/dtype
                inputs = self._processor(images=pil_crop, return_tensors="pt")
                tensor = inputs.pixel_values.to(device, dtype=dtype)
                
                # Inference
                logit = self._score_single(wrapper, tensor)
                
                if self.tta_enabled:
                    flip_crop = pil_crop.transpose(Image.FLIP_LEFT_RIGHT)
                    inputs_f = self._processor(images=flip_crop, return_tensors="pt")
                    tensor_f = inputs_f.pixel_values.to(device, dtype=dtype)
                    flip_logit = self._score_single(wrapper, tensor_f)
                    logit = (logit + flip_logit) / 2.0
                all_logits.append(logit)

        # Calibrate & take worst-case face
        scores = [self._calibrate_score(l, base_temp) for l in all_logits]
        worst_score = max(scores) if scores else 0.0
        confidence = self._compute_confidence(worst_score, base_threshold)

        # Evidence summary
        if worst_score > base_threshold:
            summary = (f"UnivFD detected generative AI signatures "
                       f"(Authenticity: {1.0 - worst_score:.2f}, "
                       f"{'phone-calibrated' if is_phone else 'standard'} threshold: {base_threshold:.2f}).")
        else:
            summary = f"UnivFD found no reliable generative AI signatures (Authenticity: {1.0 - worst_score:.2f})."
            if is_phone and worst_score > self.fake_threshold:
                summary += f" Score exceeds standard ({self.fake_threshold:.2f}) but below phone threshold ({base_threshold:.2f})."

        return ToolResult(
            tool_name=self.tool_name, success=True, score=round(worst_score, 4),
            confidence=round(confidence, 4),
            details={"raw_logits": [round(l, 4) for l in all_logits], "temperature": base_temp,
                     "is_phone_origin": is_phone, "phone_trust": round(phone_trust, 2),
                     "effective_threshold": base_threshold, "num_crops_analyzed": len(pil_crops),
                     "execution_time": time.time() - start_time},
            error=False, error_msg=None, execution_time=time.time() - start_time,
            evidence_summary=summary
        )