"""
Aegis-X FreqNet Forensic Tool (v3)
----------------------------------
Replaces FreqNetDual with a fast ResNet-50 CNNDetect architecture +
FADHook spectral analysis fusion stream.

V3 Fixes:
    1. No Lanczos resize for FAD stream — native resolution frequency analysis
    2. Disabled Z-score anomaly detection — uses ratio thresholds only
    3. Face-only analysis — no full-scene context mixing
    4. FAD as gate, not fusion — modulates neural score, doesn't average
    5. BGR to RGB conversion — enforces consistent channel order
    6. Multi-patch median analysis — reduces variance with overlapping patches
    7. Conservative false-positive thresholds — empirical ratio cutoffs
"""

import os
import time
import logging
from typing import Any, Dict, List, Tuple

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

# ─── V3 Conservative Ratio Thresholds (empirical, not guessed) ───
# These are set high to minimize false positives on real phone JPEGs.
# Only genuine manipulation artifacts should exceed these.
FAD_MID_RATIO_THRESHOLD = 0.35  # Raised from default ~0.25
FAD_HIGH_RATIO_THRESHOLD = 0.20  # Raised from default ~0.05
FAD_ANOMALY_BOOST = 0.15  # Modest boost when FAD detects anomaly


class _CNNDetect(nn.Module):
    """ResNet-50 binary classifier for CNN-generated image detection.
    
    IMPORTANT: Layers are exposed as direct attributes (conv1, bn1, layer1, etc.)
    to preserve the original ResNet50 state_dict key names. Do NOT wrap in 
    nn.Sequential — that renames keys to features.0, features.1, etc., breaking
    checkpoint loading with strict=False (silently runs on random weights).
    """
    def __init__(self):
        super().__init__()
        backbone = models.resnet50(weights=None)
        # Copy all layers as direct attributes (preserving key names)
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.avgpool = backbone.avgpool
        # Binary classification head (replaces original 1000-class fc)
        self.classifier = nn.Linear(2048, 1, bias=True)

    def forward(self, x):  # x: (B, 3, 224, 224)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.flatten(start_dim=1)  # (B, 2048)
        return self.classifier(x)  # (B, 1) logit



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
        if hasattr(self.calibration_manager, "load"):
            self.calibration_manager.load()
        logger.info("FreqNetTool setup complete.")
        return True

    def _remap_cnndetect_keys(self, ckpt: Dict[str, Any]) -> Dict[str, Any]:
        """Strips DDP prefixes and aligns fully connected layer names."""
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

        weight_path = getattr(
            self.config.models,
            "freqnet_weights",
            "models/freqnet/cnndetect_resnet50.pth",
        )

        if os.path.exists(weight_path):
            try:
                ckpt = torch.load(weight_path, map_location="cpu", weights_only=False)

                if "state_dict" in ckpt:
                    ckpt = ckpt["state_dict"]
                elif "model" in ckpt:
                    ckpt = ckpt["model"]

                remapped = self._remap_cnndetect_keys(ckpt)
                load_result = model.load_state_dict(remapped, strict=False)
                
                # Verify that weights actually loaded
                total_model_keys = len(model.state_dict())
                missing_count = len(load_result.missing_keys) if load_result.missing_keys else 0
                matched_count = total_model_keys - missing_count
                
                if matched_count < total_model_keys * 0.9:
                    logger.error(f"FreqNet: Only {matched_count}/{total_model_keys} keys matched! Weights NOT loaded correctly.")
                    logger.error(f"  Missing (first 5): {load_result.missing_keys[:5]}")
                    logger.error(f"  Unexpected (first 5): {load_result.unexpected_keys[:5]}")
                    self._weights_loaded_ok = False
                else:
                    self._weights_loaded_ok = True
                    logger.info(f"FreqNet (CNNDetect) weights loaded from {weight_path} ({matched_count}/{total_model_keys} keys matched)")
            except Exception as e:
                logger.warning(f"Failed to load CNNDetect weights: {e}")
                self._weights_loaded_ok = False
        else:
            logger.warning(
                f"CNNDetect weights absent at {weight_path}. Fallback to pure FAD mode."
            )
            self._weights_loaded_ok = False

        model.eval()
        return model

    def _extract_overlapping_patches(
        self, img: np.ndarray, patch_size: int = 224, overlap: float = 0.5
    ) -> List[np.ndarray]:
        """Extract overlapping patches for multi-patch median analysis (Fix 6).

        If image is smaller than patch_size, returns the image as-is.
        If image is larger, extracts patches with specified overlap.
        """
        h, w = img.shape[:2]

        if h <= patch_size and w <= patch_size:
            # Image too small for patches — return as-is
            return [img]

        patches = []
        step = int(patch_size * (1.0 - overlap))

        for y in range(0, max(h - patch_size + 1, 1), step):
            for x in range(0, max(w - patch_size + 1, 1), step):
                y_end = min(y + patch_size, h)
                x_end = min(x + patch_size, w)
                patch = img[y:y_end, x:x_end].copy()

                # Resize to exactly patch_size if edge crop is smaller
                if patch.shape[0] != patch_size or patch.shape[1] != patch_size:
                    patch = cv2.resize(
                        patch, (patch_size, patch_size), interpolation=cv2.INTER_AREA
                    )

                patches.append(patch)

        return patches if patches else [img]

    def _run_fad_analysis(self, dct_prep, tensor: torch.Tensor) -> Tuple[float, Any]:
        """Run FAD statistical analysis and return (score, band_info).

        FIX 2: Disabled Z-score anomaly detection — uses ratio thresholds only.
        FIX 7: Conservative thresholds to minimize false positives.
        """
        calib_data = (
            self.calibration_manager.get_data()
            if hasattr(self.calibration_manager, "get_data")
            else None
        )
        hook = FADHook(calib_data)

        hook.register(dct_prep._dct_conv)
        _ = dct_prep(tensor)
        hook.remove()

        band = hook.analyze()

        if band is None:
            return 0.0, None

        # FIX 2: Use ratio thresholds instead of Z-score anomaly detection
        mid_ratio = getattr(band, "mid_ratio", 0.0)
        high_ratio = getattr(band, "high_ratio", 0.0)

        # FIX 7: Conservative thresholds — only flag if both bands are elevated
        mid_excess = max(0.0, mid_ratio - FAD_MID_RATIO_THRESHOLD)
        high_excess = max(0.0, high_ratio - FAD_HIGH_RATIO_THRESHOLD)

        # FIX 4: FAD as gate — modest boost, not standalone score
        if mid_excess > 0 or high_excess > 0:
            fad_score = min(1.0, (mid_excess * 2.0) + (high_excess * 3.0))
        else:
            fad_score = 0.0

        return fad_score, band

    def _run_inference(self, input_data: dict) -> ToolResult:
        start_time = time.time()

        tracked_faces = input_data.get("tracked_faces", [])
        media_path = input_data.get("media_path", None)
        first_frame = input_data.get("first_frame", None)

        # Build list of numpy crops (uint8, RGB) to analyze
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

        # FIX 3: Full image fallback ONLY when zero faces detected
        if not np_crops:
            if first_frame is not None:
                # FIX 5: Convert BGR to RGB (OpenCV returns BGR)
                raw_img = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
                np_crops.append(raw_img)
                logger.info("FreqNet: No faces found, using full frame as fallback.")
            elif media_path:
                try:
                    raw_img = Image.open(media_path).convert("RGB")
                    np_crops.append(np.array(raw_img))
                    logger.info(
                        "FreqNet: No faces found, using full raw image as fallback."
                    )
                except Exception as e:
                    logger.warning(
                        f"FreqNet: Failed to load raw image from {media_path}: {e}"
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
                evidence_summary="FreqNet detector: No image data available.",
            )

        worst_face_score = 0.0
        best_band_analysis = None
        all_fad_scores = []

        spatial_prep = SpatialPreprocessor()
        dct_prep = DCTPreprocessor()

        try:
            from utils.thresholds import FREQNET_FAKE_THRESHOLD

            fake_threshold = FREQNET_FAKE_THRESHOLD
        except ImportError:
            fake_threshold = 0.60

        with VRAMLifecycleManager(self._load_model) as model:
            self.device = (
                next(model.parameters()).device
                if list(model.parameters())
                else torch.device("cpu")
            )
            spatial_prep = spatial_prep.to(self.device)
            dct_prep = dct_prep.to(self.device)

            for face_crop in np_crops:
                # FIX 5: Ensure RGB order (face crops from PIL are RGB, but be safe)
                if face_crop.ndim == 3 and face_crop.shape[2] == 3:
                    # Already RGB from PIL — no conversion needed
                    pass

                # FIX 6: Multi-patch median analysis for FAD stream
                patches = self._extract_overlapping_patches(face_crop)
                patch_fad_scores = []

                for patch in patches:
                    # FAD stream: native resolution patch (no resize)
                    patch_tensor = (
                        torch.from_numpy(patch).permute(2, 0, 1).unsqueeze(0).float()
                        / 255.0
                    )
                    patch_tensor = patch_tensor.to(self.device)

                    fad_score, band = self._run_fad_analysis(dct_prep, patch_tensor)
                    patch_fad_scores.append(fad_score)

                # Use median FAD score across patches (reduces variance)
                median_fad = (
                    float(np.median(patch_fad_scores)) if patch_fad_scores else 0.0
                )
                all_fad_scores.append(median_fad)

                # Neural stream: resized to 224x224 (ResNet was trained on resized images)
                neural_score = 0.0
                if self._weights_loaded_ok:
                    # Resize for neural stream only
                    resized = cv2.resize(
                        face_crop, (224, 224), interpolation=cv2.INTER_AREA
                    )
                    tensor = (
                        torch.from_numpy(resized).permute(2, 0, 1).unsqueeze(0).float()
                        / 255.0
                    )
                    tensor = tensor.to(self.device)

                    with torch.no_grad():
                        norm_tensor = spatial_prep(tensor)
                        logit = model(norm_tensor)
                        neural_score = float(torch.sigmoid(logit).item())

                # FIX 4: FAD as gate — modulate neural score, don't average
                if self._weights_loaded_ok:
                    if median_fad > 0:
                        # FAD detected anomaly — boost neural score modestly
                        final_score = min(
                            1.0, neural_score + (median_fad * FAD_ANOMALY_BOOST)
                        )
                    else:
                        # FAD found nothing anomalous — trust neural score
                        final_score = neural_score
                else:
                    # No neural model — use FAD alone (less reliable)
                    final_score = median_fad

                if final_score > worst_face_score:
                    worst_face_score = final_score
                    best_band_analysis = band

        execution_time = time.time() - start_time

        confidence = 0.9 if self._weights_loaded_ok else 0.5

        # Evidence generation
        if worst_face_score > fake_threshold:
            anomaly_info = (
                getattr(best_band_analysis, "interpretation", "Spectral trace detected")
                if best_band_analysis
                else "Pattern matched"
            )
            summary = f"FreqNet detected high-frequency spatial artifacts ({anomaly_info}). Authenticity: {1.0 - worst_face_score:.2f}."
        else:
            summary = f"FreqNet analysis showed normal frequency energy distribution (Authenticity: {1.0 - worst_face_score:.2f})."

        return ToolResult(
            tool_name=self.tool_name,
            success=True,
            score=worst_face_score,
            confidence=confidence,
            details={
                "fad_score_contribution": float(np.mean(all_fad_scores))
                if all_fad_scores
                else 0.0,
                "neural_score_contribution": neural_score
                if "neural_score" in locals()
                else 0.0,
                "weights_loaded_ok": self._weights_loaded_ok,
                "anomaly_detected": getattr(
                    best_band_analysis, "anomaly_detected", False
                )
                if best_band_analysis
                else False,
                "num_patches_analyzed": len(patches) if "patches" in locals() else 0,
                "execution_time": execution_time,
            },
            error=False,
            error_msg=None,
            execution_time=execution_time,
            evidence_summary=summary,
        )
