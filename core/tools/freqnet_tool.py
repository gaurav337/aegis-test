"""
Aegis-X FreqNet Forensic Tool (v3.1 - Audit Corrected)
ResNet-50 CNNDetect + FADHook spectral analysis fusion stream.

Key Fixes:
1. S-04: Removed hardcoded `and False` — FAD stream now safely modulates neural score.
2. C-06: Confidence is now epistemic (based on signal quality + weight load), not binary.
3. Native-Resolution FAD: FAD stream runs on native crops; neural stream uses 224px resize.
4. RGB/BGR Enforcement: Explicit channel validation on fallback frames.
5. Multi-Patch Median: Reduces variance from localized JPEG artifacts.
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

logger = logging.getLogger(__name__)  # ✅ FIX: Standard logging initialization

class _CNNDetect(nn.Module):
    """ResNet-50 binary classifier for CNN-generated image detection."""
    def __init__(self):
        super().__init__()
        backbone = models.resnet50(weights=None)
        # Preserve original state_dict key names
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.avgpool = backbone.avgpool
        self.classifier = nn.Linear(2048, 1, bias=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.flatten(start_dim=1)
        return self.classifier(x)

class FreqNetTool(BaseForensicTool):
    @property
    def tool_name(self) -> str:
        return "run_freqnet"

    def __init__(self):
        super().__init__()
        self.requires_gpu = True
        self.config = AegisConfig()
        self.fusion_cfg = self.config.freqnet_fusion
        self.calibration_manager = CalibrationManager()
        self._weights_loaded_ok = False
        self.normalize = models.resnet50(weights=None)  # Dummy ref for normalize params

    def setup(self):
        if hasattr(self.calibration_manager, "load"):
            self.calibration_manager.load()
        logger.info("FreqNetTool setup complete.")
        return True

    def _remap_cnndetect_keys(self, ckpt: Dict[str, Any]) -> Dict[str, Any]:
        remapped = {}
        for k, v in ckpt.items():
            new_k = k.replace("module.", "").replace("model.", "")
            if "fc." in new_k:
                new_k = new_k.replace("fc.", "classifier.", 1)
            remapped[new_k] = v
        return remapped

    def _load_model(self) -> torch.nn.Module:
        logger.info("Loading CNNDetect ResNet-50 architecture...")
        model = _CNNDetect()
        weight_path = getattr(self.config.models, "freqnet_weights", "models/freqnet/cnndetect_resnet50.pth")

        if os.path.exists(weight_path):
            try:
                ckpt = torch.load(weight_path, map_location="cpu", weights_only=False)
                if "state_dict" in ckpt: ckpt = ckpt["state_dict"]
                elif "model" in ckpt: ckpt = ckpt["model"]

                remapped = self._remap_cnndetect_keys(ckpt)
                load_result = model.load_state_dict(remapped, strict=False)
                matched = len(model.state_dict()) - len(load_result.missing_keys or [])
                total = len(model.state_dict())
                
                if matched >= total * 0.9:
                    self._weights_loaded_ok = True
                    logger.info(f"FreqNet weights loaded ({matched}/{total} keys)")
                else:
                    logger.warning(f"FreqNet weight match too low ({matched}/{total}). Random init.")
            except Exception as e:
                logger.error(f"Failed to load FreqNet weights: {e}")
                self._weights_loaded_ok = False
        else:
            logger.warning("FreqNet weights missing. Fallback to FAD-only mode.")
            self._weights_loaded_ok = False

        model.eval()
        return model

    def _extract_overlapping_patches(self, img: np.ndarray, patch_size: int = 224, overlap: float = 0.5) -> List[np.ndarray]:
        h, w = img.shape[:2]
        if h <= patch_size and w <= patch_size:
            return [img]
        
        patches, step = [], int(patch_size * (1.0 - overlap))
        for y in range(0, max(h - patch_size + 1, 1), step):
            for x in range(0, max(w - patch_size + 1, 1), step):
                patch = img[y:y+patch_size, x:x+patch_size].copy()
                if patch.shape[0] != patch_size or patch.shape[1] != patch_size:
                    patch = cv2.resize(patch, (patch_size, patch_size), interpolation=cv2.INTER_AREA)
                patches.append(patch)
        return patches if patches else [img]

    def _run_fad_analysis(self, dct_prep, tensor: torch.Tensor) -> Tuple[float, Any]:
        calib_data = self.calibration_manager.get_data() if hasattr(self.calibration_manager, "get_data") else None
        hook = FADHook(calib_data)
        hook.register(dct_prep._dct_conv)
        _ = dct_prep(tensor)
        hook.remove()
        band = hook.analyze()
        
        if band is None:
            return 0.0, None
            
        mid_ratio = getattr(band, "mid_ratio", 0.0)
        high_ratio = getattr(band, "high_ratio", 0.0)
        
        # Conservative empirical thresholds from thresholds.py
        mid_excess = max(0.0, mid_ratio - self.fusion_cfg.mid_excess_threshold)
        high_excess = max(0.0, high_ratio - self.fusion_cfg.high_excess_threshold)
        
        if mid_excess > 0 or high_excess > 0:
            fad_score = min(1.0, (mid_excess * self.fusion_cfg.mid_multiplier) + (high_excess * self.fusion_cfg.high_multiplier))
        else:
            fad_score = 0.0
        return fad_score, band

    def _calibrate_confidence(self, neural_score: float, fad_score: float, weights_ok: bool) -> float:
        """Returns epistemic confidence: probability that reported score is reliable."""
        if not weights_ok:
            return 0.2  # Low confidence when running blind
        # High confidence when neural & FAD agree or score is far from boundary
        agreement = 1.0 - abs(neural_score - fad_score)
        margin = abs(neural_score - 0.5) * 2.0  # 0.0 at 0.5, 1.0 at 0.0 or 1.0
        return min(0.95, max(0.4, 0.5 * margin + 0.3 * agreement + 0.2))

    def _run_inference(self, input_data: Dict[str, Any]) -> ToolResult:
        start_time = time.time()
        tracked_faces = input_data.get("tracked_faces", [])
        first_frame = input_data.get("first_frame", None)
        media_path = input_data.get("media_path", None)

        np_crops = []
        if tracked_faces:
            for face in tracked_faces:
                face_crop = getattr(face, "face_crop_224", None)
                if face_crop is None: continue
                if isinstance(face_crop, Image.Image): face_crop = np.array(face_crop)
                if face_crop.dtype != np.uint8: face_crop = face_crop.astype(np.uint8)
                np_crops.append(face_crop)

        # Fallback to full frame (ensure RGB order)
        if not np_crops:
            if first_frame is not None:
                raw = first_frame if first_frame.shape[2] == 3 else first_frame
                if raw.ndim == 3 and raw.shape[2] == 3 and raw[:,:,0].mean() < raw[:,:,2].mean() * 0.9:
                    raw = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
                np_crops.append(raw)
            elif media_path:
                try:
                    img = Image.open(media_path).convert("RGB")
                    np_crops.append(np.array(img))
                except Exception: pass

        if not np_crops:
            return ToolResult(tool_name=self.tool_name, success=False, real_prob=0.5, confidence=0.0, details={}, error=True, error_msg="No image data", execution_time=0.0, evidence_summary="No image data available.")

        spatial_prep = SpatialPreprocessor()
        dct_prep = DCTPreprocessor()
        worst_face_fake_prob, best_fad, best_band, all_neural, all_fad = 0.0, 0.0, 0.0, [], []

        try:
            from utils.thresholds import FREQNET_FAKE_THRESHOLD
            fake_threshold = FREQNET_FAKE_THRESHOLD
        except ImportError:
            fake_threshold = 0.50

        with VRAMLifecycleManager(self._load_model) as model:
            if not self._weights_loaded_ok:
                logger.warning("FreqNet running in degraded FAD-only mode.")
                
            device = next(model.parameters()).device if list(model.parameters()) else torch.device("cpu")
            spatial_prep = spatial_prep.to(device)
            dct_prep = dct_prep.to(device)

            for face_crop in np_crops:
                patches = self._extract_overlapping_patches(face_crop)
                patch_fads = []
                
                for patch in patches:
                    tensor = torch.from_numpy(patch).permute(2, 0, 1).unsqueeze(0).float() / 255.0
                    tensor = tensor.to(device)
                    fad_score, band = self._run_fad_analysis(dct_prep, tensor)
                    patch_fads.append(fad_score)
                
                median_fad = float(np.median(patch_fads)) if patch_fads else 0.0
                all_fad.append(median_fad)

                # Neural stream (always resized to 224 for ResNet training compatibility)
                neural_score = 0.0
                if self._weights_loaded_ok:
                    resized = cv2.resize(face_crop, (224, 224), interpolation=cv2.INTER_AREA)
                    tensor = torch.from_numpy(resized).permute(2, 0, 1).unsqueeze(0).float() / 255.0
                    tensor = tensor.to(device)
                    with torch.no_grad():
                        logit = model(spatial_prep(tensor))
                        neural_score = float(torch.sigmoid(logit).item())
                all_neural.append(neural_score)

                # ✅ FIX S-04: Calibrated FAD Gating Logic
                # FAD only boosts if: (A) Neural is already suspicious (>0.35) OR (B) FAD signal is very strong (>0.40)
                # This prevents FAD false positives on real JPEGs from overriding clean neural scores.
                if self._weights_loaded_ok:
                    if (neural_score > 0.35 and median_fad > 0.15) or median_fad > 0.40:
                        boost = min(0.25, median_fad * self.fusion_cfg.fad_weight * 0.8)
                        final_score = min(1.0, neural_score + boost)
                    else:
                        final_score = neural_score
                else:
                    # FAD-only mode: highly conservative, requires strong spectral evidence
                    final_score = median_fad if median_fad > 0.45 else 0.0

                if final_score > worst_face_fake_prob:
                    worst_face_fake_prob = final_score
                    best_fad = median_fad
                    best_band = band

        worst_face_real_prob = 1.0 - worst_face_fake_prob
        confidence = self._calibrate_confidence(worst_face_fake_prob, best_fad, self._weights_loaded_ok)
        execution_time = time.time() - start_time

        if worst_face_fake_prob > fake_threshold:
            anomaly = getattr(best_band, "interpretation", "Spectral trace detected") if best_band else "Pattern matched"
            summary = f"FreqNet detected high-frequency spatial artifacts ({anomaly}). Authenticity: {worst_face_real_prob:.2f}."
        else:
            summary = f"FreqNet analysis showed normal frequency energy distribution (Authenticity: {worst_face_real_prob:.2f})."

        return ToolResult(
            tool_name=self.tool_name, success=True, real_prob=float(worst_face_real_prob),
            confidence=float(confidence),
            details={
                "neural_score_contribution": float(np.mean(all_neural)) if all_neural else 0.0,
                "fad_score_contribution": float(np.mean(all_fad)) if all_fad else 0.0,
                "weights_loaded_ok": self._weights_loaded_ok,
                "anomaly_detected": getattr(best_band, "anomaly_detected", False) if best_band else False,
                "num_patches_analyzed": len(patches) if "patches" in locals() else 0,
                "execution_time": execution_time,
            },
            error=False, error_msg=None, execution_time=execution_time,
            evidence_summary=summary
        )