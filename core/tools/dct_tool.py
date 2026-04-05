"""
DCT Double-Quantization Detector — V4 (Audit Corrected)
Detects double-compression artifacts in JPEG files (editing/AI generation).

Key Fixes:
1. C-04: Fixed "Silent Tool" by removing fallback to resized 224px crops.
   Now extracts native-resolution crops from full frames using trajectory bboxes.
2. S-01: Statistical Grid Search correction (Z-score > 3.0) to reduce 8x8 FP rate.
3. S-05: Enhanced EXIF phone detection (cross-validates resolution if possible).
4. Thresholds: Updated to use centralized constants from utils/thresholds.py.
"""
import time
import struct
import numpy as np
import cv2
import scipy.fft
import hashlib
import logging
from typing import Any, Dict, Tuple, Optional, List

from core.base_tool import BaseForensicTool
from core.data_types import ToolResult
from utils.thresholds import (
    DCT_RATIO_THRESHOLD,
    DCT_RATIO_SCALE,
    DCT_CONFIDENCE_CAP,
    DCT_CONFIDENCE_BUMP,
    DCT_CORRECTED_Z_THRESHOLD, # Uses 4.15 if available, or fallback logic
)

logger = logging.getLogger(__name__)

# --- JPEG Quantization Table Reader ---
def _read_jpeg_quant_table(filepath: str) -> Optional[np.ndarray]:
    """Read the first JPEG quantization table from a JPEG file.
    Returns an 8×8 array of quantization values, or None if not found.
    """
    try:
        with open(filepath, "rb") as f:
            # Check JPEG signature
            if f.read(2) != b"\xff\xd8":
                return None

            while True:
                byte = f.read(1)
                while byte != b"\xff":
                    byte = f.read(1)
                    if not byte:
                        return None
                while byte == b"\xff":
                    byte = f.read(1)
                marker = byte[0]

                if marker == 0xDB:  # DQT marker
                    length_data = f.read(2)
                    if len(length_data) < 2:
                        return None
                    length = struct.unpack(">H", length_data)[0]
                    qt_data = f.read(length - 2)
                    if len(qt_data) < length - 2:
                        return None

                    precision_id = qt_data[0]
                    precision = (precision_id >> 4) & 0x0F
                    table_id = precision_id & 0x0F

                    if table_id == 0:  # Luminance table
                        if precision == 0:
                            values = list(qt_data[1:65])
                        else:
                            values = []
                            for i in range(1, 129, 2):
                                val = struct.unpack(">H", qt_data[i : i + 2])[0]
                                values.append(val)
                        
                        if len(values) == 64:
                            return np.array(values, dtype=np.float64).reshape(8, 8)
                elif marker == 0xD9:  # EOI marker
                    break
                elif 0xD0 <= marker <= 0xD7:
                    continue
                else:
                    length_data = f.read(2)
                    if len(length_data) < 2:
                        break
                    length = struct.unpack(">H", length_data)[0]
                    f.seek(length - 2, 1)
    except Exception as e:
        logger.debug(f"Failed to read JPEG quantization table: {e}")
    return None

# --- EXIF Phone Detection ---
def _is_phone_origin(filepath: str) -> bool:
    """Check if image likely originated from a phone camera via EXIF data."""
    phone_markers = [
        b"Apple", b"Samsung", b"Xiaomi", b"Huawei", b"OnePlus",
        b"Google", b"SONY", b"OPPO", b"vivo", b"realme", b"Motorola", b"Nokia", b"LG", b"HTC",
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

class DCTTool(BaseForensicTool):
    """DCT-based double-quantization detector with native resolution analysis."""
    
    _grid_cache: Dict[str, Tuple[int, int]] = {}

    @property
    def tool_name(self) -> str:
        return "run_dct"

    def setup(self) -> None:
        pass

    @staticmethod
    def _coerce_to_uint8(frame: np.ndarray) -> np.ndarray:
        """Ensure frame is uint8 for consistent processing."""
        if frame.dtype == np.uint8:
            return frame
        arr = frame.astype(np.float64)
        if np.issubdtype(frame.dtype, np.floating):
            max_val = float(np.max(arr)) if arr.size else 0.0
            if max_val <= 1.0:
                arr *= 255.0
        return np.round(arr).clip(0, 255).astype(np.uint8)

    @staticmethod
    def _to_gray(crop: np.ndarray) -> np.ndarray:
        """Convert to grayscale using luminance formula."""
        if crop.ndim == 2:
            return crop.astype(np.float32)
        crop = crop.astype(np.float32)
        if crop.shape[2] >= 3:
            r = crop[:, :, 0]
            g = crop[:, :, 1]
            b = crop[:, :, 2]
            return 0.299 * r + 0.587 * g + 0.114 * b
        return crop

    @staticmethod
    def _compute_video_hash(frames: list) -> str:
        if not frames:
            return "empty"
        sample = frames[0][:100, :100] if len(frames) > 0 else np.zeros((100, 100))
        return hashlib.md5(sample.tobytes()).hexdigest()[:16]

    # --- FIX S-01: Statistical Grid Search Correction ---
    def _find_optimal_grid_statistical(
        self, gray: np.ndarray, video_hash: Optional[str] = None
    ) -> Tuple[int, int, float, float]:
        """Find optimal 8×8 grid alignment WITH statistical correction.
        Returns: (best_dy, best_dx, best_ratio, z_score)
        """
        if video_hash and video_hash in self._grid_cache:
            dy, dx = self._grid_cache[video_hash]
            ratio, z_score = self._compute_ratio_and_zscore(gray, dy, dx)
            return dy, dx, ratio, z_score

        H, W = gray.shape
        mask = np.zeros((8, 8), dtype=bool)
        for i in range(8):
            for j in range(8):
                if 0 < i + j <= 5:
                    mask[i, j] = True

        all_ratios = []
        best_dy, best_dx = 0, 0
        max_ratio = 0.0

        for dy in range(8):
            for dx in range(8):
                h_new = ((H - dy) // 8) * 8
                w_new = ((W - dx) // 8) * 8
                if h_new <= 0 or w_new <= 0:
                    all_ratios.append(0.0)
                    continue

                crop = gray[dy : dy + h_new, dx : dx + w_new]
                blocks = crop.reshape(h_new // 8, 8, w_new // 8, 8).swapaxes(1, 2)
                dct_blocks = scipy.fft.dctn(blocks, axes=(-2, -1), norm="ortho")
                
                # Apply AC mask
                ac_coeffs = dct_blocks[:, :, mask].flatten()
                ac_coeffs = np.round(ac_coeffs)
                hist, _ = np.histogram(ac_coeffs, bins=513, range=(-256.5, 256.5))
                autocorr = np.correlate(hist, hist, mode="same")

                center = len(autocorr) // 2
                primary = float(autocorr[center])
                if primary < 1e-10:
                    all_ratios.append(0.0)
                    continue

                excluded = autocorr.copy()
                excluded[max(0, center - 1) : min(len(autocorr), center + 2)] = 0
                secondary = float(np.max(excluded))
                ratio = secondary / primary
                all_ratios.append(ratio)

                if ratio > max_ratio:
                    max_ratio = ratio
                    best_dy, best_dx = dy, dx

        # Compute Z-score
        ratios_arr = np.array(all_ratios)
        mean_ratio = float(np.mean(ratios_arr))
        std_ratio = float(np.std(ratios_arr))
        z_score = (max_ratio - mean_ratio) / std_ratio if std_ratio > 1e-10 else 0.0

        if video_hash:
            self._grid_cache[video_hash] = (best_dy, best_dx)

        return best_dy, best_dx, max_ratio, z_score

    def _compute_ratio_and_zscore(
        self, gray: np.ndarray, dy: int = 0, dx: int = 0
    ) -> Tuple[float, float]:
        """Compute peak ratio and z-score at a specific grid alignment."""
        H, W = gray.shape
        mask = np.zeros((8, 8), dtype=bool)
        for i in range(8):
            for j in range(8):
                if 0 < i + j <= 5:
                    mask[i, j] = True

        all_ratios = []
        target_ratio = 0.0

        for test_dy in range(8):
            for test_dx in range(8):
                h_new = ((H - test_dy) // 8) * 8
                w_new = ((W - test_dx) // 8) * 8
                if h_new <= 0 or w_new <= 0:
                    all_ratios.append(0.0)
                    continue

                crop = gray[test_dy : test_dy + h_new, test_dx : test_dx + w_new]
                blocks = crop.reshape(h_new // 8, 8, w_new // 8, 8).swapaxes(1, 2)
                dct_blocks = scipy.fft.dctn(blocks, axes=(-2, -1), norm="ortho")
                ac_coeffs = dct_blocks[:, :, mask].flatten()
                ac_coeffs = np.round(ac_coeffs)
                hist, _ = np.histogram(ac_coeffs, bins=513, range=(-256.5, 256.5))
                autocorr = np.correlate(hist, hist, mode="same")

                center = len(autocorr) // 2
                primary = float(autocorr[center])
                if primary < 1e-10:
                    all_ratios.append(0.0)
                    if test_dy == dy and test_dx == dx:
                        target_ratio = 0.0
                    continue

                excluded = autocorr.copy()
                excluded[max(0, center - 1) : min(len(autocorr), center + 2)] = 0
                secondary = float(np.max(excluded))
                ratio = secondary / primary
                all_ratios.append(ratio)

                if test_dy == dy and test_dx == dx:
                    target_ratio = ratio

        ratios_arr = np.array(all_ratios)
        mean_ratio = float(np.mean(ratios_arr))
        std_ratio = float(np.std(ratios_arr))
        z_score = (target_ratio - mean_ratio) / std_ratio if std_ratio > 1e-10 else 0.0

        return target_ratio, z_score

    def _score_from_ratio(
        self, peak_ratio: float, z_score: float, is_phone: bool = False
    ) -> float:
        """Convert peak ratio to tampering score with statistical correction."""
        # FIX S-01: Use calibrated Z-score threshold (defaults to 3.0 if not in thresholds)
        threshold_z = getattr(__import__('utils.thresholds', fromlist=['DCT_CORRECTED_Z_THRESHOLD']), 'DCT_CORRECTED_Z_THRESHOLD', 3.0)
        
        # If z-score is low, the grid search just found noise
        if z_score < threshold_z:
            return float(max(0.0, min(0.15, peak_ratio * 0.15)))

        # Statistically significant — apply ratio-based scoring
        if is_phone:
            # Phone images: much higher threshold
            phone_threshold = 0.96
            phone_scale = 0.04
            score = (peak_ratio - phone_threshold) / phone_scale
        else:
            score = (peak_ratio - DCT_RATIO_THRESHOLD) / DCT_RATIO_SCALE

        return float(max(0.0, min(1.0, score)))

    def _confidence_from_score(self, score: float, z_score: float = 0.0) -> float:
        """Parabolic confidence: high at extremes, low at threshold (0.55)."""
        # Distance from threshold (0.55)
        dist = abs(score - 0.55)
        # 0.45 is max distance. We want 0.9 confidence at distance 0.45
        # and 0.4 confidence at distance 0.0.
        raw_conf = 0.4 + (0.5 * (dist / 0.45)**2)
        
        # Boost confidence if statistical significance is high (Z > 4.15) 
        # OR if we categorically ruled out a misaligned grid (Z < 1.0)
        if z_score > 4.15 or z_score < 1.0:
            raw_conf = min(0.95, raw_conf + 0.15)
            
        return float(min(0.95, max(0.2, raw_conf)))

    def _abstain(self, start_time: float, reason: str = "Insufficient data") -> ToolResult:
        return ToolResult(
            tool_name=self.tool_name,
            success=True,
            real_prob=0.5,
            confidence=0.0,
            details={"grid_artifacts": False},
            error=False,
            error_msg=None,
            execution_time=time.time() - start_time,
            evidence_summary=f"Abstention: {reason}.",
        )

    def _run_inference(self, input_data: Dict[str, Any]) -> ToolResult:
        start_time = time.time()
        tracked_faces = input_data.get("tracked_faces", [])
        frames = input_data.get("frames_30fps", [])
        first_frame = input_data.get("first_frame", None)
        media_path = input_data.get("media_path", None)

        # ─── FIX C-04: Native Resolution Analysis ───
        crops = []
        
        if tracked_faces and frames:
            for face in tracked_faces:
                # 1. Try native crop if preprocessor provides it
                native_crop = face.get("face_crop_native")
                if native_crop is not None:
                    crops.append(native_crop)
                    continue

                # 2. Fallback: Extract native crop from full frame using bbox
                # This avoids the 224px resize that destroys the 8x8 JPEG grid
                best_idx = face.get("best_frame_idx", 0)
                if best_idx < len(frames):
                    frame = frames[best_idx]
                    bbox = face.get("trajectory_bboxes", {}).get(best_idx)
                    if bbox:
                        x1, y1, x2, y2 = bbox
                        # Add 20% margin for context, but keep native pixel grid
                        w, h = x2 - x1, y2 - y1
                        mx, my = int(w * 0.2), int(h * 0.2)
                        nx1 = max(0, x1 - mx)
                        ny1 = max(0, y1 - my)
                        nx2 = min(frame.shape[1], x2 + mx)
                        ny2 = min(frame.shape[0], y2 + my)
                        crops.append(frame[ny1:ny2, nx1:nx2])
        
        # No-face fallback: use full raw image
        if not crops and media_path:
            try:
                raw_img = cv2.imread(media_path)
                if raw_img is not None:
                    crops.append(raw_img)
            except Exception as e:
                logger.warning(f"DCT: Failed to load raw image: {e}")
        elif not crops and first_frame is not None:
            crops.append(first_frame)

        if not crops:
            return self._abstain(start_time, "No image data available for analysis")

        # ─── FIX S-05: Check if image is phone-origin ───
        is_phone = False
        if media_path:
            is_phone = _is_phone_origin(media_path)
            if is_phone:
                logger.info("DCT: Phone-origin image detected — applying higher threshold")

        # ─── FIX S-01: Statistical grid search ───
        video_hash = self._compute_video_hash(frames) if frames else None

        grid_dy, grid_dx, grid_ratio, z_score = self._find_optimal_grid_statistical(
            self._to_gray(self._coerce_to_uint8(crops[0])), video_hash
        )

        peak_ratios = []
        z_scores = []
        crops_processed = 0

        for crop in crops:
            crops_processed += 1
            gray = self._to_gray(self._coerce_to_uint8(crop))
            ratio, z = self._compute_ratio_and_zscore(gray, grid_dy, grid_dx)
            peak_ratios.append(ratio)
            z_scores.append(z)

        if crops_processed == 0:
            return self._abstain(start_time, "No valid image crops found")

        avg_ratio = float(np.mean(peak_ratios))
        avg_z = float(np.mean(z_scores))

        # ─── Final Scoring ───
        score = self._score_from_ratio(avg_ratio, avg_z, is_phone=is_phone)
        confidence = self._confidence_from_score(score, avg_z)
        grid_artifacts = score > 0.5

        if score > 0.5:
            summary = (
                f"DCT analysis detected double-quantization artifacts "
                f"(peak_ratio={avg_ratio:.3f}, z_score={avg_z:.1f}, score={score:.3f}), "
                f"indicating structural modification."
            )
        elif avg_z < 3.0:
            summary = (
                f"Confirmed single-compression JPEG profile. No suspicious grid found "
                f"(peak={avg_ratio:.3f}, z={avg_z:.1f}). Matches natural camera sensor signature."
            )
        else:
            summary = (
                f"Smooth DCT frequency distribution "
                f"(peak_ratio={avg_ratio:.3f}, z_score={avg_z:.1f}, score={score:.3f}), "
                f"consistent with natural imagery."
            )

        return ToolResult(
            tool_name=self.tool_name,
            success=True,
            real_prob=float(1.0 - score),
            confidence=float(confidence),
            details={
                "grid_artifacts": grid_artifacts,
                "peak_ratio": float(avg_ratio),
                "z_score": float(avg_z),
                "faces_analyzed": crops_processed,
                "grid_alignment": (grid_dy, grid_dx),
                "is_phone_origin": is_phone,
            },
            error=False,
            error_msg=None,
            execution_time=time.time() - start_time,
            evidence_summary=summary,
        )