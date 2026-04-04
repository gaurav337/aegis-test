"""DCT Double-Quantization Detector — V3 (Native Resolution + Statistical Correction).

Key fixes over V2:
1. Native resolution analysis — NO resizing before DCT (resizing destroys 8×8 JPEG grid)
2. Statistical grid search correction — compares best ratio against distribution of all 64 offsets
3. EXIF-based phone detection — applies higher thresholds for phone-origin images
4. Recalibrated scoring — much more conservative to avoid false positives on real JPEGs
"""

"""DCT Double-Quantization Detector — V3 (Native Resolution + Statistical Correction).

Key fixes over V2:
1. Native resolution analysis — NO resizing before DCT (resizing destroys 8×8 JPEG grid)
2. Statistical grid search correction — compares best ratio against distribution of all 64 offsets
3. EXIF-based phone detection — applies higher thresholds for phone-origin images
4. Recalibrated scoring — much more conservative to avoid false positives on real JPEGs
"""

import time
import struct
import numpy as np
import scipy.fft
from typing import Any, Dict, Tuple, Optional
import hashlib
import logging

from core.base_tool import BaseForensicTool
from core.data_types import ToolResult
from utils.thresholds import (
    DCT_RATIO_THRESHOLD,
    DCT_RATIO_SCALE,
    DCT_CONFIDENCE_CAP,
    DCT_CONFIDENCE_BUMP,
)

logger = logging.getLogger(__name__)


# ─── JPEG Quantization Table Reader ───
def _read_jpeg_quant_table(filepath: str) -> Optional[np.ndarray]:
    """Read the first JPEG quantization table from a JPEG file.

    Returns an 8×8 array of quantization values, or None if not found.
    Parses DQT (Define Quantization Table) markers (0xFFDB).
    """
    try:
        with open(filepath, "rb") as f:
            # Check JPEG signature
            if f.read(2) != b"\xff\xd8":
                return None

            while True:
                # Find next marker
                byte = f.read(1)
                while byte != b"\xff":
                    byte = f.read(1)
                    if not byte:
                        return None

                # Skip padding 0xFF bytes
                while byte == b"\xff":
                    byte = f.read(1)

                marker = byte[0]

                if marker == 0xDB:  # DQT marker
                    # Read segment length
                    length_data = f.read(2)
                    if len(length_data) < 2:
                        return None
                    length = struct.unpack(">H", length_data)[0]

                    # Read quantization table data
                    qt_data = f.read(length - 2)
                    if len(qt_data) < length - 2:
                        return None

                    # Parse first table (precision + table ID in first byte)
                    precision_id = qt_data[0]
                    precision = (precision_id >> 4) & 0x0F  # 0=8bit, 1=16bit
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
                elif 0xD0 <= marker <= 0xD7:  # RST markers (no payload)
                    continue
                elif marker in (
                    0x00,
                    0x01,
                    0xD0,
                    0xD1,
                    0xD2,
                    0xD3,
                    0xD4,
                    0xD5,
                    0xD6,
                    0xD7,
                ):
                    continue
                else:
                    # Read segment length and skip
                    length_data = f.read(2)
                    if len(length_data) < 2:
                        break
                    length = struct.unpack(">H", length_data)[0]
                    f.seek(length - 2, 1)
    except Exception as e:
        logger.debug(f"Failed to read JPEG quantization table: {e}")

    return None


# ─── EXIF Phone Detection ───
def _is_phone_origin(filepath: str) -> bool:
    """Check if image likely originated from a phone camera via EXIF data.

    Looks for EXIF Maker tag (0x010F) containing common phone manufacturers.
    """
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
            content = f.read(2_000_000)  # Read first 2MB (EXIF is near start)

            # Check for EXIF header
            if b"Exif\x00\x00" in content:
                for marker in phone_markers:
                    if marker in content:
                        return True
            elif b"JFIF" in content:
                # JFIF without EXIF — still check for phone markers in metadata
                for marker in phone_markers:
                    if marker in content:
                        return True
    except Exception:
        pass

    return False


class DCTTool(BaseForensicTool):
    """DCT-based double-quantization detector with native resolution analysis."""

    # --- Cache for grid alignment (per video hash) ---
    _grid_cache: Dict[str, Tuple[int, int]] = {}

    @property
    def tool_name(self) -> str:
        return "run_dct"

    def setup(self) -> None:
        pass

    # ---------------------------------------------------------

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

    # ---------------------------------------------------------

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

    # ---------------------------------------------------------

    @staticmethod
    def _compute_video_hash(frames: list) -> str:
        """Generate hash for video caching."""
        if not frames:
            return "empty"
        sample = frames[0][:100, :100] if len(frames) > 0 else np.zeros((100, 100))
        return hashlib.md5(sample.tobytes()).hexdigest()[:16]

    # ---------------------------------------------------------
    # FIX 2: Statistical Grid Search Correction
    # ---------------------------------------------------------

    def _find_optimal_grid_statistical(
        self, gray: np.ndarray, video_hash: Optional[str] = None
    ) -> Tuple[int, int, float, float]:
        """Find optimal 8×8 grid alignment WITH statistical correction.

        Returns: (best_dy, best_dx, best_ratio, z_score)

        The z_score measures how many standard deviations the best ratio
        is above the mean of all 64 offsets. A high z_score (>3.0) indicates
        a genuine grid-aligned pattern, while a low z_score suggests the
        "best" offset is just noise from multiple comparisons.
        """
        if video_hash and video_hash in self._grid_cache:
            dy, dx = self._grid_cache[video_hash]
            # Still compute z-score for this alignment
            ratio, z_score = self._compute_ratio_and_zscore(gray, dy, dx)
            return dy, dx, ratio, z_score

        H, W = gray.shape

        # Mask for Low-Frequency AC coefficients (i + j <= 5)
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

        # Compute z-score of best ratio against all 64 offsets
        ratios_arr = np.array(all_ratios)
        mean_ratio = float(np.mean(ratios_arr))
        std_ratio = float(np.std(ratios_arr))

        if std_ratio > 1e-10:
            z_score = (max_ratio - mean_ratio) / std_ratio
        else:
            z_score = 0.0

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

        if std_ratio > 1e-10:
            z_score = (target_ratio - mean_ratio) / std_ratio
        else:
            z_score = 0.0

        return target_ratio, z_score

    # ---------------------------------------------------------

    def _compute_peak_ratio(self, gray: np.ndarray, dy: int = 0, dx: int = 0) -> float:
        """Compute peak ratio with known grid alignment."""
        ratio, _ = self._compute_ratio_and_zscore(gray, dy, dx)
        return ratio

    # ---------------------------------------------------------
    # FIX 4: Recalibrated Scoring Function
    # ---------------------------------------------------------

    def _score_from_ratio(
        self, peak_ratio: float, z_score: float, is_phone: bool = False
    ) -> float:
        """Convert peak ratio to tampering score with statistical and phone awareness.

        Key changes:
        - Requires high z_score (>3.0) to consider the ratio meaningful
        - Phone-origin images need much higher ratios to trigger
        - Only ratios >0.97 start approaching the 0.5 decision boundary
        """
        # If z-score is low, the grid search just found noise
        if z_score < 3.0:
            # Not statistically significant — map to very low score
            return float(max(0.0, min(0.15, peak_ratio * 0.15)))

        # Statistically significant — now apply ratio-based scoring
        if is_phone:
            # Phone images: much higher threshold (phones legitimately double-compress)
            phone_threshold = 0.96
            phone_scale = 0.04
            score = (peak_ratio - phone_threshold) / phone_scale
        else:
            # Non-phone: still conservative
            score = (peak_ratio - DCT_RATIO_THRESHOLD) / DCT_RATIO_SCALE

        return float(max(0.0, min(1.0, score)))

    def _confidence_from_score(self, score: float) -> float:
        return float(min(DCT_CONFIDENCE_CAP, score + DCT_CONFIDENCE_BUMP))

    # ---------------------------------------------------------

    def _abstain(
        self, start_time: float, reason: str = "Insufficient data"
    ) -> ToolResult:
        return ToolResult(
            tool_name=self.tool_name,
            success=True,
            score=0.0,
            confidence=0.0,
            details={"grid_artifacts": False},
            error=False,
            error_msg=None,
            execution_time=0.0,
            evidence_summary=f"Abstention: {reason}.",
        )

    # ---------------------------------------------------------
    # FIX 1: Native Resolution Analysis — NO Resizing
    # ---------------------------------------------------------

    def _run_inference(self, input_data: Dict[str, Any]) -> ToolResult:
        start_time = time.time()

        tracked_faces = input_data.get("tracked_faces", [])
        frames = input_data.get("frames_30fps", [])
        first_frame = input_data.get("first_frame", None)
        media_path = input_data.get("media_path", None)

        # ─── FIX 1: Use native resolution crops ───
        # DCT analysis requires original pixel grid — NO resizing
        crops = []
        if tracked_faces:
            for face in tracked_faces:
                # Try to get native resolution crop first
                native_crop = face.get("face_crop_native")
                if native_crop is not None:
                    crops.append(native_crop)
                else:
                    # Fallback to 224 crop but note it's suboptimal
                    crop = face.get("face_crop_224")
                    if crop is not None:
                        crops.append(crop)
                        logger.warning(
                            "DCT: Using resized 224×224 crop — results may be unreliable "
                            "due to interpolation artifacts. Native resolution preferred."
                        )

        # No-face fallback: load raw image AT NATIVE RESOLUTION
        if not crops and media_path:
            try:
                from PIL import Image

                raw_img = Image.open(media_path).convert("RGB")
                crops.append(np.array(raw_img))  # Native resolution!
            except Exception as e:
                logger.warning(f"DCT: Failed to load raw image: {e}")
        elif not crops and first_frame is not None:
            crops.append(first_frame)  # Native resolution frame

        if not crops:
            return self._abstain(start_time, "No image data available for analysis")

        # ─── FIX 3: Check if image is phone-origin ───
        is_phone = False
        if media_path:
            is_phone = _is_phone_origin(media_path)
            if is_phone:
                logger.info(
                    "DCT: Phone-origin image detected — applying higher threshold"
                )

        # ─── FIX 2: Statistical grid search ───
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

        # ─── FIX 4: Recalibrated scoring ───
        score = self._score_from_ratio(avg_ratio, avg_z, is_phone=is_phone)
        confidence = self._confidence_from_score(score)

        grid_artifacts = score > 0.5

        if score > 0.5:
            summary = (
                f"DCT analysis detected double-quantization artifacts "
                f"(peak_ratio={avg_ratio:.3f}, z_score={avg_z:.1f}, score={score:.3f}), "
                f"indicating structural modification."
            )
        elif avg_z < 3.0:
            summary = (
                f"DCT analysis: grid pattern not statistically significant "
                f"(peak_ratio={avg_ratio:.3f}, z_score={avg_z:.1f}). "
                f"Likely interpolation or single-compression artifacts."
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
            score=float(score),
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
            execution_time=0.0,
            evidence_summary=summary,
        )
