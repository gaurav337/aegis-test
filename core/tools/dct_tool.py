import time
import numpy as np
import scipy.fft
from typing import Any, Dict, Tuple, Optional
import hashlib

from core.base_tool import BaseForensicTool
from core.data_types import ToolResult
from utils.thresholds import (
    DCT_RATIO_THRESHOLD, DCT_RATIO_SCALE,
    DCT_CONFIDENCE_CAP, DCT_CONFIDENCE_BUMP
)


class DCTTool(BaseForensicTool):

    # --- Configuration ---
    
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
            # ✅ FIX 2: RGB order per Spec Section 1.1 (not BGR)
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

    def _find_optimal_grid(self, gray: np.ndarray, video_hash: Optional[str] = None) -> Tuple[int, int]:
        """Find optimal 8x8 grid alignment with caching."""
        if video_hash and video_hash in self._grid_cache:
            return self._grid_cache[video_hash]

        H, W = gray.shape
        max_ratio = 0.0
        best_dy, best_dx = 0, 0

        # Mask for Low-Frequency AC coefficients (i + j <= 5)
        mask = np.zeros((8, 8), dtype=bool)
        for i in range(8):
            for j in range(8):
                if 0 < i + j <= 5:
                    mask[i, j] = True

        for dy in range(8):
            for dx in range(8):
                h_new = ((H - dy) // 8) * 8
                w_new = ((W - dx) // 8) * 8

                if h_new <= 0 or w_new <= 0:
                    continue

                crop = gray[dy:dy + h_new, dx:dx + w_new]
                blocks = crop.reshape(h_new // 8, 8, w_new // 8, 8).swapaxes(1, 2)

                dct_blocks = scipy.fft.dctn(blocks, axes=(-2, -1), norm="ortho")
                ac_coeffs = dct_blocks[:, :, mask].flatten()
                ac_coeffs = np.round(ac_coeffs)

                hist, _ = np.histogram(ac_coeffs, bins=513, range=(-256.5, 256.5))
                autocorr = np.correlate(hist, hist, mode="same")

                center = len(autocorr) // 2
                primary = float(autocorr[center])

                if primary < 1e-10:
                    continue

                excluded = autocorr.copy()
                excluded[max(0, center - 1):min(len(autocorr), center + 2)] = 0
                secondary = float(np.max(excluded))
                ratio = secondary / primary

                if ratio > max_ratio:
                    max_ratio = ratio
                    best_dy, best_dx = dy, dx

        if video_hash:
            self._grid_cache[video_hash] = (best_dy, best_dx)

        return best_dy, best_dx

    # ---------------------------------------------------------

    def _compute_peak_ratio(self, gray: np.ndarray, dy: int = 0, dx: int = 0) -> float:
        """Compute peak ratio with known grid alignment."""
        H, W = gray.shape

        h_new = ((H - dy) // 8) * 8
        w_new = ((W - dx) // 8) * 8

        if h_new <= 0 or w_new <= 0:
            return 0.0

        crop = gray[dy:dy + h_new, dx:dx + w_new]
        blocks = crop.reshape(h_new // 8, 8, w_new // 8, 8).swapaxes(1, 2)

        dct_blocks = scipy.fft.dctn(blocks, axes=(-2, -1), norm="ortho")

        mask = np.zeros((8, 8), dtype=bool)
        for i in range(8):
            for j in range(8):
                if 0 < i + j <= 5:
                    mask[i, j] = True

        ac_coeffs = dct_blocks[:, :, mask].flatten()
        ac_coeffs = np.round(ac_coeffs)

        hist, _ = np.histogram(ac_coeffs, bins=513, range=(-256.5, 256.5))
        autocorr = np.correlate(hist, hist, mode="same")

        center = len(autocorr) // 2
        primary = float(autocorr[center])

        if primary < 1e-10:
            return 0.0

        excluded = autocorr.copy()
        excluded[max(0, center - 1):min(len(autocorr), center + 2)] = 0
        secondary = float(np.max(excluded))

        return float(secondary / primary)

    # ---------------------------------------------------------

    def _score_from_ratio(self, peak_ratio: float) -> float:
        """
        Higher peak_ratio = Higher tampering score.
        Grid alignment search finds JPEG quantization patterns better,
        creating stronger autocorrelation peaks in compressed/tampered images.
        """
        return float(max(0.0, min(1.0, (peak_ratio - DCT_RATIO_THRESHOLD) / DCT_RATIO_SCALE)))

    def _confidence_from_score(self, score: float) -> float:
        return float(min(DCT_CONFIDENCE_CAP, score + DCT_CONFIDENCE_BUMP))

    # ---------------------------------------------------------

    def _abstain(self, start_time: float, reason: str = "Insufficient data") -> ToolResult:
        # ✅ FIX 3: Abstention is successful execution with no evidence (Spec Section 4)
        return ToolResult(
            tool_name=self.tool_name,
            success=True,
            score=0.0,
            confidence=0.0,
            details={"grid_artifacts": False},  # ✅ FIX 4: Required field per Spec Section 2.3
            error=False,
            error_msg=None,
            execution_time=0.0,  # Base class will overwrite
            evidence_summary=f"Abstention: {reason}."
        )

    # ---------------------------------------------------------

    def _run_inference(self, input_data: Dict[str, Any]) -> ToolResult:
        start_time = time.time()

        # ✅ FIX 5: Rely on Preprocessing Contract (Spec Section 1.5)
        tracked_faces = input_data.get("tracked_faces", [])
        frames = input_data.get("frames_30fps", [])
        first_frame = input_data.get("first_frame", None)
        media_path = input_data.get("media_path", None)

        # Build crops list — face crops or raw image fallback
        crops = []
        if tracked_faces:
            for face in tracked_faces:
                crop = face.get("face_crop_224")
                if crop is not None:
                    crops.append(crop)
        
        # No-face fallback: load raw image
        if not crops and first_frame is not None:
            import cv2
            crop = cv2.resize(first_frame, (224, 224), interpolation=cv2.INTER_LANCZOS4)
            crops.append(crop)
        elif not crops and media_path:
            try:
                from PIL import Image
                raw_img = Image.open(media_path).convert("RGB").resize((224, 224), Image.LANCZOS)
                crops.append(np.array(raw_img))
            except Exception:
                pass
        
        if not crops:
            return self._abstain(start_time, "No image data available for analysis")

        # Compute hash for grid caching (use first frame of video if available)
        video_hash = self._compute_video_hash(frames) if frames else None

        # Determine grid alignment on the first available crop
        grid_dy, grid_dx = 0, 0
        if video_hash and crops:
            sample_crop = self._coerce_to_uint8(crops[0])
            sample_gray = self._to_gray(sample_crop)
            grid_dy, grid_dx = self._find_optimal_grid(sample_gray, video_hash)

        peak_ratios = []
        crops_processed = 0

        for crop in crops:
            crops_processed += 1
            gray = self._to_gray(self._coerce_to_uint8(crop))
            peak_ratio = self._compute_peak_ratio(gray, grid_dy, grid_dx)
            peak_ratios.append(peak_ratio)

        if crops_processed == 0:
            return self._abstain(start_time, "No valid image crops found")

        avg_ratio = float(np.mean(peak_ratios))
        score = self._score_from_ratio(avg_ratio)
        confidence = self._confidence_from_score(score)

        # ✅ FIX 4: Add grid_artifacts to details per Spec Section 2.3
        grid_artifacts = score > 0.5

        if score > 0.5:
            summary = (
                f"DCT analysis detected double-quantization artifacts "
                f"(peak_ratio={avg_ratio:.3f}, score={score:.3f}), indicating structural modification."
            )
        else:
            summary = (
                f"Smooth DCT frequency distribution "
                f"(peak_ratio={avg_ratio:.3f}, score={score:.3f}), consistent with natural imagery."
            )

        return ToolResult(
            tool_name=self.tool_name,
            success=True,
            score=float(score),
            confidence=float(confidence),
            details={
                "grid_artifacts": grid_artifacts,  # ✅ FIX 4: Required for ensemble routing
                "peak_ratio": float(avg_ratio),
                "faces_analyzed": crops_processed,
                "grid_alignment": (grid_dy, grid_dx)
            },
            error=False,
            error_msg=None,
            execution_time=0.0,  # ✅ FIX: Base class execute() will overwrite this
            evidence_summary=summary
        )