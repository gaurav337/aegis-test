"""Illumination Tool — Shape-from-Shading Physics Check (V5).

Compares dominant illumination gradient of the face against scene background.
Diffusion models often composite photorealistic faces into scenes with
conflicting directional light sources.

V5 Fixes (on top of V4):
     1. _ensure_rgb uses robust multi-channel heuristic (not fragile mean comparison)
     2. Context gradient computed per-strip with vector averaging (no vstack discontinuity)
     3. Scoring driven by angular mismatch magnitude (not crude left/right ratio)
     4. Context luma bilateral filtered (matching face preprocessing)
     5. Skin mask fallback returns None instead of unmasked luma when insufficient skin
"""

import time
import cv2
import numpy as np
from typing import Any, Dict, List, Tuple, Optional

from core.base_tool import BaseForensicTool
from core.data_types import ToolResult
from utils.thresholds import (
    ILLUMINATION_DIFFUSE_THRESHOLD,
    ILLUMINATION_GRADIENT_CONSISTENT_WEIGHT,
    ILLUMINATION_GRADIENT_MISMATCH_WEIGHT,
    ILLUMINATION_MISMATCH_BASE_PENALTY,
)
from utils.logger import setup_logger

logger = setup_logger(__name__)

# ─── V5 Constants ───
MIN_CONTEXT_HEIGHT_RATIO = 0.20  # Context must be ≥20% of face height
ANGULAR_MISMATCH_THRESHOLD_DEG = 90.0  # Angular difference = genuine mismatch
MIN_GRADIENT_MAGNITUDE = 0.02  # Below this = direction unreliable, abstain
BILATERAL_FILTER_DIAMETER = 9  # Bilateral filter for albedo smoothing
BILATERAL_SIGMA_COLOR = 75
BILATERAL_SIGMA_SPACE = 75
MIN_SKIN_PIXELS = 100  # Minimum skin pixels required for reliable masking


class IlluminationTool(BaseForensicTool):
    """Tool for detecting illumination direction mismatches between face and scene."""

    @property
    def tool_name(self) -> str:
        return "run_illumination"

    def setup(self) -> None:
        """Initialize tool configuration from thresholds.py."""
        self.diffuse_threshold = ILLUMINATION_DIFFUSE_THRESHOLD
        self.consistent_weight = ILLUMINATION_GRADIENT_CONSISTENT_WEIGHT
        self.mismatch_weight = ILLUMINATION_GRADIENT_MISMATCH_WEIGHT
        self.mismatch_base_penalty = ILLUMINATION_MISMATCH_BASE_PENALTY

    # ─── FIX 1: Robust RGB/BGR detection ───

    @staticmethod
    def _ensure_rgb(img: np.ndarray) -> np.ndarray:
        """Ensure image is in RGB color space for consistent luma extraction.

        Uses a robust heuristic: in natural RGB images, the red channel
        (index 0) typically has higher mean than the blue channel (index 2).
        In a misinterpreted BGR image, channel 0 is actually blue and
        channel 2 is actually red, so ch0_mean < ch2_mean. We also require
        sufficient variance in both channels to avoid false triggers on
        uniformly-coloured images.
        """
        if img.ndim != 3 or img.shape[2] != 3:
            return img

        mean_ch0 = float(np.mean(img[:, :, 0]))
        mean_ch2 = float(np.mean(img[:, :, 2]))
        var_ch0 = float(np.var(img[:, :, 0]))
        var_ch2 = float(np.var(img[:, :, 2]))

        # BGR detected: first channel (supposedly red) is darker than last
        # channel (supposedly blue), with sufficient texture in both.
        if (mean_ch0 < mean_ch2 * 0.90) and (var_ch0 > 50) and (var_ch2 > 50):
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    # ─── Skin mask for albedo isolation ───

    @staticmethod
    def _create_skin_mask(rgb_img: np.ndarray) -> np.ndarray:
        """Create binary mask of skin-colored regions in YCrCb space.

        Excludes glasses, lips, eyebrows, facial hair, and background.
        """
        ycrcb = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2YCrCb)

        # CrCb skin color range (broad to handle diverse skin tones)
        skin_mask = cv2.inRange(
            ycrcb,
            np.array([0, 133, 77], dtype=np.uint8),
            np.array([255, 173, 127], dtype=np.uint8),
        )

        # Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)

        return skin_mask

    # ─── Luma extraction ───

    @staticmethod
    def _extract_luma(rgb_img: np.ndarray) -> np.ndarray:
        """Convert RGB to YCrCb and extract Y (luma) channel."""
        ycrcb = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2YCrCb)
        return ycrcb[:, :, 0].astype(np.float32)

    @staticmethod
    def _safe_divide(numerator: float, denominator: float) -> float:
        """Division with ZeroDivisionError protection."""
        return numerator / (denominator + 1e-6)

    # ─── FIX 2: Extract scene context with per-strip gradient averaging ───

    def _extract_scene_context_gradient(
        self,
        frame: np.ndarray,
        bbox: Tuple[int, int, int, int],
        face_h: int,
    ) -> Optional[Tuple[float, float, float]]:
        """Extract background gradient from regions beside the face.

        FIX 2: Computes gradient per-strip and averages vectors to avoid
        artificial discontinuity from stacking strips together.
        FIX 4: Applies bilateral filter to context luma (matching face).
        FIX 7: Requires minimum height (20% of face height).

        Returns (angle_degrees, magnitude, sharpness) or None if insufficient.
        """
        x1, y1, x2, y2 = bbox
        h, w = frame.shape[:2]

        min_ctx_height = int(face_h * MIN_CONTEXT_HEIGHT_RATIO)

        # Context region: same vertical span as face, beside the face
        ctx_y1 = max(0, y1)
        ctx_y2 = min(h, y2)
        ctx_height = ctx_y2 - ctx_y1

        if ctx_height < min_ctx_height:
            return None

        # Collect left and right context strips
        ctx_strips = []

        # Left context
        left_width = min(x1, face_h)
        if left_width >= min_ctx_height:
            left_ctx = frame[ctx_y1:ctx_y2, max(0, x1 - left_width) : x1]
            if left_ctx.size > 0 and left_ctx.shape[0] >= min_ctx_height:
                ctx_strips.append(left_ctx)

        # Right context
        right_width = min(w - x2, face_h)
        if right_width >= min_ctx_height:
            right_ctx = frame[ctx_y1:ctx_y2, x2 : min(w, x2 + right_width)]
            if right_ctx.size > 0 and right_ctx.shape[0] >= min_ctx_height:
                ctx_strips.append(right_ctx)

        if not ctx_strips:
            return None

        # Compute gradient per-strip and average the vectors
        angles = []
        magnitudes = []
        for strip in ctx_strips:
            rgb_strip = self._ensure_rgb(strip)
            strip_luma = self._extract_luma(rgb_strip)

            # FIX 4: Bilateral filter context luma to reduce texture artifacts
            smoothed = cv2.bilateralFilter(
                strip_luma.astype(np.uint8),
                BILATERAL_FILTER_DIAMETER,
                BILATERAL_SIGMA_COLOR,
                BILATERAL_SIGMA_SPACE,
            )

            a, m, _ = self._compute_gradient_direction(smoothed.astype(np.float32))
            angles.append(a)
            magnitudes.append(m)

        # Average gradient vectors (convert to Cartesian, mean, convert back)
        mean_gx = float(
            np.mean([m * np.cos(np.radians(a)) for a, m in zip(angles, magnitudes)])
        )
        mean_gy = float(
            np.mean([m * np.sin(np.radians(a)) for a, m in zip(angles, magnitudes)])
        )
        avg_angle = float(np.degrees(np.arctan2(mean_gy, mean_gx)))
        if avg_angle < 0:
            avg_angle += 360
        avg_magnitude = float(np.sqrt(mean_gx**2 + mean_gy**2))

        return avg_angle, avg_magnitude, float(np.mean(magnitudes))

    # ─── 2D Gradient Direction ───

    def _compute_gradient_direction(
        self, luma: np.ndarray
    ) -> Tuple[float, float, float]:
        """Compute 2D gradient vector using Sobel operators.

        Returns (angle_degrees, magnitude, sharpness).
        angle: 0=right, 90=up, 180=left, 270=down
        """
        gx = cv2.Sobel(luma, cv2.CV_64F, 1, 0, ksize=5)
        gy = cv2.Sobel(luma, cv2.CV_64F, 0, 1, ksize=5)

        magnitudes = np.sqrt(gx**2 + gy**2)
        weights = magnitudes / (magnitudes.sum() + 1e-10)

        mean_gx = float((gx * weights).sum())
        mean_gy = float((gy * weights).sum())

        angle = float(np.degrees(np.arctan2(mean_gy, mean_gx)))
        magnitude = float(np.sqrt(mean_gx**2 + mean_gy**2))

        if angle < 0:
            angle += 360

        return angle, magnitude, float(magnitudes.mean())

    # ─── Nose shadow direction check ───

    def _check_nose_shadow_direction(
        self, luma: np.ndarray
    ) -> Tuple[Optional[str], str]:
        """Check which side of the nose is darker (shadow direction).

        The nose shadow is the most reliable single indicator of light direction.
        Returns (shadow_side, detail_string) or (None, detail_string) if inconclusive.
        """
        if luma.ndim != 2 or luma.shape[0] < 20 or luma.shape[1] < 20:
            return None, "Insufficient resolution for nose shadow check"

        h, w = luma.shape
        # Nose region: center vertical strip, upper-middle portion
        nose_x1 = int(w * 0.35)
        nose_x2 = int(w * 0.65)
        nose_y1 = int(h * 0.25)
        nose_y2 = int(h * 0.65)

        if nose_x2 <= nose_x1 or nose_y2 <= nose_y1:
            return None, "Nose region out of bounds"

        nose_region = luma[nose_y1:nose_y2, nose_x1:nose_x2]
        nose_mid = nose_region.shape[1] // 2

        left_mean = float(np.mean(nose_region[:, :nose_mid]))
        right_mean = float(np.mean(nose_region[:, nose_mid:]))

        diff = abs(left_mean - right_mean)
        if diff < 2.0:
            return None, f"Nose shadow too subtle (diff={diff:.1f})"

        shadow_side = "left" if left_mean < right_mean else "right"
        return shadow_side, f"Nose shadow on {shadow_side} side (diff={diff:.1f})"

    # ─── FIX 5: Preprocess face luma — skin mask failure returns None ───

    def _preprocess_face_luma(self, face_rgb: np.ndarray) -> Optional[np.ndarray]:
        """Isolate illumination signal from albedo signal.

        FIX 5: When <100 skin pixels detected, returns None to signal
        unreliable luma rather than falling back to unmasked luma
        (which defeats the purpose of the mask).
        """
        # Bilateral filter: smooths texture but preserves edges
        smoothed = cv2.bilateralFilter(
            face_rgb.astype(np.uint8),
            BILATERAL_FILTER_DIAMETER,
            BILATERAL_SIGMA_COLOR,
            BILATERAL_SIGMA_SPACE,
        )

        # Extract luma from smoothed image
        luma = self._extract_luma(smoothed)

        # Create skin mask and apply
        skin_mask = self._create_skin_mask(smoothed)

        if np.sum(skin_mask) < MIN_SKIN_PIXELS:
            return None  # FIX 5: Signal unreliable luma instead of fallback

        # Apply mask: set non-skin regions to median skin luma
        median_skin_luma = float(np.median(luma[skin_mask > 0]))
        luma_masked = luma.copy()
        luma_masked[skin_mask == 0] = median_skin_luma

        return luma_masked

    # ─── Dynamic Confidence ───

    def _calculate_confidence(
        self,
        face_grad: float,
        crop_sharpness: float,
        face_width: int,
    ) -> float:
        """Confidence adjusts based on gradient strength, sharpness, and face size."""
        grad_conf = min(0.9, face_grad * 10)

        if crop_sharpness < 50.0:
            grad_conf *= 0.6
        elif crop_sharpness < 100.0:
            grad_conf *= 0.8

        if face_width < 80:
            grad_conf *= 0.7
        elif face_width < 120:
            grad_conf *= 0.85

        return round(min(grad_conf, 0.95), 3)

    # ─── MAIN INFERENCE ───

    def _run_inference(self, input_data: Dict[str, Any]) -> ToolResult:
        """Run illumination analysis on all tracked faces."""
        start_time = time.time()

        tracked_faces = input_data.get("tracked_faces", [])
        frames = input_data.get("frames_30fps", [])

        if not tracked_faces:
            return ToolResult(
                tool_name=self.tool_name,
                success=True,
                score=0.0,
                confidence=0.0,
                details={
                    "illumination_score": 0.0,
                    "face_gradient": 0.0,
                    "lighting_consistent": None,
                    "faces_analyzed": 0,
                },
                error=False,
                error_msg=None,
                execution_time=time.time() - start_time,
                evidence_summary="Illumination analysis skipped: No tracked faces provided.",
            )

        if not frames:
            return ToolResult(
                tool_name=self.tool_name,
                success=True,
                score=0.0,
                confidence=0.0,
                details={
                    "illumination_score": 0.0,
                    "face_gradient": 0.0,
                    "lighting_consistent": None,
                    "faces_analyzed": 0,
                },
                error=False,
                error_msg=None,
                execution_time=time.time() - start_time,
                evidence_summary="Illumination analysis skipped: No frames provided for scene context.",
            )

        face_results = []

        for face in tracked_faces:
            face_crop = face.get("face_crop_224")
            trajectory_bboxes = face.get("trajectory_bboxes", {})
            landmarks = face.get("landmarks")

            if face_crop is None:
                continue

            # Ensure RGB
            face_crop = self._ensure_rgb(np.array(face_crop, dtype=np.uint8))

            if (
                face_crop.ndim != 3
                or face_crop.shape[0] != 224
                or face_crop.shape[1] != 224
            ):
                logger.warning(
                    f"Face identity {face.get('identity_id', 'unknown')}: "
                    f"Invalid crop shape {face_crop.shape}, expected (224, 224, 3)"
                )
                continue

            # Get best frame
            best_frame_idx = face.get("best_frame_idx", 0)
            if best_frame_idx >= len(frames):
                best_frame_idx = 0

            original_frame = self._ensure_rgb(frames[best_frame_idx])

            # Get face bbox
            bbox = None
            if trajectory_bboxes and best_frame_idx in trajectory_bboxes:
                bbox = trajectory_bboxes[best_frame_idx]
            elif landmarks is not None:
                x_min, y_min = np.min(landmarks, axis=0)
                x_max, y_max = np.max(landmarks, axis=0)
                bbox = (int(x_min), int(y_min), int(x_max), int(y_max))

            if bbox is None:
                logger.warning(
                    f"Face identity {face.get('identity_id', 'unknown')}: "
                    f"No bounding box available for scene context extraction"
                )
                continue

            # Preprocess face luma (bilateral + skin mask)
            face_luma = self._preprocess_face_luma(face_crop)

            if face_luma is None:
                face_results.append(
                    {
                        "identity_id": face.get("identity_id", 0),
                        "fake_score": 0.0,
                        "confidence": 0.0,
                        "face_gradient": 0.0,
                        "face_dom": "no_skin_mask",
                        "ctx_dom": "none",
                        "lighting_consistent": None,
                        "interpretation": "Insufficient skin detected for reliable gradient. Abstaining.",
                    }
                )
                continue

            # Compute 2D gradient direction for face
            face_angle, face_magnitude, _ = self._compute_gradient_direction(face_luma)

            # Left/right gradient for confidence calculation
            midpoint_x = 112
            face_l = face_luma[:, :midpoint_x].mean()
            face_r = face_luma[:, midpoint_x:].mean()
            face_grad = self._safe_divide(abs(face_l - face_r), face_l + face_r)

            # Diffuse lighting = abstain
            if face_grad < self.diffuse_threshold:
                face_results.append(
                    {
                        "identity_id": face.get("identity_id", 0),
                        "fake_score": 0.0,
                        "confidence": 0.0,
                        "face_gradient": face_grad,
                        "face_dom": "diffuse",
                        "ctx_dom": "none",
                        "lighting_consistent": None,
                        "interpretation": "Diffuse lighting detected — no directional signal available. Abstaining.",
                    }
                )
                continue

            # Extract scene context gradient (per-strip averaging, bilateral filtered)
            face_h = bbox[3] - bbox[1]
            ctx_result = self._extract_scene_context_gradient(
                original_frame, bbox, face_h
            )

            if ctx_result is None:
                face_results.append(
                    {
                        "identity_id": face.get("identity_id", 0),
                        "fake_score": 0.0,
                        "confidence": 0.0,
                        "face_gradient": face_grad,
                        "face_dom": "none",
                        "ctx_dom": "none",
                        "lighting_consistent": None,
                        "interpretation": "Scene context extraction failed — insufficient background. Abstaining.",
                    }
                )
                continue

            ctx_angle, ctx_magnitude, _ = ctx_result

            # Angular comparison
            angular_diff = abs(face_angle - ctx_angle)
            if angular_diff > 180:
                angular_diff = 360 - angular_diff

            # Check if both gradients are strong enough to be reliable
            both_strong = (
                face_magnitude > MIN_GRADIENT_MAGNITUDE
                and ctx_magnitude > MIN_GRADIENT_MAGNITUDE
            )

            if both_strong and angular_diff > ANGULAR_MISMATCH_THRESHOLD_DEG:
                lighting_consistent = False
                face_dom = f"{face_angle:.0f}°"
                ctx_dom = f"{ctx_angle:.0f}°"
            elif both_strong:
                lighting_consistent = True
                face_dom = f"{face_angle:.0f}°"
                ctx_dom = f"{ctx_angle:.0f}°"
            else:
                # One or both gradients too weak — abstain
                face_results.append(
                    {
                        "identity_id": face.get("identity_id", 0),
                        "fake_score": 0.0,
                        "confidence": 0.0,
                        "face_gradient": face_grad,
                        "face_dom": "weak_gradient",
                        "ctx_dom": "weak_gradient",
                        "lighting_consistent": None,
                        "interpretation": "Gradient too weak for reliable direction comparison. Abstaining.",
                    }
                )
                continue

            # Nose shadow direction check
            nose_shadow_side, nose_detail = self._check_nose_shadow_direction(face_luma)

            # FIX 3: Scoring driven by angular mismatch, not crude left/right ratio
            if lighting_consistent:
                fake_score = face_grad * self.consistent_weight
            else:
                # Scale penalty by angular difference (0.5 to 1.0 range)
                angular_penalty = angular_diff / 180.0  # 0.0 to 1.0
                fake_score = self.mismatch_base_penalty + (
                    angular_penalty * self.mismatch_weight
                )

            # Nose shadow consistency: if nose shadow contradicts face gradient
            if nose_shadow_side is not None:
                gradient_side = "left" if face_l > face_r else "right"
                if nose_shadow_side != gradient_side:
                    fake_score = min(1.0, fake_score + 0.1)

            fake_score = min(fake_score, 1.0)

            # Confidence
            face_width = bbox[2] - bbox[0]
            crop_sharpness = cv2.Laplacian(
                cv2.cvtColor(face_crop, cv2.COLOR_RGB2GRAY), cv2.CV_64F
            ).var()
            confidence = self._calculate_confidence(
                face_grad, crop_sharpness, face_width
            )

            # Interpretation
            if lighting_consistent:
                interpretation = (
                    f"Consistent lighting (face angle: {face_dom}, context: {ctx_dom}). "
                    f"Gradient: {face_grad:.3f}. {nose_detail}"
                )
            else:
                interpretation = (
                    f"Illumination direction mismatch (face angle: {face_dom}, "
                    f"context: {ctx_dom}, angular diff: {angular_diff:.0f}°). "
                    f"Gradient: {face_grad:.3f}. {nose_detail}"
                )

            face_results.append(
                {
                    "identity_id": face.get("identity_id", 0),
                    "fake_score": fake_score,
                    "confidence": confidence,
                    "face_gradient": face_grad,
                    "face_angle": face_angle,
                    "face_magnitude": face_magnitude,
                    "ctx_angle": ctx_angle,
                    "ctx_magnitude": ctx_magnitude,
                    "angular_diff": angular_diff,
                    "face_dom": face_dom,
                    "ctx_dom": ctx_dom,
                    "lighting_consistent": lighting_consistent,
                    "nose_shadow_side": nose_shadow_side,
                    "interpretation": interpretation,
                }
            )

        if not face_results:
            return ToolResult(
                tool_name=self.tool_name,
                success=True,
                score=0.0,
                confidence=0.0,
                details={
                    "illumination_score": 0.0,
                    "face_gradient": 0.0,
                    "lighting_consistent": None,
                    "faces_analyzed": 0,
                },
                error=False,
                error_msg=None,
                execution_time=time.time() - start_time,
                evidence_summary="Illumination analysis skipped: No valid face crops found.",
            )

        # Use median score instead of worst
        sorted_scores = sorted([f["fake_score"] for f in face_results])
        median_score = sorted_scores[len(sorted_scores) // 2]
        median_face = min(
            face_results, key=lambda x: abs(x["fake_score"] - median_score)
        )

        if median_face["lighting_consistent"] is None:
            summary = median_face["interpretation"]
        elif median_face["lighting_consistent"]:
            summary = f"Consistent lighting for face {median_face['identity_id']} (out of {len(face_results)} analyzed)."
        else:
            summary = (
                f"Illumination direction mismatch for face {median_face['identity_id']} "
                f"(out of {len(face_results)} analyzed)."
            )

        details = {
            "illumination_score": median_face["fake_score"],
            "face_gradient": float(median_face["face_gradient"]),
            "lighting_consistent": median_face["lighting_consistent"],
            "faces_analyzed": len(face_results),
            "face_dom": median_face.get("face_dom", "none"),
            "ctx_dom": median_face.get("ctx_dom", "none"),
            "worst_face_id": median_face["identity_id"],
            "face_angle": median_face.get("face_angle", 0),
            "ctx_angle": median_face.get("ctx_angle", 0),
            "angular_diff": median_face.get("angular_diff", 0),
            "nose_shadow_side": median_face.get("nose_shadow_side"),
        }

        return ToolResult(
            tool_name=self.tool_name,
            success=True,
            score=float(median_face["fake_score"]),
            confidence=float(median_face["confidence"]),
            details=details,
            error=False,
            error_msg=None,
            execution_time=time.time() - start_time,
            evidence_summary=summary,
        )
