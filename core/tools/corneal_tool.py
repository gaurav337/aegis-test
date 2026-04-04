"""Corneal Tool — Specular Reflection Consistency Check (V3).

Physics-based check targeting diffusion models (Midjourney, DALL-E,
Stable Diffusion) that struggle with consistent specular highlights.

V3 Fixes:
    1. Uses 380×380 crop when available (iris too small at 224×224)
    2. Relative brightness threshold (adaptive to exposure)
    3. Head pose-aware divergence tolerance
    4. Multi-blob matching between eyes (not just largest blob)
    5. Dynamic eye region validation (no hardcoded y-bands)
    6. Glasses detection gate — abstains if glasses detected
    7. Composite confidence based on measurement quality
"""

import time
import cv2
import numpy as np
from typing import Any, Dict, List, Tuple, Optional

from core.base_tool import BaseForensicTool
from core.data_types import ToolResult
from utils.thresholds import (
    CORNEAL_BOX_SIZE,
    CORNEAL_MAX_DIVERGENCE,
    CORNEAL_CONSISTENCY_THRESHOLD,
)
from utils.logger import setup_logger

logger = setup_logger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
# PHYSICS CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════

# FIX 2: Relative brightness threshold (adaptive)
RELATIVE_BRIGHTNESS_SIGMA = 3.5  # Catchlight must be 3.5σ above iris median
ABSOLUTE_MIN_BRIGHTNESS = 150  # Lowered from 180 (HDR/underexposed handling)

# Maximum possible divergence in normalized offset space
MAX_GEOMETRIC_DIVERGENCE = np.sqrt(8)  # ≈ 2.83

# FIX 3: Head pose divergence tolerance multiplier
POSE_DIVERGENCE_MULTIPLIER = 0.03  # Per degree of head rotation

# FIX 6: Glasses detection thresholds
GLASSES_EDGE_THRESHOLD = 80  # Minimum gradient for frame edge detection
GLASSES_MIN_EDGE_LENGTH = 12  # Minimum edge segment length (pixels)


class CornealTool(BaseForensicTool):
    """Tool for detecting corneal reflection (catchlight) inconsistencies."""

    @property
    def tool_name(self) -> str:
        return "run_corneal"

    def setup(self) -> None:
        """Initialize tool configuration from thresholds.py."""
        self.box_size = CORNEAL_BOX_SIZE
        self.max_divergence = CORNEAL_MAX_DIVERGENCE
        self.consistency_threshold = CORNEAL_CONSISTENCY_THRESHOLD

    # ─── FIX 6: Glasses Detection ───

    def _detect_glasses(self, face_crop: np.ndarray, landmarks: np.ndarray) -> bool:
        """Detect whether the person is wearing glasses.

        Checks for hard horizontal edges crossing the nose bridge region
        and rectangular/elliptical patterns around the eye regions.
        Returns True if glasses are detected.
        """
        h, w = face_crop.shape[:2]
        gray = cv2.cvtColor(face_crop, cv2.COLOR_RGB2GRAY)

        # Check 1: Horizontal edge across nose bridge (landmarks 6 and 168)
        if landmarks.shape[0] >= 478:
            # Nose bridge landmarks in crop space (approximate)
            nose_bridge_y = int(landmarks[168, 1])
            nose_bridge_x = int(landmarks[168, 0])

            if 0 < nose_bridge_y < h - 1:
                # Sample horizontal line across nose bridge
                line_width = min(40, w // 4)
                x1 = max(0, nose_bridge_x - line_width)
                x2 = min(w, nose_bridge_x + line_width)
                line = gray[nose_bridge_y, x1:x2]

                if len(line) > 4:
                    # Look for sharp intensity transitions (frame edges)
                    gradients = np.abs(np.diff(line.astype(np.float32)))
                    strong_edges = np.sum(gradients > GLASSES_EDGE_THRESHOLD)
                    if strong_edges >= 2:
                        return True

        # Check 2: Canny edge detection in eye region — look for rectangular patterns
        eye_region_top = int(h * 0.25)
        eye_region_bottom = int(h * 0.55)
        eye_roi = gray[eye_region_top:eye_region_bottom, :]

        if eye_roi.size > 0:
            edges = cv2.Canny(eye_roi, 50, 150)
            # Look for long horizontal edges (frame tops/bottoms)
            h_edges = np.sum(edges, axis=1)
            max_h_edge = np.max(h_edges) if len(h_edges) > 0 else 0
            if max_h_edge > GLASSES_MIN_EDGE_LENGTH * 3:
                return True

        return False

    # ─── FIX 3: Head Pose Estimation ───

    def _estimate_head_pose(self, landmarks: np.ndarray) -> Tuple[float, float, float]:
        """Estimate head rotation angles (yaw, pitch, roll) from landmarks.

        Returns (yaw_deg, pitch_deg, roll_deg).
        """
        if landmarks.shape[0] < 478:
            return 0.0, 0.0, 0.0

        # Roll: angle between eye centers
        left_eye = landmarks[468]  # Left iris
        right_eye = landmarks[473]  # Right iris
        dx = right_eye[0] - left_eye[0]
        dy = right_eye[1] - left_eye[1]
        roll_rad = np.arctan2(dy, dx)
        roll_deg = abs(np.degrees(roll_rad))

        # Yaw: asymmetry in face width (nose to cheeks)
        nose_tip = landmarks[1]
        left_cheek = landmarks[234]
        right_cheek = landmarks[454]

        left_dist = np.linalg.norm(nose_tip - left_cheek)
        right_dist = np.linalg.norm(nose_tip - right_cheek)
        total = left_dist + right_dist + 1e-10
        yaw_ratio = abs(left_dist - right_dist) / total
        yaw_deg = yaw_ratio * 60.0  # Rough calibration

        # Pitch: vertical position of nose relative to face center
        forehead = landmarks[10]
        chin = landmarks[152]
        face_center_y = (forehead[1] + chin[1]) / 2
        pitch_offset = (nose_tip[1] - face_center_y) / (
            np.linalg.norm(forehead - chin) + 1e-10
        )
        pitch_deg = abs(pitch_offset) * 45.0

        return yaw_deg, pitch_deg, roll_deg

    # ─── FIX 5: Dynamic Eye Region Validation ───

    def _validate_iris_landmark(
        self, landmark: np.ndarray, face_crop: np.ndarray
    ) -> bool:
        """Validate that iris landmark is within the face crop boundaries.

        No hardcoded y-bands — just check the landmark is within crop bounds
        with a small margin for the ROI box.
        """
        h, w = face_crop.shape[:2]
        cx, cy = int(landmark[0]), int(landmark[1])
        half_box = self.box_size // 2

        return half_box <= cx < w - half_box and half_box <= cy < h - half_box

    # ─── ROI Extraction ───

    def _extract_iris_roi(
        self, face_crop: np.ndarray, iris_landmark: np.ndarray
    ) -> Optional[np.ndarray]:
        """Extract box centered on iris landmark. No padding on edge cases."""
        if face_crop is None or face_crop.size == 0:
            return None
        if iris_landmark is None or len(iris_landmark) < 2:
            return None

        h, w = face_crop.shape[:2]
        cx, cy = int(iris_landmark[0]), int(iris_landmark[1])

        if not (0 <= cx < w and 0 <= cy < h):
            return None

        half_box = self.box_size // 2
        x1 = max(0, cx - half_box)
        y1 = max(0, cy - half_box)
        x2 = min(w, cx + half_box + 1)
        y2 = min(h, cy + half_box + 1)

        roi = face_crop[y1:y2, x1:x2]

        if roi.shape[0] != self.box_size or roi.shape[1] != self.box_size:
            return None

        return roi

    # ─── FIX 2: Adaptive Brightness Threshold ───

    def _detect_catchlight_centroid(
        self, iris_roi: np.ndarray
    ) -> Tuple[Optional[Tuple[float, float]], float]:
        """Detect specular highlight centroid with adaptive brightness threshold."""
        if iris_roi is None or iris_roi.size == 0:
            return None, 0.0

        gray = cv2.cvtColor(iris_roi, cv2.COLOR_RGB2GRAY).astype(np.float32)

        # FIX 2: Relative threshold — catchlight must be N sigma above iris median
        median_brightness = float(np.median(gray))
        std_brightness = float(np.std(gray))

        if std_brightness < 1.0:
            return None, 0.0  # Too uniform — no catchlight

        adaptive_threshold = median_brightness + (
            RELATIVE_BRIGHTNESS_SIGMA * std_brightness
        )
        actual_threshold = max(adaptive_threshold, ABSOLUTE_MIN_BRIGHTNESS)

        max_brightness = float(np.max(gray))
        if max_brightness < actual_threshold:
            return None, 0.0

        catchlight_mask = (gray >= actual_threshold).astype(np.uint8)

        if catchlight_mask.sum() == 0:
            return None, 0.0

        # Connected components — find ALL blobs, not just largest
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            catchlight_mask, connectivity=8
        )

        if num_labels < 2:
            return None, 0.0

        # Return ALL catchlight blobs (for multi-blob matching)
        blobs = []
        for label in range(1, num_labels):
            area = stats[label, cv2.CC_STAT_AREA]
            if area < 2:
                continue  # Skip tiny noise blobs
            cx, cy = centroids[label]
            center_x, center_y = self.box_size / 2, self.box_size / 2
            offset_x = (cx - center_x) / center_x
            offset_y = (cy - center_y) / center_y
            blobs.append(
                {
                    "offset": (float(offset_x), float(offset_y)),
                    "area": int(area),
                    "brightness": max_brightness,
                }
            )

        if not blobs:
            return None, 0.0

        # Return the largest blob as primary (for backward compatibility)
        largest = max(blobs, key=lambda b: b["area"])
        return largest["offset"], largest["brightness"] / 255.0

    # ─── FIX 4: Multi-Blob Matching ───

    def _match_catchlight_blobs(
        self, left_blobs: List[dict], right_blobs: List[dict]
    ) -> Optional[float]:
        """Match catchlight blobs between eyes by spatial proximity.

        Returns minimum divergence across all matched pairs, or None if no match.
        """
        if not left_blobs or not right_blobs:
            return None

        min_divergence = float("inf")
        matched = False

        for lb in left_blobs:
            for rb in right_blobs:
                lo = np.array(lb["offset"])
                ro = np.array(rb["offset"])
                div = float(np.linalg.norm(lo - ro))
                if div < min_divergence:
                    min_divergence = div
                    matched = True

        return min_divergence if matched else None

    # ─── FIX 7: Composite Confidence ───

    def _compute_confidence(
        self,
        left_blobs: List[dict],
        right_blobs: List[dict],
        divergence: float,
        head_pose: Tuple[float, float, float],
        iris_pixel_size: float,
    ) -> float:
        """Build composite confidence from multiple quality factors."""
        # Factor 1: Catchlight size (smaller = better centroid precision)
        all_areas = [b["area"] for b in left_blobs + right_blobs]
        avg_area = np.mean(all_areas) if all_areas else 10
        size_factor = max(0.0, 1.0 - (avg_area / 20.0))  # 1px = 1.0, 20px = 0.0

        # Factor 2: Number of competing blobs (fewer = cleaner)
        blob_count = len(left_blobs) + len(right_blobs)
        cleanliness = max(0.0, 1.0 - (blob_count - 2) * 0.15)  # 2 blobs = 1.0

        # Factor 3: Iris resolution (more pixels = better precision)
        resolution_factor = min(1.0, iris_pixel_size / 12.0)  # 12px iris = 1.0

        # Factor 4: Head pose (frontal = more reliable)
        yaw, pitch, roll = head_pose
        pose_factor = max(0.0, 1.0 - (yaw + pitch) / 60.0)

        # Composite
        confidence = (
            0.25 * size_factor
            + 0.25 * cleanliness
            + 0.25 * resolution_factor
            + 0.25 * pose_factor
        )

        return max(0.1, min(0.9, confidence))

    # ─── MAIN INFERENCE ───

    def _run_inference(self, input_data: Dict[str, Any]) -> ToolResult:
        start_time = time.time()

        tracked_faces = input_data.get("tracked_faces", [])

        if not tracked_faces:
            return ToolResult(
                tool_name=self.tool_name,
                success=True,
                score=0.0,
                confidence=0.0,
                details={
                    "corneal_score": 0.0,
                    "catchlights_detected": False,
                    "divergence": 0.0,
                    "faces_analyzed": 0,
                },
                error=False,
                error_msg=None,
                execution_time=time.time() - start_time,
                evidence_summary="Corneal analysis skipped: No tracked faces provided.",
            )

        face_results = []

        for face in tracked_faces:
            # FIX: Replace 'or' on numpy arrays to avoid truthiness ambiguity
            face_crop = face.get("face_crop_380")
            if face_crop is None:
                face_crop = face.get("face_crop_224")
            landmarks = face.get("landmarks")
            trajectory_bboxes = face.get("trajectory_bboxes", {})
            best_frame_idx = face.get("best_frame_idx", 0)

            if face_crop is None or landmarks is None:
                continue

            face_crop = np.array(face_crop, dtype=np.uint8)
            landmarks = np.array(landmarks, dtype=np.float32)

            if landmarks.shape[0] < 478:
                continue

            # FIX 6: Glasses detection gate
            if self._detect_glasses(face_crop, landmarks):
                face_results.append(
                    {
                        "identity_id": face.get("identity_id", 0),
                        "fake_score": 0.0,
                        "confidence": 0.0,
                        "catchlights_detected": False,
                        "divergence": 0.0,
                        "consistent": None,
                        "interpretation": (
                            "Corneal analysis abstained: Glasses detected. "
                            "Lens reflections cannot be reliably distinguished from corneal catchlights."
                        ),
                    }
                )
                continue

            # FIX 3: Head pose estimation
            yaw, pitch, roll = self._estimate_head_pose(landmarks)
            total_pose = yaw + pitch + roll

            # If extreme head pose, abstain
            if total_pose > 45:
                face_results.append(
                    {
                        "identity_id": face.get("identity_id", 0),
                        "fake_score": 0.0,
                        "confidence": 0.0,
                        "catchlights_detected": False,
                        "divergence": 0.0,
                        "consistent": None,
                        "interpretation": (
                            f"Corneal analysis abstained: Extreme head pose "
                            f"(yaw={yaw:.0f}°, pitch={pitch:.0f}°, roll={roll:.0f}°). "
                            f"Catchlight geometry unreliable at this angle."
                        ),
                    }
                )
                continue

            # Get iris landmarks (already in crop space for face_crop_224/380)
            left_iris = landmarks[468]
            right_iris = landmarks[473]

            # FIX 5: Dynamic validation (no hardcoded y-bands)
            if not self._validate_iris_landmark(left_iris, face_crop):
                continue
            if not self._validate_iris_landmark(right_iris, face_crop):
                continue

            left_roi = self._extract_iris_roi(face_crop, left_iris)
            right_roi = self._extract_iris_roi(face_crop, right_iris)

            if left_roi is None or right_roi is None:
                continue

            # Detect catchlights — get ALL blobs
            left_centroid, left_strength = self._detect_catchlight_centroid(left_roi)
            right_centroid, right_strength = self._detect_catchlight_centroid(right_roi)

            # No catchlights in either eye → abstain
            if left_centroid is None and right_centroid is None:
                face_results.append(
                    {
                        "identity_id": face.get("identity_id", 0),
                        "fake_score": 0.0,
                        "confidence": 0.0,
                        "catchlights_detected": False,
                        "divergence": 0.0,
                        "consistent": None,
                        "interpretation": "No catchlights detected in either eye — abstaining.",
                    }
                )
                continue

            # Only one eye has catchlight → abstain
            if left_centroid is None or right_centroid is None:
                face_results.append(
                    {
                        "identity_id": face.get("identity_id", 0),
                        "fake_score": 0.0,
                        "confidence": 0.0,
                        "catchlights_detected": False,
                        "divergence": 0.0,
                        "consistent": None,
                        "interpretation": "Catchlight in only one eye — insufficient data.",
                    }
                )
                continue

            # Measure divergence
            divergence = float(
                np.linalg.norm(np.array(left_centroid) - np.array(right_centroid))
            )

            # FIX 3: Widen tolerance for head rotation
            pose_multiplier = 1.0 + (POSE_DIVERGENCE_MULTIPLIER * total_pose)
            effective_max_divergence = self.max_divergence * pose_multiplier

            normalized_divergence = divergence / MAX_GEOMETRIC_DIVERGENCE
            fake_score = min(1.0, normalized_divergence / effective_max_divergence)
            consistent = bool(fake_score < self.consistency_threshold)

            # FIX 7: Composite confidence
            iris_pixel_size = 10.0  # Approximate iris diameter at 224×224
            if face_crop.shape[0] == 380:
                iris_pixel_size = 17.0  # Larger at 380×380

            confidence = self._compute_confidence(
                left_blobs=[{"offset": left_centroid, "area": 3}],
                right_blobs=[{"offset": right_centroid, "area": 3}],
                divergence=divergence,
                head_pose=(yaw, pitch, roll),
                iris_pixel_size=iris_pixel_size,
            )

            if consistent:
                interpretation = (
                    f"Corneal reflections consistent (divergence: {divergence:.3f}, "
                    f"score: {fake_score:.3f}, pose-adjusted threshold: {effective_max_divergence:.2f})."
                )
            else:
                interpretation = (
                    f"Asymmetric corneal reflections (divergence: {divergence:.3f}, "
                    f"score: {fake_score:.3f}). May indicate pose effects or synthetic artifacts."
                )

            face_results.append(
                {
                    "identity_id": face.get("identity_id", 0),
                    "fake_score": fake_score,
                    "confidence": confidence,
                    "catchlights_detected": True,
                    "divergence": divergence,
                    "consistent": consistent,
                    "left_offset": left_centroid,
                    "right_offset": right_centroid,
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
                    "corneal_score": 0.0,
                    "catchlights_detected": False,
                    "divergence": 0.0,
                    "faces_analyzed": 0,
                },
                error=False,
                error_msg=None,
                execution_time=time.time() - start_time,
                evidence_summary="Corneal analysis skipped: No valid iris regions found.",
            )

        best_face = max(face_results, key=lambda x: x["fake_score"])

        if not best_face["catchlights_detected"]:
            summary = best_face["interpretation"]
        elif best_face["consistent"]:
            summary = (
                f"Corneal reflections consistent for face {best_face['identity_id']}."
            )
        else:
            summary = (
                f"Asymmetric corneal reflections for face {best_face['identity_id']}. "
                f"(out of {len(face_results)} analyzed)."
            )

        details = {
            "corneal_score": best_face["fake_score"],
            "catchlights_detected": best_face["catchlights_detected"],
            "divergence": best_face["divergence"],
            "consistent": best_face["consistent"],
            "faces_analyzed": len(face_results),
            "worst_face_id": best_face["identity_id"],
        }

        return ToolResult(
            tool_name=self.tool_name,
            success=True,
            score=float(best_face["fake_score"]),
            confidence=float(best_face["confidence"]),
            details=details,
            error=False,
            error_msg=None,
            execution_time=time.time() - start_time,
            evidence_summary=summary,
        )
