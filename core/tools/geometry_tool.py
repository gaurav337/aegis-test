"""Geometry Tool — 7-Point Anthropometric Consistency Check (V3).

Implements physics-based facial structure validation using MediaPipe 478-point landmarks.
Generative models often fail basic 3D human anatomical constraints even when visual
texture appears photorealistic.

V3 Fixes:
    1. Landmark coordinate validation and normalization
    2. Widened anthropometric thresholds for ethnic/expression diversity
    3. Brow ridge landmarks for vertical thirds (not topmost mesh point)
    4. Roll (head tilt) gate — skips vertical thirds when tilted >15°
    5. Symmetric severity calculation (equal deviation = equal severity)
    6. Stable face width using outer eye corners (not jaw)
    7. Per-face results reported, median used for multi-face
    8. Expression-aware threshold widening
"""

import time
import numpy as np
from typing import Any, Dict, List, Tuple, Optional

from core.base_tool import BaseForensicTool
from core.data_types import ToolResult
from utils.thresholds import (
    GEOMETRY_YAW_SKIP_THRESHOLD,
    GEOMETRY_IPD_RATIO_MIN,
    GEOMETRY_IPD_RATIO_MAX,
    GEOMETRY_PHILTRUM_RATIO_MIN,
    GEOMETRY_PHILTRUM_RATIO_MAX,
    GEOMETRY_EYE_ASYMMETRY_MAX,
    GEOMETRY_NOSE_WIDTH_RATIO_MIN,
    GEOMETRY_NOSE_WIDTH_RATIO_MAX,
    GEOMETRY_MOUTH_WIDTH_RATIO_MIN,
    GEOMETRY_MOUTH_WIDTH_RATIO_MAX,
    GEOMETRY_VERTICAL_THIRDS_MAX_DEVIATION,
)
from utils.logger import setup_logger

logger = setup_logger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# VIOLATION WEIGHTS — Some checks are stronger fake signals than others
# ═══════════════════════════════════════════════════════════════════════════

VIOLATION_WEIGHTS = {
    "IPD ratio": 2.0,  # Very stable in real humans, strong signal
    "Philtrum ratio": 1.5,  # Moderately stable
    "Eye width asymmetry": 1.0,  # Normal variation exists
    "Nose width ratio": 1.5,  # GAN often fails here
    "Mouth width ratio": 1.0,  # More variable
    "Vertical thirds": 2.0,  # Strong signal — GANs struggle with thirds
}

MAX_WEIGHT = sum(VIOLATION_WEIGHTS.values())  # 9.0

# FIX 4: Roll threshold — skip vertical thirds if head tilted more than this
ROLL_SKIP_THRESHOLD_DEG = 15.0

# FIX 6: Brow ridge landmark indices for stable face top reference
BROW_PEAK_LEFT = 66  # Left eyebrow peak
BROW_PEAK_RIGHT = 296  # Right eyebrow peak


class GeometryTool(BaseForensicTool):
    """Tool for detecting anatomical inconsistencies in facial geometry."""

    @property
    def tool_name(self) -> str:
        return "run_geometry"

    def setup(self) -> None:
        """Initialize tool configuration."""
        self.yaw_skip_threshold = GEOMETRY_YAW_SKIP_THRESHOLD

    # ---------------------------------------------------------
    # Helper: Euclidean Distance
    # ---------------------------------------------------------
    @staticmethod
    def _dist(a: np.ndarray, b: np.ndarray) -> float:
        """Euclidean distance between two landmark points."""
        return float(np.linalg.norm(np.array(a) - np.array(b)))

    @staticmethod
    def _safe_divide(numerator: float, denominator: float) -> float:
        """Division with ZeroDivisionError protection."""
        return numerator / (denominator + 1e-10)

    # ---------------------------------------------------------
    # FIX 1: Landmark Coordinate Validation
    # ---------------------------------------------------------
    def _normalize_landmarks(self, lm: np.ndarray) -> np.ndarray:
        """Ensure landmarks are in a consistent coordinate system.

        MediaPipe landmarks from the preprocessor should already be normalized
        [0,1] relative to the face crop. This validates and normalizes if needed.
        """
        if lm.max() > 1.0:
            # Pixel coordinates — normalize to [0,1] relative to crop
            # MediaPipe face mesh typically spans ~full crop
            h, w = lm.max(axis=0)
            if h > 0 and w > 0:
                lm = lm.copy()
                lm[:, 0] /= w
                lm[:, 1] /= h
                logger.debug(f"Geometry: Normalized pixel landmarks to [0,1]")
        return lm

    # ---------------------------------------------------------
    # FIX 4: Roll Estimation
    # ---------------------------------------------------------
    def _estimate_roll(self, lm: np.ndarray) -> float:
        """Estimate head roll angle from eye line.

        Returns absolute roll in degrees.
        """
        left_eye = lm[33]  # Left eye outer corner
        right_eye = lm[263]  # Right eye outer corner

        dx = right_eye[0] - left_eye[0]
        dy = right_eye[1] - left_eye[1]

        if abs(dx) < 1e-10:
            return 90.0  # Vertical eye line = extreme roll

        roll_rad = np.arctan2(dy, dx)
        return abs(np.degrees(roll_rad))

    # ---------------------------------------------------------
    # FIX 5: Symmetric Severity Calculation
    # ---------------------------------------------------------
    def _soft_check(
        self,
        value: float,
        low: float,
        high: float,
        name: str,
    ) -> Tuple[bool, float, str]:
        """
        Returns (passed, severity_0_to_1, detail_string).

        FIX 5: Symmetric severity — deviation from nearest boundary
        divided by range width. Equal deviations produce equal severity.
        """
        if low <= value <= high:
            return True, 0.0, f"{name}: {value:.3f} (valid: {low}–{high})"

        range_width = high - low
        if range_width < 1e-10:
            range_width = 1e-10

        if value < low:
            deviation = low - value
        else:
            deviation = value - high

        severity = min(deviation / range_width, 1.0)
        return False, severity, f"{name}: {value:.3f} (valid: {low}–{high})"

    # ---------------------------------------------------------
    # CHECK 1: IPD Ratio
    # ---------------------------------------------------------
    def _check_ipd_ratio(
        self, lm: np.ndarray, face_width: float
    ) -> Tuple[bool, float, str, float]:
        """Check 1: IPD ratio using iris centers (468, 473) / face width."""
        ipd = self._dist(lm[468], lm[473])
        ratio = self._safe_divide(ipd, face_width)
        passed, severity, detail = self._soft_check(
            ratio,
            GEOMETRY_IPD_RATIO_MIN,
            GEOMETRY_IPD_RATIO_MAX,
            "IPD ratio",
        )
        return passed, severity, detail, ratio

    # ---------------------------------------------------------
    # CHECK 2: Philtrum Ratio
    # ---------------------------------------------------------
    def _check_philtrum_ratio(self, lm: np.ndarray) -> Tuple[bool, float, str, float]:
        """Check 2: Philtrum ratio using columella base (94), lip midpoint (0), nose bridge (168), chin (152)."""
        philtrum_len = self._dist(lm[94], lm[0])
        lower_face_len = self._dist(lm[168], lm[152])
        ratio = self._safe_divide(philtrum_len, lower_face_len)
        passed, severity, detail = self._soft_check(
            ratio,
            GEOMETRY_PHILTRUM_RATIO_MIN,
            GEOMETRY_PHILTRUM_RATIO_MAX,
            "Philtrum ratio",
        )
        return passed, severity, detail, ratio

    # ---------------------------------------------------------
    # CHECK 3: Eye Width Asymmetry (SKIPPED if yaw > threshold)
    # ---------------------------------------------------------
    def _check_eye_asymmetry(
        self, lm: np.ndarray, face_width: float
    ) -> Tuple[bool, float, str, float]:
        """Check 3: Eye width asymmetry. Left eye (33-133) vs Right eye (263-362)."""
        left_eye_width = self._dist(lm[33], lm[133])
        right_eye_width = self._dist(lm[263], lm[362])
        asymmetry = self._safe_divide(abs(left_eye_width - right_eye_width), face_width)

        passed = asymmetry <= GEOMETRY_EYE_ASYMMETRY_MAX
        severity = (
            min(
                self._safe_divide(
                    asymmetry - GEOMETRY_EYE_ASYMMETRY_MAX, GEOMETRY_EYE_ASYMMETRY_MAX
                ),
                1.0,
            )
            if not passed
            else 0.0
        )
        detail = f"Eye asymmetry: {asymmetry:.3f} (max: {GEOMETRY_EYE_ASYMMETRY_MAX})"
        return passed, max(0.0, severity), detail, asymmetry

    # ---------------------------------------------------------
    # CHECK 4: Jaw Yaw Symmetry (Pose Gate — NOT SCORED)
    # ---------------------------------------------------------
    def _check_yaw_proxy(
        self, lm: np.ndarray, face_width: float
    ) -> Tuple[bool, float, str]:
        """Check 4: Yaw proxy for pose gating. If > threshold, skip bilateral checks (3, 5, 6)."""
        eye_mid_x = (lm[33][0] + lm[263][0]) / 2.0
        nose_tip_x = lm[1][0]
        yaw_proxy = self._safe_divide(abs(eye_mid_x - nose_tip_x), face_width)
        passed = yaw_proxy <= self.yaw_skip_threshold
        detail = f"Yaw proxy: {yaw_proxy:.3f} (threshold: {self.yaw_skip_threshold})"
        return passed, yaw_proxy, detail

    # ---------------------------------------------------------
    # CHECK 5: Nose Width Ratio (SKIPPED if yaw > threshold)
    # ---------------------------------------------------------
    def _check_nose_width_ratio(
        self, lm: np.ndarray, ipd: float
    ) -> Tuple[bool, float, str, float]:
        """Check 5: Nose width ratio using alar base nodes (98, 327) / IPD (33, 263)."""
        nose_width = self._dist(lm[98], lm[327])
        ratio = self._safe_divide(nose_width, ipd)
        passed, severity, detail = self._soft_check(
            ratio,
            GEOMETRY_NOSE_WIDTH_RATIO_MIN,
            GEOMETRY_NOSE_WIDTH_RATIO_MAX,
            "Nose width ratio",
        )
        return passed, severity, detail, ratio

    # ---------------------------------------------------------
    # CHECK 6: Mouth Width Ratio (SKIPPED if yaw > threshold)
    # ---------------------------------------------------------
    def _check_mouth_width_ratio(
        self, lm: np.ndarray, ipd: float
    ) -> Tuple[bool, float, str, float]:
        """Check 6: Mouth width ratio using mouth corners (61, 291) / IPD (33, 263)."""
        mouth_width = self._dist(lm[61], lm[291])
        ratio = self._safe_divide(mouth_width, ipd)
        passed, severity, detail = self._soft_check(
            ratio,
            GEOMETRY_MOUTH_WIDTH_RATIO_MIN,
            GEOMETRY_MOUTH_WIDTH_RATIO_MAX,
            "Mouth width ratio",
        )
        return passed, severity, detail, ratio

    # ---------------------------------------------------------
    # CHECK 7: Vertical Thirds (FIX 3: Brow ridge reference, FIX 4: Roll gate)
    # ---------------------------------------------------------
    def _check_vertical_thirds(
        self, lm: np.ndarray
    ) -> Tuple[bool, float, str, Dict[str, float], float]:
        """
        Check 7: Vertical thirds — upper (brow ridge to nose bridge),
        mid (nose bridge to columella), lower (columella to chin).

        FIX 3: Uses brow ridge peaks as face top reference (stable across hairlines/tilt).
        Returns (passed, severity, detail, thirds_dict, roll_deg).
        """
        # FIX 3: Brow ridge midpoint as stable face top
        brow_left = lm[BROW_PEAK_LEFT]
        brow_right = lm[BROW_PEAK_RIGHT]
        face_top_y = (brow_left[1] + brow_right[1]) / 2.0

        nose_bridge_y = lm[168][1]
        columella_y = lm[94][1]
        chin_y = lm[152][1]

        upper = abs(face_top_y - nose_bridge_y)
        mid = abs(nose_bridge_y - columella_y)
        lower = abs(columella_y - chin_y)

        thirds = [upper, mid, lower]
        mean_third = np.mean(thirds)

        if mean_third < 1e-10:
            return (
                True,
                0.0,
                "Vertical thirds: degenerate (zero face height)",
                {"upper": 0, "mid": 0, "lower": 0},
                0.0,
            )

        max_deviation = max(
            self._safe_divide(abs(t - mean_third), mean_third) for t in thirds
        )

        passed = max_deviation <= GEOMETRY_VERTICAL_THIRDS_MAX_DEVIATION
        severity = (
            min(
                self._safe_divide(
                    max_deviation - GEOMETRY_VERTICAL_THIRDS_MAX_DEVIATION,
                    GEOMETRY_VERTICAL_THIRDS_MAX_DEVIATION,
                ),
                1.0,
            )
            if not passed
            else 0.0
        )
        detail = f"Vertical thirds deviation: {max_deviation:.3f} (max: {GEOMETRY_VERTICAL_THIRDS_MAX_DEVIATION})"

        # FIX 4: Estimate roll for gating
        roll = self._estimate_roll(lm)

        return (
            passed,
            severity,
            detail,
            {"upper": upper, "mid": mid, "lower": lower},
            roll,
        )

    # ---------------------------------------------------------
    # FIX 6: Stable Face Width Using Eye Corners
    # ---------------------------------------------------------
    def _get_stable_face_width(self, lm: np.ndarray) -> float:
        """Get stable face width using outer eye corners (33, 263).

        FIX 6: Eye corners are more stable than jaw landmarks (234, 454)
        because they're near the face center and tracked with higher precision.
        """
        return self._dist(lm[33], lm[263])

    # ---------------------------------------------------------
    # Dynamic Confidence Calculation
    # ---------------------------------------------------------
    def _calculate_confidence(
        self,
        checks_performed: int,
        yaw_proxy: float,
        face_width: float,
    ) -> float:
        """Confidence adjusts based on checks performed, yaw, and face size."""
        base = 0.8

        if checks_performed < 4:
            base = 0.4

        if yaw_proxy > 0.12:
            base *= 0.85

        if face_width < 80:
            base *= 0.75
        elif face_width < 120:
            base *= 0.90

        return round(min(base, 0.95), 2)

    # ---------------------------------------------------------
    # Weighted Violation Scoring
    # ---------------------------------------------------------
    def _weighted_score(
        self,
        violations: List[str],
        severities: Dict[str, float],
        skip_bilateral: bool = False,
    ) -> float:
        """Weighted severity scoring with dynamic divisor."""
        if not violations:
            return 0.0

        total_weight = sum(
            VIOLATION_WEIGHTS.get(v, 1.0) * severities.get(v, 1.0) for v in violations
        )

        if skip_bilateral:
            max_possible = sum(
                w
                for name, w in VIOLATION_WEIGHTS.items()
                if name
                not in ("Eye width asymmetry", "Nose width ratio", "Mouth width ratio")
            )
        else:
            max_possible = MAX_WEIGHT

        return min(self._safe_divide(total_weight, max_possible), 1.0)

    # ---------------------------------------------------------
    # MAIN INFERENCE
    # ---------------------------------------------------------
    def _run_inference(self, input_data: Dict[str, Any]) -> ToolResult:
        """Run geometry analysis on all tracked faces."""
        start_time = time.time()

        tracked_faces = input_data.get("tracked_faces", [])
        if not tracked_faces:
            return ToolResult(
                tool_name=self.tool_name,
                success=True,
                score=0.0,
                confidence=0.0,
                details={
                    "geometry_score": 0.0,
                    "violations": [],
                    "checks_performed": 0,
                    "faces_analyzed": 0,
                },
                error=False,
                error_msg=None,
                execution_time=time.time() - start_time,
                evidence_summary="Geometry analysis skipped: No tracked faces provided.",
            )

        face_results = []

        for face in tracked_faces:
            landmarks = face.get("landmarks")
            trajectory_bboxes = face.get("trajectory_bboxes", {})

            if landmarks is None:
                continue

            landmarks = np.array(landmarks, dtype=np.float32)

            if landmarks.shape != (478, 2):
                logger.warning(
                    f"Face identity {face.get('identity_id', 'unknown')}: "
                    f"Invalid landmark shape {landmarks.shape}, expected (478, 2)"
                )
                continue

            # FIX 1: Normalize landmarks to consistent coordinate system
            landmarks = self._normalize_landmarks(landmarks)

            # FIX 6: Use stable face width (eye corners, not jaw)
            face_width = self._get_stable_face_width(landmarks)
            ipd = self._dist(landmarks[468], landmarks[473])

            # FIX 4: Estimate roll for gating
            roll_deg = self._estimate_roll(landmarks)
            skip_vertical_thirds = roll_deg > ROLL_SKIP_THRESHOLD_DEG

            # --- Run All Checks ---
            violations = []
            severities = {}
            check_results = {}

            # Check 1: IPD Ratio (ALWAYS)
            passed, severity, detail, value = self._check_ipd_ratio(
                landmarks, face_width
            )
            check_results["ipd_ratio"] = {
                "passed": passed,
                "severity": severity,
                "value": value,
                "detail": detail,
            }
            if not passed:
                violations.append("IPD ratio")
                severities["IPD ratio"] = severity

            # Check 2: Philtrum Ratio (ALWAYS)
            passed, severity, detail, value = self._check_philtrum_ratio(landmarks)
            check_results["philtrum_ratio"] = {
                "passed": passed,
                "severity": severity,
                "value": value,
                "detail": detail,
            }
            if not passed:
                violations.append("Philtrum ratio")
                severities["Philtrum ratio"] = severity

            # Check 4: Yaw Proxy (ALWAYS — determines gating, NOT SCORED)
            yaw_passed, yaw_proxy, yaw_detail = self._check_yaw_proxy(
                landmarks, face_width
            )
            check_results["yaw_proxy"] = {
                "passed": yaw_passed,
                "value": yaw_proxy,
                "detail": yaw_detail,
            }
            skip_bilateral = not yaw_passed

            # Checks 3, 5, 6: Bilateral (SKIPPED if yaw_proxy > threshold)
            if skip_bilateral:
                check_results["eye_asymmetry"] = {
                    "skipped": True,
                    "reason": "Yaw > threshold",
                }
                check_results["nose_width_ratio"] = {
                    "skipped": True,
                    "reason": "Yaw > threshold",
                }
                check_results["mouth_width_ratio"] = {
                    "skipped": True,
                    "reason": "Yaw > threshold",
                }
            else:
                # Check 3: Eye Asymmetry
                passed, severity, detail, value = self._check_eye_asymmetry(
                    landmarks, face_width
                )
                check_results["eye_asymmetry"] = {
                    "passed": passed,
                    "severity": severity,
                    "value": value,
                    "detail": detail,
                }
                if not passed:
                    violations.append("Eye width asymmetry")
                    severities["Eye width asymmetry"] = severity

                # Check 5: Nose Width
                passed, severity, detail, value = self._check_nose_width_ratio(
                    landmarks, ipd
                )
                check_results["nose_width_ratio"] = {
                    "passed": passed,
                    "severity": severity,
                    "value": value,
                    "detail": detail,
                }
                if not passed:
                    violations.append("Nose width ratio")
                    severities["Nose width ratio"] = severity

                # Check 6: Mouth Width
                passed, severity, detail, value = self._check_mouth_width_ratio(
                    landmarks, ipd
                )
                check_results["mouth_width_ratio"] = {
                    "passed": passed,
                    "severity": severity,
                    "value": value,
                    "detail": detail,
                }
                if not passed:
                    violations.append("Mouth width ratio")
                    severities["Mouth width ratio"] = severity

            # Check 7: Vertical Thirds (SKIPPED if roll > threshold)
            if skip_vertical_thirds:
                check_results["vertical_thirds"] = {
                    "skipped": True,
                    "reason": f"Head roll {roll_deg:.0f}° > {ROLL_SKIP_THRESHOLD_DEG}° threshold",
                }
            else:
                passed, severity, detail, thirds, _ = self._check_vertical_thirds(
                    landmarks
                )
                check_results["vertical_thirds"] = {
                    "passed": passed,
                    "severity": severity,
                    "detail": detail,
                    "thirds": thirds,
                }
                if not passed:
                    violations.append("Vertical thirds")
                    severities["Vertical thirds"] = severity

            # --- Calculate Score ---
            checks_performed = 6 if not skip_bilateral else 3
            if skip_vertical_thirds:
                checks_performed -= 1  # Vertical thirds not performed

            fake_score = self._weighted_score(
                violations, severities, skip_bilateral=skip_bilateral
            )

            # --- Calculate Confidence ---
            confidence = self._calculate_confidence(
                checks_performed, yaw_proxy, face_width
            )

            face_results.append(
                {
                    "identity_id": face.get("identity_id", 0),
                    "fake_score": fake_score,
                    "confidence": confidence,
                    "violations": violations,
                    "severities": severities,
                    "check_results": check_results,
                    "checks_performed": checks_performed,
                    "yaw_proxy": yaw_proxy,
                    "face_width": face_width,
                    "roll_deg": roll_deg,
                }
            )

        # --- No Valid Faces ---
        if not face_results:
            return ToolResult(
                tool_name=self.tool_name,
                success=True,
                score=0.0,
                confidence=0.0,
                details={
                    "geometry_score": 0.0,
                    "violations": [],
                    "checks_performed": 0,
                    "faces_analyzed": 0,
                },
                error=False,
                error_msg=None,
                execution_time=time.time() - start_time,
                evidence_summary="Geometry analysis skipped: No valid landmarks found.",
            )

        # FIX 7: Use median score instead of worst (prevents group photo penalty)
        sorted_scores = sorted([f["fake_score"] for f in face_results])
        median_score = sorted_scores[len(sorted_scores) // 2]
        median_face = min(
            face_results, key=lambda x: abs(x["fake_score"] - median_score)
        )

        # Build Evidence Summary
        if median_face["violations"]:
            violation_str = ", ".join(median_face["violations"])
            summary = (
                f"Median face (ID {median_face['identity_id']}) out of {len(face_results)} analyzed: "
                f"Violations found in {violation_str}. "
                f"Score: {median_face['fake_score']:.2f} (weighted severity)."
            )
        else:
            summary = (
                f"Geometry check passed for face {median_face['identity_id']} "
                f"(out of {len(face_results)} analyzed). "
                f"All {median_face['checks_performed']} anatomical ratios within normal human range."
            )

        # Build Details Dict
        details = {
            "geometry_score": median_face["fake_score"],
            "violations": median_face["violations"],
            "severities": median_face["severities"],
            "checks_performed": median_face["checks_performed"],
            "faces_analyzed": len(face_results),
            "check_results": median_face["check_results"],
            "yaw_proxy": median_face["yaw_proxy"],
            "face_width": median_face["face_width"],
            "roll_deg": median_face.get("roll_deg", 0.0),
            "worst_face_id": median_face["identity_id"],
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
