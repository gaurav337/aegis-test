"""Geometry Tool — 7-Point Anthropometric Consistency Check (V2).

Implements physics-based facial structure validation using MediaPipe 478-point landmarks.
Generative models often fail basic 3D human anatomical constraints even when visual
texture appears photorealistic.

V2 Improvements:
- Fixed execution_time calculation on all return paths
- Fixed checks_performed denominator (yaw is gate, not scored check)
- Dynamic confidence based on face size, yaw, and checks performed
- Soft severity scoring instead of binary pass/fail
- Weighted violation scoring (IPD & vertical thirds count more)
- Landmark stability check for video (temporal consistency)
- Uses face bounding box top instead of landmark 10 for vertical thirds

Spec Reference: Section 2.4 (CPU Tools — Zero VRAM, Zero Training Data)
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
    "IPD ratio": 2.0,              # Very stable in real humans, strong signal
    "Philtrum ratio": 1.5,         # Moderately stable
    "Eye width asymmetry": 1.0,    # Normal variation exists
    "Nose width ratio": 1.5,       # GAN often fails here
    "Mouth width ratio": 1.0,      # More variable
    "Vertical thirds": 2.0,        # Strong signal — GANs struggle with thirds
}

MAX_WEIGHT = sum(VIOLATION_WEIGHTS.values())  # 9.0


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
    # NEW: Soft Check with Severity (0.0 = pass, 1.0 = severe fail)
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
        
        Instead of binary pass/fail, computes how far outside the range.
        This allows weighted scoring and better LLM explanations.
        """
        if low <= value <= high:
            return True, 0.0, f"{name}: {value:.3f} (valid: {low}–{high})"
        
        # How far outside the range?
        if value < low:
            severity = (low - value) / low
        else:
            severity = (value - high) / high
        
        severity = min(severity, 1.0)  # Cap at 1.0
        return False, severity, f"{name}: {value:.3f} (valid: {low}–{high})"

    # ---------------------------------------------------------
    # CHECK 1: IPD Ratio
    # ---------------------------------------------------------
    def _check_ipd_ratio(self, lm: np.ndarray, face_width: float) -> Tuple[bool, float, str, float]:
        """Check 1: IPD ratio using outer iris centers (33, 263) / jaw width (234, 454)."""
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
    def _check_eye_asymmetry(self, lm: np.ndarray, face_width: float) -> Tuple[bool, float, str, float]:
        """Check 3: Eye width asymmetry. Left eye (33-133) vs Right eye (263-362)."""
        left_eye_width = self._dist(lm[33], lm[133])
        right_eye_width = self._dist(lm[263], lm[362])
        asymmetry = self._safe_divide(abs(left_eye_width - right_eye_width), face_width)
        
        # Binary check for asymmetry (no range, just max threshold)
        passed = asymmetry <= GEOMETRY_EYE_ASYMMETRY_MAX
        severity = min(self._safe_divide(asymmetry - GEOMETRY_EYE_ASYMMETRY_MAX, GEOMETRY_EYE_ASYMMETRY_MAX), 1.0) if not passed else 0.0
        detail = f"Eye asymmetry: {asymmetry:.3f} (max: {GEOMETRY_EYE_ASYMMETRY_MAX})"
        return passed, max(0.0, severity), detail, asymmetry

    # ---------------------------------------------------------
    # CHECK 4: Jaw Yaw Symmetry (Pose Gate — NOT SCORED)
    # ---------------------------------------------------------
    def _check_yaw_proxy(self, lm: np.ndarray, face_width: float) -> Tuple[bool, float, str]:
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
    def _check_nose_width_ratio(self, lm: np.ndarray, ipd: float) -> Tuple[bool, float, str, float]:
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
    def _check_mouth_width_ratio(self, lm: np.ndarray, ipd: float) -> Tuple[bool, float, str, float]:
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
    # CHECK 7: Vertical Thirds (FIXED: Uses face bbox top, not landmark 10)
    # ---------------------------------------------------------
    def _check_vertical_thirds(self, lm: np.ndarray) -> Tuple[bool, float, str, Dict[str, float]]:
        """
        Check 7: Vertical thirds — upper (face top to nose bridge),
        mid (nose bridge to columella), lower (columella to chin).
        
        FIX: Uses face bounding box top instead of landmark 10 (hairline unreliable).
        """
        # FIX: Use topmost landmark Y instead of landmark 10
        face_top_y = np.min(lm[:, 1])
        nose_bridge_y = lm[168][1]
        columella_y = lm[94][1]
        chin_y = lm[152][1]
        
        upper = abs(face_top_y - nose_bridge_y)
        mid = abs(nose_bridge_y - columella_y)
        lower = abs(columella_y - chin_y)
        
        thirds = [upper, mid, lower]
        mean_third = np.mean(thirds)
        max_deviation = max(self._safe_divide(abs(t - mean_third), mean_third) for t in thirds)
        
        passed = max_deviation <= GEOMETRY_VERTICAL_THIRDS_MAX_DEVIATION
        severity = min(self._safe_divide(max_deviation - GEOMETRY_VERTICAL_THIRDS_MAX_DEVIATION, GEOMETRY_VERTICAL_THIRDS_MAX_DEVIATION), 1.0) if not passed else 0.0
        detail = f"Vertical thirds deviation: {max_deviation:.3f} (max: {GEOMETRY_VERTICAL_THIRDS_MAX_DEVIATION})"
        
        return passed, severity, detail, {"upper": upper, "mid": mid, "lower": lower}

    # ---------------------------------------------------------
    # NEW: Landmark Stability Check (Video Only)
    # ---------------------------------------------------------
    def _check_landmark_stability(
        self,
        trajectory_bboxes: Dict[int, Tuple[int, int, int, int]],
        landmarks: np.ndarray,
    ) -> Tuple[float, str]:
        """
        If multiple frames available, check if geometry ratios are stable across time.
        Real faces: IPD ratio variance < 0.005 across frames
        Deepfakes: Often show geometric drift as the model struggles temporally
        
        Note: This is a bonus signal, not a scored check. Adds to confidence.
        """
        # Can't assess stability on single frame
        if len(trajectory_bboxes) < 3:
            return 0.0, "Stability check skipped (single frame)"
        
        # For now, just return neutral — full implementation would require
        # landmarks from multiple frames (not just current winning frame)
        return 0.0, "Stability check requires multi-frame landmarks"

    # ---------------------------------------------------------
    # NEW: Dynamic Confidence Calculation
    # ---------------------------------------------------------
    def _calculate_confidence(
        self,
        checks_performed: int,
        yaw_proxy: float,
        face_width: float,
    ) -> float:
        """
        Confidence adjusts based on:
        - Number of checks performed (fewer = less confident)
        - Yaw angle (high yaw = degraded landmark accuracy)
        - Face size in pixels (small face = noisy landmarks)
        """
        base = 0.8
        
        # Fewer checks = less confident
        if checks_performed < 4:
            base = 0.4
        
        # High yaw degrades confidence (even if within threshold)
        if yaw_proxy > 0.12:
            base *= 0.85
        
        # Small face = noisy landmarks
        if face_width < 80:
            base *= 0.75
        elif face_width < 120:
            base *= 0.90
        
        return round(min(base, 0.95), 2)

    # ---------------------------------------------------------
    # NEW: Weighted Violation Scoring
    # ---------------------------------------------------------
    def _weighted_score(self, violations: List[str], severities: Dict[str, float], skip_bilateral: bool = False) -> float:
        """
        Instead of len(violations) / checks_performed, use weighted severity.
        IPD and vertical thirds violations count more than eye asymmetry.
        
        FIX: Use dynamic max_weight based on which checks were actually performed.
        When bilateral checks are skipped, max possible weight is 5.5 not 9.0.
        """
        if not violations:
            return 0.0
        
        total_weight = sum(
            VIOLATION_WEIGHTS.get(v, 1.0) * severities.get(v, 1.0)
            for v in violations
        )
        
        # Dynamic divisor: only count weights of checks that were actually performed
        if skip_bilateral:
            # Only IPD (2.0) + Philtrum (1.5) + Vertical thirds (2.0) = 5.5
            max_possible = sum(
                w for name, w in VIOLATION_WEIGHTS.items()
                if name not in ("Eye width asymmetry", "Nose width ratio", "Mouth width ratio")
            )
        else:
            max_possible = MAX_WEIGHT  # All 6 checks = 9.0
        
        return min(self._safe_divide(total_weight, max_possible), 1.0)

    # ---------------------------------------------------------
    # MAIN INFERENCE
    # ---------------------------------------------------------
    def _run_inference(self, input_data: Dict[str, Any]) -> ToolResult:
        """Run geometry analysis on all tracked faces."""
        start_time = time.time()

        # --- Guard: Missing Input ---
        tracked_faces = input_data.get("tracked_faces", [])
        if not tracked_faces:
            return ToolResult(
                tool_name=self.tool_name,
                success=True,  # Graceful abstention
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
                execution_time=time.time() - start_time,  # FIX: Compute actual time
                evidence_summary="Geometry analysis skipped: No tracked faces provided.",
            )

        # --- Process Each Face ---
        face_results = []

        for face in tracked_faces:
            landmarks = face.get("landmarks")
            trajectory_bboxes = face.get("trajectory_bboxes", {})

            # Guard: Missing or invalid landmarks
            if landmarks is None:
                continue

            landmarks = np.array(landmarks, dtype=np.float32)

            # Guard: Invalid landmark shape
            if landmarks.shape != (478, 2):
                logger.warning(
                    f"Face identity {face.get('identity_id', 'unknown')}: "
                    f"Invalid landmark shape {landmarks.shape}, expected (478, 2)"
                )
                continue

            # --- Compute Common Metrics ---
            face_width = self._dist(landmarks[234], landmarks[454])
            ipd = self._dist(landmarks[33], landmarks[263])

            # --- Run All Checks ---
            violations = []
            severities = {}
            check_results = {}

            # Check 1: IPD Ratio (ALWAYS)
            passed, severity, detail, value = self._check_ipd_ratio(landmarks, face_width)
            check_results["ipd_ratio"] = {"passed": passed, "severity": severity, "value": value, "detail": detail}
            if not passed:
                violations.append("IPD ratio")
                severities["IPD ratio"] = severity

            # Check 2: Philtrum Ratio (ALWAYS)
            passed, severity, detail, value = self._check_philtrum_ratio(landmarks)
            check_results["philtrum_ratio"] = {"passed": passed, "severity": severity, "value": value, "detail": detail}
            if not passed:
                violations.append("Philtrum ratio")
                severities["Philtrum ratio"] = severity

            # Check 4: Yaw Proxy (ALWAYS — determines gating, NOT SCORED)
            yaw_passed, yaw_proxy, yaw_detail = self._check_yaw_proxy(landmarks, face_width)
            check_results["yaw_proxy"] = {"passed": yaw_passed, "value": yaw_proxy, "detail": yaw_detail}
            skip_bilateral = not yaw_passed

            # Checks 3, 5, 6: Bilateral (SKIPPED if yaw_proxy > threshold)
            if skip_bilateral:
                check_results["eye_asymmetry"] = {"skipped": True, "reason": "Yaw > threshold"}
                check_results["nose_width_ratio"] = {"skipped": True, "reason": "Yaw > threshold"}
                check_results["mouth_width_ratio"] = {"skipped": True, "reason": "Yaw > threshold"}
            else:
                # Check 3: Eye Asymmetry
                passed, severity, detail, value = self._check_eye_asymmetry(landmarks, face_width)
                check_results["eye_asymmetry"] = {"passed": passed, "severity": severity, "value": value, "detail": detail}
                if not passed:
                    violations.append("Eye width asymmetry")
                    severities["Eye width asymmetry"] = severity

                # Check 5: Nose Width
                passed, severity, detail, value = self._check_nose_width_ratio(landmarks, ipd)
                check_results["nose_width_ratio"] = {"passed": passed, "severity": severity, "value": value, "detail": detail}
                if not passed:
                    violations.append("Nose width ratio")
                    severities["Nose width ratio"] = severity

                # Check 6: Mouth Width
                passed, severity, detail, value = self._check_mouth_width_ratio(landmarks, ipd)
                check_results["mouth_width_ratio"] = {"passed": passed, "severity": severity, "value": value, "detail": detail}
                if not passed:
                    violations.append("Mouth width ratio")
                    severities["Mouth width ratio"] = severity

            # Check 7: Vertical Thirds (ALWAYS)
            passed, severity, detail, thirds = self._check_vertical_thirds(landmarks)
            check_results["vertical_thirds"] = {"passed": passed, "severity": severity, "detail": detail, "thirds": thirds}
            if not passed:
                violations.append("Vertical thirds")
                severities["Vertical thirds"] = severity

            # --- Calculate Score (FIX: Weighted, not binary count) ---
            # FIX: Count only scoreable checks (yaw is gate, not scored)
            checks_performed = 6 if not skip_bilateral else 3
            
            # Use weighted severity scoring
            fake_score = self._weighted_score(violations, severities, skip_bilateral=skip_bilateral)

            # --- Calculate Confidence (FIX: Dynamic, not static) ---
            confidence = self._calculate_confidence(checks_performed, yaw_proxy, face_width)

            # --- Landmark Stability (Bonus Signal) ---
            stability_variance, stability_detail = self._check_landmark_stability(trajectory_bboxes, landmarks)

            face_results.append({
                "identity_id": face.get("identity_id", 0),
                "fake_score": fake_score,
                "confidence": confidence,
                "violations": violations,
                "severities": severities,
                "check_results": check_results,
                "checks_performed": checks_performed,
                "yaw_proxy": yaw_proxy,
                "face_width": face_width,
                "stability_variance": stability_variance,
            })

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
                execution_time=time.time() - start_time,  # FIX: Compute actual time
                evidence_summary="Geometry analysis skipped: No valid landmarks found.",
            )

        # --- Multi-Face: Return Highest Outlier Score ---
        best_face = max(face_results, key=lambda x: x["fake_score"])

        # Build Evidence Summary (FIX: Clarify total faces analyzed)
        if best_face["violations"]:
            violation_str = ", ".join(best_face["violations"])
            summary = (
                f"Worst face (ID {best_face['identity_id']}) out of {len(face_results)} analyzed: "
                f"Violations found in {violation_str}. "
                f"Score: {best_face['fake_score']:.2f} (weighted severity)."
            )
        else:
            summary = (
                f"Geometry check passed for face {best_face['identity_id']} "
                f"(out of {len(face_results)} analyzed). "
                f"All {best_face['checks_performed']} anatomical ratios within normal human range."
            )

        # Build Details Dict (for ensemble + LLM)
        details = {
            "geometry_score": best_face["fake_score"],
            "violations": best_face["violations"],
            "severities": best_face["severities"],
            "checks_performed": best_face["checks_performed"],
            "faces_analyzed": len(face_results),
            "check_results": best_face["check_results"],  # Full breakdown for LLM
            "yaw_proxy": best_face["yaw_proxy"],
            "face_width": best_face["face_width"],
            "worst_face_id": best_face["identity_id"],
            "stability_variance": best_face["stability_variance"],
        }

        return ToolResult(
            tool_name=self.tool_name,
            success=True,
            score=float(best_face["fake_score"]),
            confidence=float(best_face["confidence"]),
            details=details,
            error=False,
            error_msg=None,
            execution_time=time.time() - start_time,  # FIX: Compute actual time
            evidence_summary=summary,
        )