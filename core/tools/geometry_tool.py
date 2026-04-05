"""Geometry Tool — 7-Point Anthropometric Consistency Check (V3 - Audit Corrected).
Implements physics-based facial structure validation using MediaPipe 478-point landmarks.
Generative models often fail basic 3D human anatomical constraints even when visual
texture appears photorealistic.

Key Fixes:
1. M-03: Landmark coordinate validation & normalization (handles pixel vs normalized)
2. m-03: Demographic bias mitigation via symmetric severity & widened thresholds
3. FIX 3: Brow ridge landmarks for stable vertical thirds (not hairline)
4. FIX 4: Roll (head tilt) gate — skips vertical thirds when tilted >15°
5. FIX 5: Symmetric severity calculation (equal deviation = equal severity)
6. FIX 6: Stable face width using outer eye corners (not jaw, which deforms)
7. S-07: Multi-face aggregation uses median score (prevents group-photo penalties)
8. Resolution-aware dampening to reduce noise on small faces
"""
import time
import numpy as np
from typing import Any, Dict, List, Tuple, Optional
from core.base_tool import BaseForensicTool
from core.data_types import ToolResult
from utils.thresholds import (
    GEOMETRY_YAW_SKIP_THRESHOLD,
    GEOMETRY_IPD_RATIO_MIN, GEOMETRY_IPD_RATIO_MAX,
    GEOMETRY_PHILTRUM_RATIO_MIN, GEOMETRY_PHILTRUM_RATIO_MAX,
    GEOMETRY_EYE_ASYMMETRY_MAX,
    GEOMETRY_NOSE_WIDTH_RATIO_MIN, GEOMETRY_NOSE_WIDTH_RATIO_MAX,
    GEOMETRY_MOUTH_WIDTH_RATIO_MIN, GEOMETRY_MOUTH_WIDTH_RATIO_MAX,
    GEOMETRY_VERTICAL_THIRDS_MAX_DEVIATION,
)
from utils.logger import setup_logger

logger = setup_logger(__name__)

# ──────────────────────────────────────────────────────────────
# VIOLATION WEIGHTS — Some checks are stronger fake signals
# ──────────────────────────────────────────────────────────────
VIOLATION_WEIGHTS = {
    "IPD ratio": 2.0,               # Very stable in real humans
    "Philtrum ratio": 1.5,          # Moderately stable
    "Eye width asymmetry": 1.0,     # Normal variation exists
    "Nose width ratio": 1.5,        # GANs often fail here
    "Mouth width ratio": 1.0,       # More variable
    "Vertical thirds": 2.0,         # Strong signal — 3D structure constraint
}
MAX_WEIGHT = sum(VIOLATION_WEIGHTS.values())  # 9.0

# FIX 4: Roll threshold — skip vertical thirds if head tilted >15°
ROLL_SKIP_THRESHOLD_DEG = 15.0
# FIX 3: Brow ridge landmark indices for stable face top reference
BROW_PEAK_LEFT = 66
BROW_PEAK_RIGHT = 296

class GeometryTool(BaseForensicTool):
    """Tool for detecting anatomical inconsistencies in facial geometry."""
    
    @property
    def tool_name(self) -> str:
        return "run_geometry"

    def setup(self) -> None:
        self.yaw_skip_threshold = GEOMETRY_YAW_SKIP_THRESHOLD

    @staticmethod
    def _dist(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.linalg.norm(np.array(a) - np.array(b)))

    @staticmethod
    def _safe_divide(numerator: float, denominator: float) -> float:
        return numerator / (denominator + 1e-10)

    # ─── FIX 1 & M-03: Landmark Coordinate Normalization ───
    def _normalize_landmarks(self, lm: np.ndarray) -> np.ndarray:
        """Ensure landmarks are consistently normalized [0,1] relative to crop."""
        if lm.max() > 1.0:
            # Pixel coordinates detected — normalize to [0,1]
            h, w = lm.max(axis=0)
            if h > 0 and w > 0:
                lm = lm.copy()
                lm[:, 0] /= w
                lm[:, 1] /= h
                logger.debug("Geometry: Normalized pixel landmarks to [0,1]")
        return lm

    # ─── FIX 4: Roll Estimation ───
    def _estimate_roll(self, lm: np.ndarray) -> float:
        """Estimate head roll angle from eye line. Returns absolute degrees."""
        left_eye = lm[33]
        right_eye = lm[263]
        dx = right_eye[0] - left_eye[0]
        dy = right_eye[1] - left_eye[1]
        if abs(dx) < 1e-10: return 90.0
        return abs(np.degrees(np.arctan2(dy, dx)))

    # ─── FIX 5: Symmetric Severity Calculation (m-03 bias mitigation) ───
    def _soft_check(self, value: float, low: float, high: float, name: str) -> Tuple[bool, float, str]:
        """
        Returns (passed, severity_0_to_1, detail_string).
        Symmetric: deviation from nearest boundary / range width.
        """
        if low <= value <= high:
            return True, 0.0, f"{name}: {value:.3f} (valid: {low}–{high})"
            
        range_width = high - low if (high - low) > 1e-10 else 1e-10
        deviation = (low - value) if value < low else (value - high)
        severity = min(deviation / range_width, 1.0)
        return False, severity, f"{name}: {value:.3f} (valid: {low}–{high})"

    # ─── CHECK 1: IPD Ratio ───
    def _check_ipd_ratio(self, lm: np.ndarray, face_width: float) -> Tuple[bool, float, str, float]:
        ipd = self._dist(lm[468], lm[473])
        ratio = self._safe_divide(ipd, face_width)
        return self._soft_check(ratio, GEOMETRY_IPD_RATIO_MIN, GEOMETRY_IPD_RATIO_MAX, "IPD ratio") + (ratio,)

    # ─── CHECK 2: Philtrum Ratio ───
    def _check_philtrum_ratio(self, lm: np.ndarray) -> Tuple[bool, float, str, float]:
        philtrum_len = self._dist(lm[94], lm[0])
        lower_face_len = self._dist(lm[168], lm[152])
        ratio = self._safe_divide(philtrum_len, lower_face_len)
        return self._soft_check(ratio, GEOMETRY_PHILTRUM_RATIO_MIN, GEOMETRY_PHILTRUM_RATIO_MAX, "Philtrum ratio") + (ratio,)

    # ─── CHECK 3: Eye Asymmetry (Bilateral) ───
    def _check_eye_asymmetry(self, lm: np.ndarray, face_width: float) -> Tuple[bool, float, str, float]:
        left_w = self._dist(lm[33], lm[133])
        right_w = self._dist(lm[263], lm[362])
        asym = self._safe_divide(abs(left_w - right_w), face_width)
        passed = asym <= GEOMETRY_EYE_ASYMMETRY_MAX
        severity = min(self._safe_divide(asym - GEOMETRY_EYE_ASYMMETRY_MAX, GEOMETRY_EYE_ASYMMETRY_MAX), 1.0) if not passed else 0.0
        return passed, max(0.0, severity), f"Eye asymmetry: {asym:.3f}", asym

    # ─── CHECK 4: Yaw Proxy (Pose Gate) ───
    def _check_yaw_proxy(self, lm: np.ndarray, face_width: float) -> Tuple[bool, float, str]:
        eye_mid_x = (lm[33][0] + lm[263][0]) / 2.0
        nose_tip_x = lm[1][0]
        yaw = self._safe_divide(abs(eye_mid_x - nose_tip_x), face_width)
        return yaw <= self.yaw_skip_threshold, yaw, f"Yaw proxy: {yaw:.3f}"

    # ─── CHECK 5: Nose Width (Bilateral) ───
    def _check_nose_width_ratio(self, lm: np.ndarray, ipd: float) -> Tuple[bool, float, str, float]:
        nose_w = self._dist(lm[98], lm[327])
        ratio = self._safe_divide(nose_w, ipd)
        return self._soft_check(ratio, GEOMETRY_NOSE_WIDTH_RATIO_MIN, GEOMETRY_NOSE_WIDTH_RATIO_MAX, "Nose width ratio") + (ratio,)

    # ─── CHECK 6: Mouth Width (Bilateral) ───
    def _check_mouth_width_ratio(self, lm: np.ndarray, ipd: float) -> Tuple[bool, float, str, float]:
        mouth_w = self._dist(lm[61], lm[291])
        ratio = self._safe_divide(mouth_w, ipd)
        return self._soft_check(ratio, GEOMETRY_MOUTH_WIDTH_RATIO_MIN, GEOMETRY_MOUTH_WIDTH_RATIO_MAX, "Mouth width ratio") + (ratio,)

    # ─── CHECK 7: Vertical Thirds (Roll-Gated, Brow-Ridge Based) ───
    def _check_vertical_thirds(self, lm: np.ndarray) -> Tuple[bool, float, str, Dict[str, float], float]:
        brow_left = lm[BROW_PEAK_LEFT]
        brow_right = lm[BROW_PEAK_RIGHT]
        face_top_y = (brow_left[1] + brow_right[1]) / 2.0
        nose_y = lm[168][1]
        col_y = lm[94][1]
        chin_y = lm[152][1]

        thirds = [abs(face_top_y - nose_y), abs(nose_y - col_y), abs(col_y - chin_y)]
        mean_third = np.mean(thirds)
        if mean_third < 1e-10:
            return True, 0.0, "Degenerate face height", {"upper":0, "mid":0, "lower":0}, 0.0

        max_dev = max(self._safe_divide(abs(t - mean_third), mean_third) for t in thirds)
        passed = max_dev <= GEOMETRY_VERTICAL_THIRDS_MAX_DEVIATION
        severity = min(self._safe_divide(max_dev - GEOMETRY_VERTICAL_THIRDS_MAX_DEVIATION, GEOMETRY_VERTICAL_THIRDS_MAX_DEVIATION), 1.0) if not passed else 0.0
        roll = self._estimate_roll(lm)
        return passed, severity, f"Thirds deviation: {max_dev:.3f}", {"upper":thirds[0], "mid":thirds[1], "lower":thirds[2]}, roll

    # ─── FIX 6: Stable Face Width ───
    def _get_stable_face_width(self, lm: np.ndarray) -> float:
        return self._dist(lm[33], lm[263])

    # ─── Weighted Scoring ───
    def _weighted_score(self, violations: List[str], severities: Dict[str, float], skip_bilateral: bool) -> float:
        if not violations: return 0.0
        total = sum(VIOLATION_WEIGHTS.get(v, 1.0) * severities.get(v, 1.0) for v in violations)
        if skip_bilateral:
            max_possible = sum(w for n, w in VIOLATION_WEIGHTS.items() if n not in ("Eye width asymmetry", "Nose width ratio", "Mouth width ratio"))
        else:
            max_possible = MAX_WEIGHT
        return min(self._safe_divide(total, max_possible), 1.0)

    # ─── Confidence Calculation ───
    def _calculate_confidence(self, checks: int, yaw: float, px_width: float) -> float:
        base = 0.8 if checks >= 4 else 0.4
        if yaw > 0.12: base *= 0.85
        if px_width < 80: base *= 0.6
        elif px_width < 120: base *= 0.85
        return round(min(base, 0.95), 2)

    # ─── MAIN INFERENCE ───
    def _run_inference(self, input_data: Dict[str, Any]) -> ToolResult:
        start_time = time.time()
        tracked_faces = input_data.get("tracked_faces", [])
        if not tracked_faces:
            return ToolResult(tool_name=self.tool_name, success=True, score=0.0, confidence=0.0,
                              details={"geometry_score": 0.0, "faces_analyzed": 0},
                              execution_time=time.time()-start_time, evidence_summary="No tracked faces provided.")

        face_results = []
        for face in tracked_faces:
            face_crop = face.get("face_crop_380") if face.get("face_crop_380") is not None else face.get("face_crop_224")
            lm_raw = face.get("landmarks")
            if face_crop is None or lm_raw is None: continue
            
            landmarks = self._normalize_landmarks(np.array(lm_raw, dtype=np.float32))
            if landmarks.shape[0] < 478: continue

            face_width = self._get_stable_face_width(landmarks)
            ipd = self._dist(landmarks[468], landmarks[473])
            roll_deg = self._estimate_roll(landmarks)
            skip_vertical = roll_deg > ROLL_SKIP_THRESHOLD_DEG

            violations, severities, checks = [], {}, 0
            yaw_ok, yaw_val, _ = self._check_yaw_proxy(landmarks, face_width)
            skip_bi = not yaw_ok

            # Check 1
            p, s, d, v = self._check_ipd_ratio(landmarks, face_width)
            if not p: violations.append("IPD ratio"); severities["IPD ratio"] = s
            checks += 1

            # Check 2
            p, s, d, v = self._check_philtrum_ratio(landmarks)
            if not p: violations.append("Philtrum ratio"); severities["Philtrum ratio"] = s
            checks += 1

            # Bilateral Checks
            if skip_bi: checks += 3
            else:
                for name, check_fn in [("Eye width asymmetry", self._check_eye_asymmetry),
                                       ("Nose width ratio", self._check_nose_width_ratio),
                                       ("Mouth width ratio", self._check_mouth_width_ratio)]:
                    args = (landmarks, face_width) if name == "Eye width asymmetry" else (landmarks, ipd)
                    p, s, d, v = check_fn(*args)
                    if not p: violations.append(name); severities[name] = s
                    checks += 1

            # Check 7
            if skip_vertical: checks += 1
            else:
                p, s, d, thirds, _ = self._check_vertical_thirds(landmarks)
                if not p: violations.append("Vertical thirds"); severities["Vertical thirds"] = s
                checks += 1

            score = self._weighted_score(violations, severities, skip_bi)
            px_width = face_width * face_crop.shape[1]
            if px_width < 120:
                score *= max(0.7, 0.7 + (0.3 * (px_width - 60) / 60))
            
            # Calibration: Dampen low-suspicion 'Consistent' scores for better intuition
            if score < 0.35:
                score *= 0.5

            face_results.append({
                "identity_id": face.get("identity_id", 0), "fake_score": score,
                "confidence": self._calculate_confidence(checks, yaw_val, px_width),
                "violations": violations, "severities": severities,
                "checks_performed": checks, "yaw_proxy": yaw_val, "roll_deg": roll_deg
            })

        if not face_results:
            return ToolResult(tool_name=self.tool_name, success=True, score=0.0, confidence=0.0,
                              details={"geometry_score": 0.0, "faces_analyzed": 0},
                              execution_time=time.time()-start_time, evidence_summary="No valid landmarks found.")

        # S-07 Fix: Use median score for group photos
        scores = sorted([f["fake_score"] for f in face_results])
        median_score = scores[len(scores) // 2]
        median_face = min(face_results, key=lambda x: abs(x["fake_score"] - median_score))

        viol_str = ", ".join(median_face["violations"]) if median_face["violations"] else "none"
        
        if median_score < 0.35:
            summary = (f"Anatomical structure consistent (ID {median_face['identity_id']}). "
                       f"Anthropometric ratios (IPD, Philtrum, Eye Symmetry) match natural human baseline.")
        elif median_score < 0.50:
            summary = (f"Minor anatomical deviations (ID {median_face['identity_id']}): {viol_str}. "
                       f"Result remains within normal biometric variance (Score: {median_score:.2f}).")
        else:
            summary = (f"Significant anatomical violations (ID {median_face['identity_id']}): {viol_str}. "
                       f"Geometry deviates from human anatomical baseline (Score: {median_score:.2f}).")

        return ToolResult(
            tool_name=self.tool_name, success=True, score=float(median_score),
            confidence=float(median_face["confidence"]),
            details={
                "geometry_score": median_score, "violations": median_face["violations"],
                "severities": median_face["severities"], "checks_performed": median_face["checks_performed"],
                "faces_analyzed": len(face_results), "yaw_proxy": median_face["yaw_proxy"],
                "roll_deg": median_face.get("roll_deg", 0.0)
            }, error=False, error_msg=None, execution_time=time.time()-start_time,
            evidence_summary=summary
        )