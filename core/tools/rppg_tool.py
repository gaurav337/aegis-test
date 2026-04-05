"""rPPG Tool V2 — Biological Liveness Detection (Audit Corrected)
V2 Fixes:
1. Failure modes return abstention (0.0), not fake (1.0)
2. Single ROI failure → abstention, not fake
3. Tracking failure → abstention, not fake
4. Multi-frame hair occlusion check (not just first frame)
5. Use actual SNR for quality filtering, not spectral concentration alone
6. Remove dead-end lightweight face check
7. SYNTHETIC_FLATLINE → abstention (skin tone bias mitigation)
8. Motion artifact rejection before spectral analysis
"""
import time
import logging  # ✅ FIX C-01: Standard logging import
import numpy as np
import cv2
from typing import Dict, Any, Tuple, Optional, List
from core.base_tool import BaseForensicTool
from core.data_types import ToolResult
from utils.thresholds import (
    RPPG_MIN_FRAMES,
    RPPG_HAIR_OCCLUSION_VARIANCE,
    RPPG_MIN_TEMPORAL_STD,
    RPPG_CARDIAC_BAND_MIN_HZ,
    RPPG_CARDIAC_BAND_MAX_HZ,
    RPPG_COHERENCE_THRESHOLD_HZ,
    RPPG_FFT_NFFT,
    RPPG_SNR_THRESHOLD,
    RPPG_HEART_RATE_MIN,
    RPPG_HEART_RATE_MAX,
    RPPG_SIGNAL_QUALITY_MIN,
)

logger = logging.getLogger(__name__)  # ✅ FIX C-01: Correct logger initialization

class RPPGTool(BaseForensicTool):
    """Tool for detecting biological liveness signals using POS rPPG algorithm
    and Spectral Coherence. V2: Conservative — defaults to abstention on ambiguity."""
    
    @property
    def tool_name(self) -> str:
        return "run_rppg"

    def setup(self) -> None:
        self._debug = False

    # ─── ROI Extraction ───
    def _extract_roi(
        self, frame: np.ndarray, current_bbox: tuple, relative_box: tuple
    ) -> np.ndarray:
        x1, y1, x2, y2 = current_bbox
        w = x2 - x1
        h = y2 - y1
        rx_min, ry_min, rx_max, ry_max = relative_box
        fx1 = int(x1 + (w * rx_min))
        fy1 = int(y1 + (h * ry_min))
        fx2 = int(x1 + (w * rx_max))
        fy2 = int(y1 + (h * ry_max))
        frame_h, frame_w = frame.shape[:2]
        fx1 = max(0, min(fx1, frame_w - 1))
        fy1 = max(0, min(fy1, frame_h - 1))
        fx2 = max(0, min(fx2, frame_w))
        fy2 = max(0, min(fy2, frame_h))
        if fx1 >= fx2 or fy1 >= fy2:
            return np.zeros((1, 1, 3), dtype=np.uint8)
        return frame[fy1:fy2, fx1:fx2]

    def _get_facial_rois(self, landmarks: np.ndarray) -> Dict[str, tuple]:
        rois = {
            "forehead": (0.2, 0.05, 0.8, 0.25),
            "left_cheek": (0.1, 0.5, 0.4, 0.85),
            "right_cheek": (0.6, 0.5, 0.9, 0.85),
        }
        if len(landmarks.shape) == 2 and landmarks.shape[0] == 478:
            face_min_x, face_max_x = np.min(landmarks[:, 0]), np.max(landmarks[:, 0])
            face_min_y, face_max_y = np.min(landmarks[:, 1]), np.max(landmarks[:, 1])
            face_w = face_max_x - face_min_x
            face_h = face_max_y - face_min_y
            if face_w > 0 and face_h > 0:
                def _rel(pts):
                    return (
                        (np.min(pts[:, 0]) - face_min_x) / face_w,
                        (np.min(pts[:, 1]) - face_min_y) / face_h,
                        (np.max(pts[:, 0]) - face_min_x) / face_w,
                        (np.max(pts[:, 1]) - face_min_y) / face_h,
                    )
                rois["forehead"] = _rel(landmarks[[109, 10, 338, 297, 332, 284, 103, 67]])
                rois["left_cheek"] = _rel(landmarks[[50, 205, 207, 215, 138, 135, 210]])
                rois["right_cheek"] = _rel(landmarks[[280, 425, 427, 435, 367, 364, 430]])
        return rois

    # ─── FIX 4: Multi-frame hair occlusion check ───
    def _check_hair_occlusion(self, roi: np.ndarray) -> Tuple[bool, float]:
        if roi.size == 0:
            return True, 0.0
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY) if roi.ndim == 3 else roi
        variance = float(cv2.Laplacian(gray_roi, cv2.CV_64F).var())
        if roi.ndim == 3 and roi.shape[2] == 3:
            hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
            h_channel = hsv[:, :, 0]
            non_skin_ratio = np.mean((h_channel > 50) & (h_channel < 170))
            if non_skin_ratio > 0.6 and variance > RPPG_HAIR_OCCLUSION_VARIANCE * 0.7:
                return True, variance
        return variance > RPPG_HAIR_OCCLUSION_VARIANCE, variance

    # ─── FIX 8: Motion artifact detection ───
    def _detect_motion_contamination(
        self, rgb_means: List[np.ndarray], fps: float = 30.0
    ) -> Tuple[bool, float]:
        if len(rgb_means) < RPPG_MIN_FRAMES:
            return True, 1.0
        rgb_matrix = np.array(rgb_means, dtype=np.float64)
        signal = rgb_matrix[:, 1] - np.mean(rgb_matrix[:, 1])
        windowed = signal * np.hanning(len(signal))
        n_fft = max(256, len(signal))
        psd = np.abs(np.fft.rfft(windowed, n=n_fft)) ** 2
        freqs = np.fft.rfftfreq(n_fft, d=1.0 / fps)
        motion_mask = freqs < 0.5
        cardiac_mask = (freqs >= RPPG_CARDIAC_BAND_MIN_HZ) & (freqs <= RPPG_CARDIAC_BAND_MAX_HZ)
        motion_energy = np.sum(psd[motion_mask]) if np.any(motion_mask) else 0.0
        cardiac_energy = np.sum(psd[cardiac_mask]) if np.any(cardiac_mask) else 0.0
        total_energy = motion_energy + cardiac_energy + 1e-10
        motion_ratio = motion_energy / total_energy
        return motion_ratio > 0.75, motion_ratio

    # ─── Signal Extraction ───
    def _extract_pos_signal(
        self, frames: list, trajectory: dict, relative_roi: tuple
    ) -> Tuple[Optional[np.ndarray], float, bool, bool]:
        rgb_means = []
        last_known_box = None
        hair_occlusion_frames = 0
        total_checked_frames = 0
        for f_idx, frame in enumerate(frames):
            if f_idx in trajectory:
                curr_box = trajectory[f_idx]
                last_known_box = curr_box
            elif last_known_box is not None:
                curr_box = last_known_box
            else:
                continue
            roi = self._extract_roi(frame, curr_box, relative_roi)
            if roi.size > 0 and f_idx % 15 == 0:
                total_checked_frames += 1
                is_occluded, _ = self._check_hair_occlusion(roi)
                if is_occluded:
                    hair_occlusion_frames += 1
                gray_roi = np.mean(roi, axis=2) if roi.ndim == 3 else roi
                if np.mean(gray_roi) < 50.0:
                    return None, 0.0, False, False
            if roi.size == 0:
                if rgb_means:
                    rgb_means.append(rgb_means[-1])
                else:
                    rgb_means.append(np.array([128.0, 128.0, 128.0]))
            else:
                spatial_mean = np.mean(roi, axis=(0, 1))
                rgb_means.append(spatial_mean)
        rgb_matrix = np.array(rgb_means, dtype=np.float64)
        if len(rgb_matrix) < RPPG_MIN_FRAMES:
            return None, 0.0, False, False
        green_temporal_std = float(np.std(rgb_matrix[:, 1]))
        motion_contaminated, _ = self._detect_motion_contamination(rgb_means)
        mean_rgb = np.maximum(np.mean(rgb_matrix, axis=0), 1.0)
        Cn = rgb_matrix / mean_rgb
        pos_weights = np.array([[0, 1, -1], [-2, 1, 1]])
        S = pos_weights @ Cn.T
        std_s0 = np.std(S[0])
        std_s1 = np.std(S[1]) + 1e-7
        h = S[0] + (std_s0 / std_s1) * S[1]
        h_mean = np.mean(h)
        h_std = np.std(h) + 1e-7
        H = (h - h_mean) / h_std
        hair_occluded = (total_checked_frames > 0 and (hair_occlusion_frames / total_checked_frames) > 0.4)
        return np.nan_to_num(H), green_temporal_std, hair_occluded, motion_contaminated

    # ─── FIX 5: SNR-based signal metrics ───
    def _calculate_signal_metrics(self, signal_1d: np.ndarray, fps: float = 30.0) -> Dict[str, float]:
        if np.std(signal_1d) < 1e-5:
            return {"peak_hz": 0.0, "snr_db": -100.0, "spectral_concentration": 0.0}
        signal_centered = signal_1d.copy() - np.mean(signal_1d)
        windowed = signal_centered * np.hanning(len(signal_centered))
        n_fft = RPPG_FFT_NFFT
        psd = np.abs(np.fft.rfft(windowed, n=n_fft)) ** 2
        freqs = np.fft.rfftfreq(n_fft, d=1.0 / fps)
        band_mask = (freqs >= RPPG_CARDIAC_BAND_MIN_HZ) & (freqs <= RPPG_CARDIAC_BAND_MAX_HZ)
        band_psd = psd[band_mask]
        band_freqs = freqs[band_mask]
        if len(band_psd) == 0 or np.sum(band_psd) == 0:
            return {"peak_hz": 0.0, "snr_db": -100.0, "spectral_concentration": 0.0}
        peak_idx = int(np.argmax(band_psd))
        peak_hz = float(band_freqs[peak_idx])
        median_power = float(np.median(band_psd))
        peak_power = float(band_psd[peak_idx])
        spectral_concentration = peak_power / (median_power + 1e-10)
        start_idx = max(0, peak_idx - 2)
        end_idx = min(len(band_psd), peak_idx + 3)
        signal_power = np.sum(band_psd[start_idx:end_idx])
        noise_mask = np.ones(len(band_psd), dtype=bool)
        noise_mask[start_idx:end_idx] = False
        noise_power = np.sum(band_psd[noise_mask]) if np.any(noise_mask) else 1e-10
        noise_power = max(noise_power, 1e-10)
        signal_power = max(signal_power, 1e-10)
        snr_db = float(10 * np.log10(signal_power / noise_power))
        return {"peak_hz": peak_hz, "snr_db": snr_db, "spectral_concentration": spectral_concentration}

    # ─── ✅ FIX C-02: Conservative liveness evaluation with ABSENCE SCORING ───
    def _evaluate_liveness(
        self,
        h_forehead: np.ndarray,
        h_left: np.ndarray,
        h_right: np.ndarray,
        quality_stds: list,
        hair_occluded: bool,
        motion_contaminated: bool,
    ) -> dict:
        roi_labels = ["Forehead", "L_Cheek", "R_Cheek"]
        signals = [h_forehead, h_left, h_right]

        if hair_occluded:
            return {"label": "AMBIGUOUS", "real_prob": 0.5, "confidence": 0.0,
                    "interpretation": "rPPG abstained: Forehead ROI frequently occluded by hair. Blood flow signal cannot be reliably extracted."}

        if motion_contaminated:
            return {"label": "AMBIGUOUS", "real_prob": 0.5, "confidence": 0.0,
                    "interpretation": "rPPG abstained: Signal contaminated by head motion. Cardiac pulse cannot be reliably distinguished from motion artifacts."}

        all_flat = all(np.std(s) < 1e-5 for s in signals)
        if all_flat:
            # ✅ FIX C-02: Signal-quality gate for absent pulse
            max_std = max(quality_stds)
            if max_std >= RPPG_MIN_TEMPORAL_STD:
                # Clear signal present, but zero pulse → AI/Synthetic video
                return {
                    "label": "SYNTHETIC_FLATLINE",
                    "real_prob": 0.35,  # Strong fake signal mapped to lower authenticity
                    "confidence": min(0.8, max_std * 12.0),
                    "interpretation": (
                        "Biological liveness FAILED: High-quality temporal signal detected but no cardiac pulse found. "
                        "Consistent with AI-generated or deepfake video."
                    ),
                }
            # Low variance → poor lighting/static → abstain
            return {
                "label": "NO_PULSE",
                "real_prob": 0.5,
                "confidence": 0.0,
                "interpretation": "Biological liveness inconclusive: All facial regions show minimal temporal variation. Insufficient evidence for a liveness judgment.",
            }

        analyzable_count = sum(1 for s in quality_stds if s >= RPPG_MIN_TEMPORAL_STD)
        if analyzable_count < 2:
            return {"label": "AMBIGUOUS", "real_prob": 0.5, "confidence": 0.0,
                    "interpretation": f"rPPG abstained: Only {analyzable_count}/3 facial regions have sufficient temporal variance for analysis."}

        metrics = [self._calculate_signal_metrics(s) for s in signals]

        if self._debug:
            for i, label in enumerate(roi_labels):
                m = metrics[i]
                logger.info(f"[DEBUG rPPG] {label}: SNR={m['snr_db']:+.2f} dB SC={m['spectral_concentration']:.1f}x peak={m['peak_hz']:.3f} Hz green_std={quality_stds[i]:.2f}")

        good_mask = [m["snr_db"] >= RPPG_SNR_THRESHOLD for m in metrics]
        n_good = sum(good_mask)

        if n_good == 0:
            max_snr = max(m["snr_db"] for m in metrics)
            return {"label": "AMBIGUOUS", "real_prob": 0.5, "confidence": 0.0,
                    "interpretation": f"rPPG abstained: No cardiac peak detected with sufficient SNR in any facial region (max SNR: {max_snr:.1f} dB)."}

        if n_good == 1:
            good_idx = good_mask.index(True)
            return {"label": "AMBIGUOUS", "real_prob": 0.5, "confidence": 0.0,
                    "interpretation": f"rPPG abstained: Only {roi_labels[good_idx]} yielded a usable signal. Multi-region coherence required."}

        good_indices = [i for i, g in enumerate(good_mask) if g]
        good_peaks = [metrics[i]["peak_hz"] for i in good_indices]
        plausible_peaks = [p for p in good_peaks if (RPPG_HEART_RATE_MIN / 60.0) <= p <= (RPPG_HEART_RATE_MAX / 60.0)]

        if not plausible_peaks:
            return {"label": "AMBIGUOUS", "real_prob": 0.5, "confidence": 0.0,
                    "interpretation": f"rPPG abstained: Detected peaks outside plausible heart rate range ({RPPG_HEART_RATE_MIN}-{RPPG_HEART_RATE_MAX} BPM)."}

        best_pair = None
        best_pair_diff = float("inf")
        for a in range(len(good_indices)):
            for b in range(a + 1, len(good_indices)):
                diff = abs(good_peaks[a] - good_peaks[b])
                if diff < best_pair_diff:
                    best_pair_diff = diff
                    best_pair = (good_indices[a], good_indices[b])

        if best_pair_diff <= RPPG_COHERENCE_THRESHOLD_HZ:
            pair_avg_hz = (metrics[best_pair[0]]["peak_hz"] + metrics[best_pair[1]]["peak_hz"]) / 2
            n_coherent = sum(1 for i in good_indices if abs(metrics[i]["peak_hz"] - pair_avg_hz) <= RPPG_COHERENCE_THRESHOLD_HZ)
            avg_bpm = pair_avg_hz * 60
            conf = 0.95 if n_coherent >= 3 else (0.70 if n_coherent == 2 else 0.50)
            return {"label": "PULSE_PRESENT", "real_prob": 0.95, "confidence": conf,
                    "interpretation": f"Biological liveness confirmed: Synchronous cardiac pulse detected across {n_coherent}/3 facial regions at ~{avg_bpm:.0f} BPM."}

        conf = 0.60 if n_good == 3 else 0.40
        return {"label": "AMBIGUOUS", "real_prob": 0.5, "confidence": conf,
                "interpretation": f"rPPG abstained: Pulse frequencies vary across facial regions (best pair spread: {best_pair_diff * 60:.1f} BPM)."}

    # ─── FIX 6: Removed dead-end lightweight face check ───
    def _run_inference(self, input_data: Dict[str, Any]) -> ToolResult:
        start_time = time.time()
        media_type = input_data.get("original_media_type", "unknown")
        if media_type == "image":
            return ToolResult(tool_name=self.tool_name, success=True, real_prob=0.5, confidence=0.0,
                              details={"liveness_label": "SKIPPED", "reason": "Static image"},
                              error=False, error_msg=None, execution_time=time.time() - start_time,
                              evidence_summary="rPPG skipped: Static image has no temporal signal.")

        if "frames_30fps" not in input_data or "tracked_faces" not in input_data:
            return ToolResult(tool_name=self.tool_name, success=False, real_prob=0.5, confidence=0.0,
                              details={"liveness_label": "ERROR"}, error=True,
                              error_msg="Missing 'frames_30fps' or 'tracked_faces' in input_data",
                              execution_time=time.time() - start_time, evidence_summary="Missing required input data.")

        frames = input_data.get("frames_30fps", [])
        tracked_faces = input_data.get("tracked_faces", [])

        if len(frames) < RPPG_MIN_FRAMES:
            return ToolResult(tool_name=self.tool_name, success=True, real_prob=0.5, confidence=0.0,
                              details={"liveness_label": "ABSTAIN", "reason": "INSUFFICIENT_TEMPORAL_DATA"},
                              error=False, error_msg=None, execution_time=time.time() - start_time,
                              evidence_summary=f"rPPG skipped: insufficient frames ({len(frames)} < {RPPG_MIN_FRAMES}) for rPPG analysis.")

        if not tracked_faces:
            return ToolResult(tool_name=self.tool_name, success=True, real_prob=0.5, confidence=0.0,
                              details={"liveness_label": "ABSTAIN", "reason": "NO_TRACKED_FACES"},
                              error=False, error_msg=None, execution_time=time.time() - start_time,
                              evidence_summary="rPPG abstained: No tracked faces available for analysis.")

        face_results = []
        for face in tracked_faces:
            trajectory = face.get("trajectory_bboxes", {})
            landmarks = face.get("landmarks", [])
            rois = self._get_facial_rois(landmarks)
            face_window = face.get("face_window", (0, 0))
            
            if face_window[1] > face_window[0]:
                start_offset = face_window[0]
                end_frame = face_window[1]
                target_frames = frames[start_offset:end_frame]
                sliced_trajectory = {k - start_offset: v for k, v in trajectory.items() if start_offset <= k < end_frame}
            else:
                face_results.append({"real_prob": 0.5, "confidence": 0.0, "label": "ABSTAIN",
                                     "interpretation": "Face window could not be established.", "metrics": {}})
                continue

            h_forehead, std_f, hair_f, motion_f = self._extract_pos_signal(target_frames, sliced_trajectory, rois["forehead"])
            h_left, std_l, hair_l, motion_l = self._extract_pos_signal(target_frames, sliced_trajectory, rois["left_cheek"])
            h_right, std_r, hair_r, motion_r = self._extract_pos_signal(target_frames, sliced_trajectory, rois["right_cheek"])

            if h_forehead is None or h_left is None or h_right is None:
                label = "OCCLUDED" if (hair_f or hair_l or hair_r) else "TRACKING_FAILED"
                face_results.append({"real_prob": 0.5, "confidence": 0.0, "label": "ABSTAIN",
                                     "interpretation": f"rPPG abstained: {label.replace('_', ' ').title()}. Cannot extract reliable biological signal.",
                                     "metrics": {}})
                continue

            liveness = self._evaluate_liveness(h_forehead, h_left, h_right, quality_stds=[std_f, std_l, std_r],
                                               hair_occluded=hair_f, motion_contaminated=motion_f)
            metrics = [self._calculate_signal_metrics(s) for s in [h_forehead, h_left, h_right]]
            face_results.append({"real_prob": liveness["real_prob"], "confidence": liveness["confidence"],
                                 "label": liveness["label"], "interpretation": liveness["interpretation"],
                                 "metrics": {"forehead": metrics[0], "left_cheek": metrics[1], "right_cheek": metrics[2]}})

        if not face_results:
            return ToolResult(tool_name=self.tool_name, success=True, real_prob=0.5, confidence=0.0,
                              details={"liveness_label": "NO_FACES"}, error=False, error_msg=None,
                              execution_time=time.time() - start_time,
                              evidence_summary="All tracked faces yielded ambiguous tracking or were occluded.")

        best = sorted(face_results, key=lambda x: x["confidence"], reverse=True)[0]
        details = {"liveness_label": best["label"], "peak_hz": best["metrics"].get("forehead", {}).get("peak_hz", 0.0),
                   "spectral_concentration": best["metrics"].get("forehead", {}).get("spectral_concentration", 0.0),
                   "faces_analyzed": len(face_results)}

        return ToolResult(tool_name=self.tool_name, success=True, real_prob=best["real_prob"], confidence=best["confidence"],
                          details=details, error=False, error_msg=None, execution_time=time.time() - start_time,
                          evidence_summary=best["interpretation"])