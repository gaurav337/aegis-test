"""Face preprocessing and patching leveraging MediaPipe Face Mesh.

Extracts standardized face crops and precise anatomical patches according
to Aegis-X Phase 1 specifications. Support tracking via CPU-SORT.
"""
import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict
from scipy.optimize import linear_sum_assignment

from core.config import PreprocessingConfig
from utils.video import extract_frames, is_video_file
from utils.image import load_image, is_image
from utils.logger import setup_logger

logger = setup_logger(__name__)

# --- CPU SORT IMPLEMENTATION ---
def iou_batch(bb_test, bb_gt):
    if len(bb_test) == 0 or len(bb_gt) == 0:
        return np.zeros((len(bb_test), len(bb_gt)))
    bb_test = np.expand_dims(bb_test, 1)
    bb_gt = np.expand_dims(bb_gt, 0)
    xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
    yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
    xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
    yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    area_test = (bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])
    area_gt = (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1])
    return wh / np.maximum(1e-6, area_test + area_gt - wh)

def compute_iou(box1, box2):
    x1, y1, x2, y2 = box1
    x3, y3, x4, y4 = box2
    xi1, yi1 = max(x1, x3), max(y1, y3)
    xi2, yi2 = min(x2, x4), min(y2, y4)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x4 - x3) * (y4 - y3)
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0

class KalmanBoxTracker:
    def __init__(self, bbox, track_id):
        self.kf = cv2.KalmanFilter(7, 4)
        self.kf.transitionMatrix = np.array([[1,0,0,0,1,0,0],[0,1,0,0,0,1,0],[0,0,1,0,0,0,1],[0,0,0,1,0,0,0],
                                             [0,0,0,0,1,0,0],[0,0,0,0,0,1,0],[0,0,0,0,0,0,1]], np.float32)
        self.kf.measurementMatrix = np.array([[1,0,0,0,0,0,0],[0,1,0,0,0,0,0],[0,0,1,0,0,0,0],[0,0,0,1,0,0,0]], np.float32)
        
        self.kf.processNoiseCov = np.eye(7, dtype=np.float32) * 0.1
        self.kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * 1.0
        self.kf.errorCovPost = np.eye(7, dtype=np.float32) * 1.0
        
        state = np.zeros((7, 1), dtype=np.float32)
        state[:4, 0] = self.convert_bbox_to_z(bbox).flatten()
        self.kf.statePost = state
        self.kf.statePre = state.copy()
        
        self.id = track_id
        self.time_since_update = 0
        self.hit_streak = 0

    def convert_bbox_to_z(self, bbox):
        w = max(1.0, float(bbox[2] - bbox[0]))
        h = max(1.0, float(bbox[3] - bbox[1]))
        x = bbox[0] + w/2.
        y = bbox[1] + h/2.
        return np.array([x, y, w*h, w/h], dtype=np.float32).reshape((4, 1))

    def convert_x_to_bbox(self, x):
        x = np.array(x).flatten()
        w = np.sqrt(abs(x[2] * x[3])) if x[2] * x[3] > 0 else 0
        h = x[2] / w if w > 0 else 0
        return [float(x[0]-w/2.), float(x[1]-h/2.), float(x[0]+w/2.), float(x[1]+h/2.)]

    def predict(self):
        if (self.kf.statePost[6, 0] + self.kf.statePost[2, 0]) <= 0:
            self.kf.statePost[6, 0] = 0.0
        self.kf.predict()
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        return self.convert_x_to_bbox(self.kf.statePre)

    def update(self, bbox):
        self.time_since_update = 0
        self.hit_streak += 1
        self.kf.correct(self.convert_bbox_to_z(bbox))

class SortTracker:
    def __init__(self, iou_threshold=0.3):
        self.trackers = []
        self.frame_count = 0
        self.iou_threshold = iou_threshold
        self.id_count = 0 

    def update(self, dets=None):
        if dets is None:
            dets = np.empty((0, 4))
        dets = np.array(dets)
        if len(dets) > 0 and dets.ndim == 1:
            dets = np.expand_dims(dets, 0)
        elif len(dets) == 0:
            dets = np.empty((0, 4))
            
        self.frame_count += 1
        trks = np.zeros((len(self.trackers), 4))
        to_del = []
        for t, trk in enumerate(self.trackers):
            pos = trk.predict()
            trks[t,:] = pos
            if np.any(np.isnan(pos)):
                to_del.append(t)
        for t in reversed(to_del):
            self.trackers.pop(t)

        matched, unmatched_dets, unmatched_trks = self.associate(dets, trks, self.iou_threshold)

        for m in matched:
            self.trackers[m[1]].update(dets[m[0]])
        for i in unmatched_dets:
            self.id_count += 1
            self.trackers.append(KalmanBoxTracker(dets[i], self.id_count))

        ret = []
        for trk in self.trackers:
            d = trk.convert_x_to_bbox(trk.kf.statePost)
            if trk.time_since_update < 1 and (trk.hit_streak >= 1 or self.frame_count <= 1):
                ret.append([d[0], d[1], d[2], d[3], trk.id])
        return np.array(ret)

    def associate(self, detections, trackers, iou_threshold):
        if len(trackers) == 0:
            return np.empty((0,2),dtype=int), np.arange(len(detections)), []
        iou_matrix = iou_batch(detections, trackers)
        if min(iou_matrix.shape) > 0:
            a = (iou_matrix > iou_threshold).astype(np.int32)
            if a.sum(1).max() == 1 and a.sum(0).max() == 1:
                matched_indices = np.stack(np.where(a), axis=1)
            else:
                matched_indices = np.asarray(linear_sum_assignment(-iou_matrix)).T
        else:
            matched_indices = np.empty((0,2),dtype=int)

        unmatched_dets = [d for d in range(len(detections)) if d not in matched_indices[:,0]]
        unmatched_trks = [t for t in range(len(trackers)) if t not in matched_indices[:,1]]

        matches = []
        for m in matched_indices:
            if iou_matrix[m[0], m[1]] < iou_threshold:
                unmatched_dets.append(m[0])
                unmatched_trks.append(m[1])
            else:
                matches.append(m.reshape(1,2))

        matches = np.concatenate(matches, axis=0) if len(matches) > 0 else np.empty((0,2),dtype=int)
        return matches, np.array(unmatched_dets), np.array(unmatched_trks)

@dataclass
class TrackedFace:
    """Represents a single detected identity tracked across multiple frames."""
    identity_id: int
    landmarks: np.ndarray
    trajectory_bboxes: Dict[int, Tuple[int, int, int, int]]
    best_frame_idx: int = -1  
    face_crop_224: Optional[np.ndarray] = None
    face_crop_380: Optional[np.ndarray] = None
    patch_left_periorbital: Optional[np.ndarray] = None
    patch_right_periorbital: Optional[np.ndarray] = None
    patch_nasolabial_left: Optional[np.ndarray] = None
    patch_nasolabial_right: Optional[np.ndarray] = None
    patch_hairline_band: Optional[np.ndarray] = None
    patch_chin_jaw: Optional[np.ndarray] = None
    face_window: Tuple[int, int] = (0, 0)
    heuristic_flags: List[str] = field(default_factory=list)

    def get(self, key, default=None):
        """Provide dict-like .get() compatibility for tools that expect dictionaries."""
        return getattr(self, key, default)

    def __getitem__(self, item):
        """Provide dict-like bracket compatibility for tools that expect dictionaries."""
        if not isinstance(item, str):
            raise KeyError(item)
        if hasattr(self, item):
            return getattr(self, item)
        raise KeyError(item)

    def __contains__(self, item):
        if not isinstance(item, str):
            return False
        return hasattr(self, item)

@dataclass
class PreprocessResult:
    """Standardized output payload from the MediaPipe face preprocessing pipeline."""
    has_face: bool
    tracked_faces: List[TrackedFace] = field(default_factory=list)
    frames_30fps: Optional[List[np.ndarray]] = None 
    first_frame: Optional[np.ndarray] = None
    selected_frame_index: int = 0
    selected_frame_sharpness: float = 0.0
    original_media_type: str = "image"
    frame_count_warning: bool = False  # FIX 7: Flag for rPPG
    max_confidence: float = 0.0
    max_face_area_ratio: float = 0.0
    frames_with_faces_pct: float = 0.0
    heuristic_flags: List[str] = field(default_factory=list)
    insufficient_temporal_data: bool = False
    
class Preprocessor:
    """MediaPipe-based robust face landmark extraction and patching class."""
    
    def __init__(self, config: PreprocessingConfig):
        self.config = config
        
        # FIX 2: Correct config attribute path
        max_faces = getattr(config.preprocessing, 'max_subjects_to_analyze', 2)
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            refine_landmarks=True,
            max_num_faces=max_faces,
            min_detection_confidence=0.5
        )
        self.tracker = SortTracker()
        
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        
    def close(self):
        if hasattr(self, 'face_mesh') and self.face_mesh is not None:
            try:
                self.face_mesh.close()
            except Exception:
                pass
            self.face_mesh = None

    def __del__(self):
        self.close()
        
    def _get_landmarks(self, image: np.ndarray) -> Optional[List[np.ndarray]]:
        if image is None or image.size == 0:
            return None
            
        try:
            results = self.face_mesh.process(image)
        except Exception as e:
            if "ResourceExhausted" in type(e).__name__:
                logger.error(f"OOM in landmark detection: {e}")
                raise
            logger.error(f"Landmark detection failed: {type(e).__name__}: {e}")
            return None
            
        if not results.multi_face_landmarks:
            return None
            
        h, w = image.shape[:2]
        faces_data = []
        for face_landmarks in results.multi_face_landmarks:
            coords = np.zeros((478, 2), dtype=np.float32)
            for i, lm in enumerate(face_landmarks.landmark):
                coords[i] = [lm.x * w, lm.y * h]
                
            nose = coords[1]
            jaw_l = coords[234]
            jaw_r = coords[454]
            valid = True
            for node in [nose, jaw_l, jaw_r]:
                if not (0 <= node[0] < w and 0 <= node[1] < h):
                    valid = False
                    break
            
            if valid:
                x_min, y_min = np.min(coords, axis=0)
                x_max, y_max = np.max(coords, axis=0)
                area = (x_max - x_min) * (y_max - y_min)
                faces_data.append((area, coords))
                
        faces_data.sort(key=lambda x: x[0], reverse=True)
        if not faces_data:
            return None
        return [data[1] for data in faces_data]
        
    def _crop_align(self, image: np.ndarray, landmarks: np.ndarray, size: int) -> np.ndarray:
        h, w = image.shape[:2]
        x_min, y_min = np.min(landmarks, axis=0)
        x_max, y_max = np.max(landmarks, axis=0)
        
        box_w = x_max - x_min
        box_h = y_max - y_min
        
        margin_x = box_w * 0.2
        margin_y = box_h * 0.2
        
        x1 = max(0, int(x_min - margin_x))
        y1 = max(0, int(y_min - margin_y))
        x2 = min(w, int(x_max + margin_x))
        y2 = min(h, int(y_max + margin_y))
        
        crop = image[y1:y2, x1:x2]
        if crop.size == 0 or crop.shape[0] == 0 or crop.shape[1] == 0:
            return np.zeros((size, size, 3), dtype=np.uint8)
            
        return cv2.resize(crop, (size, size), interpolation=cv2.INTER_LANCZOS4)
        
    def _extract_native_patches(self, image: np.ndarray, landmarks: np.ndarray) -> Tuple:
        h, w = image.shape[:2]
        size = self.config.preprocessing.native_patch_size  # FIX 2
        
        # FIX 1 & 5: Align patch landmarks with Spec Section 3.1
        patches_def = {
            "left_periorbital": [33, 160, 158, 133, 153, 144],      # Spec order
            "right_periorbital": [362, 385, 387, 263, 373, 380],    # Spec order
            "nasolabial_left": [92, 205, 216, 206],
            "nasolabial_right": [322, 425, 436, 426],
            "hairline_band": [10, 338, 297, 332, 284],              # 5 nodes (not 7)
            "chin_jaw": [172, 136, 150, 149, 176, 148, 152, 377, 400, 379, 365]
        }
        
        results = {}
        for name, nodes in patches_def.items():
            pts = landmarks[nodes]
            x_min, y_min = np.min(pts, axis=0)
            x_max, y_max = np.max(pts, axis=0)
            
            box_w = x_max - x_min
            box_h = y_max - y_min
            
            margin_x = box_w * 0.2
            margin_y = box_h * 0.2
            
            x1 = max(0, int(x_min - margin_x))
            y1 = max(0, int(y_min - margin_y))
            x2 = min(w, int(x_max + margin_x))
            y2 = min(h, int(y_max + margin_y))
            
            crop = image[y1:y2, x1:x2]
            if crop.size == 0 or crop.shape[0] == 0 or crop.shape[1] == 0:
                results[name] = np.zeros((size, size, 3), dtype=np.uint8)
            else:
                results[name] = cv2.resize(crop, (size, size), interpolation=cv2.INTER_LANCZOS4)
                
        return (
            results["left_periorbital"],
            results["right_periorbital"],
            results["nasolabial_left"],
            results["nasolabial_right"],
            results["hairline_band"],
            results["chin_jaw"]
        )

    def _select_sharpest_frame(self, frames: List[np.ndarray], trajectory: Dict[int, Tuple[int, int, int, int]]) -> Tuple[int, float]:
        valid_indices = list(trajectory.keys())
        if not valid_indices:
            return 0, 0.0
            
        num_samples = min(len(valid_indices), getattr(self.config.preprocessing, 'quality_snipe_samples', 5))
        sample_indices = [valid_indices[i] for i in np.linspace(0, len(valid_indices)-1, num_samples, dtype=int)]
        
        best_idx = valid_indices[0]
        # FIX 3: Initialize to prevent UnboundLocalError
        best_sharpness = 0.0
        
        for idx in sample_indices:
            frame = frames[idx]
            x1, y1, x2, y2 = trajectory[idx]
            
            h, w = frame.shape[:2]
            cx1, cy1 = max(0, x1), max(0, y1)
            cx2, cy2 = min(w, x2), min(h, y2)
            
            face_crop = frame[cy1:cy2, cx1:cx2]
            if face_crop.size == 0:
                continue
                
            # FIX 6: Ensure RGB before grayscale conversion
            gray = cv2.cvtColor(face_crop, cv2.COLOR_RGB2GRAY)
            sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            if sharpness > best_sharpness:
                best_sharpness = float(sharpness)
                best_idx = int(idx)
                
        return best_idx, max(0.0, best_sharpness)

    def process_media(self, path: Path) -> PreprocessResult:
        # Sanitize and validate
        path = Path(path).resolve()
        if not path.is_file():
            raise ValueError(f"Invalid file path: {path}")

        # Check extension whitelist
        valid_extensions = {'.jpg', '.jpeg', '.png', '.mp4', '.avi', '.mov', '.webp', '.mkv'}
        if path.suffix.lower() not in valid_extensions:
            raise ValueError(f"Unsupported file type: {path.suffix}")

        # Check file size (max 500MB)
        max_size = 500 * 1024 * 1024
        if path.stat().st_size > max_size:
            raise ValueError("File too large. Maximum size is 500MB")

        path_str = str(path)
        result = PreprocessResult(has_face=False)
        self.tracker = SortTracker()
        min_res = getattr(self.config.preprocessing, 'min_face_resolution', 64)
        
        try:
            if is_video_file(path_str):
                result.original_media_type = "video"
                frames = extract_frames(path_str, self.config.preprocessing.max_video_frames, self.config.preprocessing.extract_fps)
                if not frames:
                    return result
                result.frames_30fps = frames
                result.first_frame = frames[0]
                
                # FIX 7: Flag if video too short for rPPG
                if len(frames) < self.config.preprocessing.min_video_frames:
                    result.frame_count_warning = True
                
                established_tracks: Dict[int, TrackedFace] = {}
                
                # --- PHASE 1: BUILD TRAJECTORIES ---
                for i, frame in enumerate(frames):
                    lm_list = self._get_landmarks(frame)
                    dets = []
                    
                    if lm_list:
                        for lm in lm_list:
                            x_min, y_min = np.min(lm, axis=0)
                            x_max, y_max = np.max(lm, axis=0)
                            w = x_max - x_min
                            h = y_max - y_min
                            if w >= min_res and h >= min_res:
                                dets.append([x_min, y_min, x_max, y_max])
                                
                        if not dets and lm_list:
                            largest_lm = lm_list[0]
                            x_min, y_min = np.min(largest_lm, axis=0)
                            x_max, y_max = np.max(largest_lm, axis=0)
                            dets.append([x_min, y_min, x_max, y_max])
                            
                    dets = np.array(dets) if dets else np.empty((0, 4))
                    tracked_items = self.tracker.update(dets)
                    
                    for item in tracked_items:
                        x1, y1, x2, y2, trk_id = item
                        trk_id = int(trk_id)
                        if trk_id not in established_tracks:
                            established_tracks[trk_id] = TrackedFace(
                                identity_id=trk_id,
                                landmarks=np.zeros((478, 2)),
                                trajectory_bboxes={}
                            )
                        established_tracks[trk_id].trajectory_bboxes[i] = (int(x1), int(y1), int(x2), int(y2))
                
                # --- PHASE 2: EXTRACT CROPS PER-TRACK & HEURISTICS ---
                # Use eagerly-cached first_frame to avoid DiskBackedFrameList lazy-load race
                gray_first = cv2.cvtColor(result.first_frame, cv2.COLOR_RGB2GRAY)
                if gray_first.mean() < 50.0:
                    result.heuristic_flags.append("LOW_LIGHT")
                
                for trk_id, track_obj in established_tracks.items():
                    # FIX 4: Much more permissive — only discard tracks with < 3–5 detections
                    # The old threshold (min(15, frames//2)) was silently dropping valid face
                    # tracks that appear late or leave early in the video.
                    min_track_length = min(5, max(3, len(frames) // 30))
                    if len(track_obj.trajectory_bboxes) < min_track_length:
                        continue
                        
                    frames_present = sorted(list(track_obj.trajectory_bboxes.keys()))
                    best_window = []
                    current_window = [frames_present[0]]
                    for idx in range(1, len(frames_present)):
                        if frames_present[idx] == frames_present[idx-1] + 1:
                            current_window.append(frames_present[idx])
                        else:
                            if len(current_window) > len(best_window):
                                best_window = current_window
                            current_window = [frames_present[idx]]
                    if len(current_window) > len(best_window):
                        best_window = current_window
                        
                    if len(best_window) > 0:
                        track_obj.face_window = (best_window[0], best_window[-1] + 1)
                    else:
                        track_obj.face_window = (0, 0)
                        
                    if len(best_window) < len(frames_present) * 0.8:
                        track_obj.heuristic_flags.append("OCCLUSION")

                    avg_area_ratio = 0.0
                    avg_sharpness = 0.0
                    # Use eagerly-cached first_frame to avoid DiskBackedFrameList lazy-load race
                    frame_w, frame_h = result.first_frame.shape[1], result.first_frame.shape[0]
                    total_area = frame_w * frame_h
                    sampled = 0
                    
                    for f_idx in best_window[::max(1, len(best_window)//10)]: # sample ~10 frames
                        x1, y1, x2, y2 = track_obj.trajectory_bboxes[f_idx]
                        area = ((x2 - max(0,x1)) * (y2 - max(0,y1))) / float(total_area)
                        avg_area_ratio += area
                        
                        crop = frames[f_idx][max(0,y1):min(frame_h,y2), max(0,x1):min(frame_w,x2)]
                        if crop.size > 0:
                            gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
                            avg_sharpness += cv2.Laplacian(gray, cv2.CV_64F).var()
                        sampled += 1
                        
                    if sampled > 0:
                        avg_area_ratio /= sampled
                        avg_sharpness /= sampled
                        
                    if avg_area_ratio < 0.01:
                        track_obj.heuristic_flags.append("FACE_TOO_SMALL")
                    if avg_sharpness < 15.0:
                        track_obj.heuristic_flags.append("MOTION_BLUR")
                        
                    best_idx, best_sharpness = self._select_sharpest_frame(frames, track_obj.trajectory_bboxes)
                    target_image = frames[best_idx]
                    final_lms = self._get_landmarks(target_image)
                    
                    if final_lms is None:
                        fallback_idx = list(track_obj.trajectory_bboxes.keys())[len(track_obj.trajectory_bboxes)//2]
                        target_image = frames[fallback_idx]
                        final_lms = self._get_landmarks(target_image)
                        best_idx = fallback_idx 
                        
                    if final_lms is None:
                        continue
                        
                    trk_box = track_obj.trajectory_bboxes[best_idx]
                    best_iou = -1.0
                    matched_lm = final_lms[0]
                    
                    for lm in final_lms:
                        x_min, y_min = np.min(lm, axis=0)
                        x_max, y_max = np.max(lm, axis=0)
                        iou = compute_iou(trk_box, (x_min, y_min, x_max, y_max))
                        if iou > best_iou:
                            best_iou = iou
                            matched_lm = lm
                            
                    track_obj.best_frame_idx = best_idx
                    track_obj.landmarks = matched_lm
                    track_obj.face_crop_224 = self._crop_align(target_image, matched_lm, self.config.preprocessing.face_crop_size)
                    track_obj.face_crop_380 = self._crop_align(target_image, matched_lm, self.config.preprocessing.sbi_crop_size)
                    
                    patches = self._extract_native_patches(target_image, matched_lm)
                    track_obj.patch_left_periorbital = patches[0]
                    track_obj.patch_right_periorbital = patches[1]
                    track_obj.patch_nasolabial_left = patches[2]
                    track_obj.patch_nasolabial_right = patches[3]
                    track_obj.patch_hairline_band = patches[4]
                    track_obj.patch_chin_jaw = patches[5]
                    
                    result.tracked_faces.append(track_obj)
                    
                if len(result.tracked_faces) > 0:
                    result.has_face = True
                    result.selected_frame_index = result.tracked_faces[0].best_frame_idx
                    result.selected_frame_sharpness = best_sharpness
                    
                    # Calculate Gate Metrics
                    frame_w, frame_h = frames[0].shape[1], frames[0].shape[0]
                    total_area = frame_w * frame_h
                    max_area_ratio = 0.0
                    frames_with_face = set()
                    
                    for track in result.tracked_faces:
                        for f_idx, box in track.trajectory_bboxes.items():
                            frames_with_face.add(f_idx)
                            x1, y1, x2, y2 = box
                            area_ratio = ((x2 - x1) * (y2 - y1)) / total_area
                            if area_ratio > max_area_ratio:
                                max_area_ratio = area_ratio
                                
                    result.max_face_area_ratio = float(max_area_ratio)
                    result.frames_with_faces_pct = len(frames_with_face) / len(frames) if len(frames) > 0 else 0.0
                    
                    max_conf = 0.0
                    for track in result.tracked_faces:
                        conf = len(track.trajectory_bboxes) / len(frames) if len(frames) > 0 else 0.0
                        if conf > max_conf:
                            max_conf = conf
                    result.max_confidence = max_conf
                    
                    # Aggregate flags and check temporal insufficiency
                    best_track = max(result.tracked_faces, key=lambda t: t.face_window[1] - t.face_window[0], default=None)
                    if best_track:
                        result.heuristic_flags.extend(best_track.heuristic_flags)
                        result.heuristic_flags = list(set(result.heuristic_flags)) # dedup
                        max_len = best_track.face_window[1] - best_track.face_window[0]
                        if max_len < 90:
                            result.insufficient_temporal_data = True
                    else:
                        result.insufficient_temporal_data = True
                else:
                    result.insufficient_temporal_data = True
                
            elif is_image(path_str):
                result.original_media_type = "image"
                image = load_image(path)
                result.frames_30fps = [image]
                result.first_frame = image
                
                final_landmarks_list = self._get_landmarks(image)
                if final_landmarks_list is None:
                    return result
                    
                result.has_face = True
                for i, lm in enumerate(final_landmarks_list):
                    x_min, y_min = np.min(lm, axis=0)
                    x_max, y_max = np.max(lm, axis=0)
                    
                    track_obj = TrackedFace(
                        identity_id=i+1,
                        landmarks=lm,
                        trajectory_bboxes={0: (int(x_min), int(y_min), int(x_max), int(y_max))},
                        best_frame_idx=0
                    )
                    
                    track_obj.face_crop_224 = self._crop_align(image, lm, self.config.preprocessing.face_crop_size)
                    track_obj.face_crop_380 = self._crop_align(image, lm, self.config.preprocessing.sbi_crop_size)
                    
                    patches = self._extract_native_patches(image, lm)
                    track_obj.patch_left_periorbital = patches[0]
                    track_obj.patch_right_periorbital = patches[1]
                    track_obj.patch_nasolabial_left = patches[2]
                    track_obj.patch_nasolabial_right = patches[3]
                    track_obj.patch_hairline_band = patches[4]
                    track_obj.patch_chin_jaw = patches[5]
                    
                    result.tracked_faces.append(track_obj)
                    
                    h, w = image.shape[:2]
                    area_ratio = ((x_max - x_min) * (y_max - y_min)) / (w * h)
                    result.max_face_area_ratio = max(result.max_face_area_ratio, float(area_ratio))
                
                result.frames_with_faces_pct = 1.0
                result.max_confidence = 1.0
            
            return result
            
        except Exception as e:
            logger.error(f"Preprocessing failed for {path_str}: {e}", exc_info=True)
            # CRITICAL: Return the partially-populated result (which still has first_frame
            # and frames_30fps set) so GPU tools have image data even if face tracking crashed.
            # Only fall back to a blank result if we have no frame data at all.
            if result.first_frame is not None or result.frames_30fps:
                logger.warning("Returning partial preprocessing result — face tracking failed but frame data is available.")
                return result
            return PreprocessResult(has_face=False)