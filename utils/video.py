"""Video I/O utilities for Aegis-X.

Provides robust, memory-safe functions for extracting video frames using hardware
acceleration (NVDEC via torchcodec) when available, falling back to OpenCV.

⚙️ HARDWARE THRESHOLDS — Adjust based on your GPU VRAM:
    - Google Colab T4 (16GB): VRAM_GPU_DECODE_THRESHOLD = 12.0, GPU_DECODE_BATCH_SIZE = 32
    - Consumer RTX 3060 (12GB): VRAM_GPU_DECODE_THRESHOLD = 10.0, GPU_DECODE_BATCH_SIZE = 16
    - Consumer RTX 3050 (8GB):  VRAM_GPU_DECODE_THRESHOLD = 6.0,  GPU_DECODE_BATCH_SIZE = 8
    - Low-End GTX 1650 (4GB):  VRAM_GPU_DECODE_THRESHOLD = 4.0,  GPU_DECODE_BATCH_SIZE = 4
"""

from utils.thresholds import (
    VRAM_MODEL_LOAD_THRESHOLD,
    GPU_DECODE_BATCH_SIZE,
    CPU_DECODE_BATCH_SIZE,
    MAX_FRAME_DIMENSION,
    FALLBACK_FPS
)

import math
import cv2
import numpy as np
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple, Iterable

from utils.logger import setup_logger

class DiskBackedFrameList:
    """Disk-backed array cache to prevent OOM on long video processing."""
    def __init__(self):
        self._temp_dir = tempfile.TemporaryDirectory()
        self._length = 0
        self._paths = []

    def append(self, frame: np.ndarray):
        path = f"{self._temp_dir.name}/frame_{self._length}.npy"
        np.save(path, frame)
        self._paths.append(path)
        self._length += 1

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return [np.load(self._paths[i]) for i in range(*idx.indices(self._length))]
        return np.load(self._paths[idx])

    def __len__(self):
        return self._length
        
    def __iter__(self):
        for path in self._paths:
            yield np.load(path)
            
    def cleanup(self):
        try:
            self._temp_dir.cleanup()
        except Exception:
            pass

    def __del__(self):
        self.cleanup()

logger = setup_logger(__name__)

# Attempt to import torchcodec and torch for hardware-accelerated video decoding
TORCHCODEC_AVAILABLE = False
try:
    import torch
    from torchcodec.decoders import VideoDecoder
    TORCHCODEC_AVAILABLE = True
except Exception as e:
    logger.warning(f"Failed to load torchcodec: {e}. Falling back to OpenCV CPU decode.")
    # torch might still be available even if torchcodec fails
    import torch
    TORCHCODEC_AVAILABLE = False

# Known valid video extensions
VALID_VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".flv"}

def is_video_file(path: str) -> bool:
    """Checks if the given file path has a valid video extension."""
    return Path(path).suffix.lower() in VALID_VIDEO_EXTENSIONS

def _get_available_vram_gb() -> float:
    """Safely detect total VRAM in GB. Returns 0.0 if CUDA unavailable."""
    if not torch or not torch.cuda.is_available():
        return 0.0
    try:
        props = torch.cuda.get_device_properties(0)
        return props.total_memory / (1024 ** 3)
    except Exception:
        return 0.0

def _calculate_scale(width: int, height: int, max_width: int = 1280) -> Optional[Tuple[int, int]]:
    """Calculates dimensions to downscale a frame preserving aspect ratio."""
    if MAX_FRAME_DIMENSION <= 0 or width <= MAX_FRAME_DIMENSION:
        return None
    scale = MAX_FRAME_DIMENSION / width
    new_width = int(width * scale)
    new_height = int(height * scale)
    return (new_width, new_height)

def extract_frames(video_path: str, max_frames: int = 300, target_fps: int = 30) -> List[np.ndarray]:
    """Safely extracts a sequence of RGB frames from a video file.
    
    VRAM SAFETY: Uses VRAM_GPU_DECODE_THRESHOLD to decide GPU vs CPU decoding.
    """
    if not Path(video_path).exists():
        logger.error(f"Video file not found: {video_path}")
        return []

    # --- VRAM-Aware Device Selection ---
    use_gpu_decode = False
    if TORCHCODEC_AVAILABLE and torch and torch.cuda.is_available():
        total_vram_gb = _get_available_vram_gb()
        if total_vram_gb >= VRAM_MODEL_LOAD_THRESHOLD:
            use_gpu_decode = True
            logger.info(f"VRAM detected: {total_vram_gb:.1f}GB (threshold: {VRAM_MODEL_LOAD_THRESHOLD}GB). Using GPU decoding (NVDEC).")
        else:
            logger.warning(f"VRAM detected: {total_vram_gb:.1f}GB (threshold: {VRAM_MODEL_LOAD_THRESHOLD}GB). Forcing CPU decoding to reserve VRAM for models.")
    elif TORCHCODEC_AVAILABLE:
        logger.info("TorchCodec available but CUDA not found. Using CPU decoding.")

    # Fast path: TorchCodec Decoding (GPU or CPU)
    if TORCHCODEC_AVAILABLE:
        try:
            device = "cuda" if use_gpu_decode else "cpu"
            decoder = VideoDecoder(video_path, device=device)
            
            # Fetch metadata with safe null-checks
            total_frames = decoder.metadata.num_frames
            source_fps = decoder.metadata.average_fps
            
            if total_frames is None or source_fps is None or total_frames <= 0 or source_fps <= 0:
                 raise ValueError("Invalid metadata returned by TorchCodec.")
                 
            total_frames = int(total_frames)
            source_fps = float(source_fps)
                 
            # Compute temporal indices
            fps_ratio = source_fps / target_fps
            skip_interval = max(1.0, fps_ratio)
            
            # Generate frame indices to extract
            indices = []
            current_index = 0.0
            while current_index < total_frames and len(indices) < max_frames:
                idx = int(round(current_index))
                if idx < total_frames:
                    indices.append(idx)
                current_index += skip_interval
                
            if not indices:
                return DiskBackedFrameList()
                
            # Batch extraction
            frames_list = DiskBackedFrameList()
            batch_size = GPU_DECODE_BATCH_SIZE if use_gpu_decode else CPU_DECODE_BATCH_SIZE
            
            for i in range(0, len(indices), batch_size):
                batch_indices = indices[i:i + batch_size]
                
                # Returns a PyTorch tensor usually shape (N, C, H, W)
                frames_tensor = decoder.get_frames_at(indices=batch_indices).data
                
                if use_gpu_decode:
                    # Explicitly move to CPU and free GPU memory immediately
                    frames_np = frames_tensor.cpu().numpy()
                    del frames_tensor
                    if torch and torch.cuda.is_available():
                        torch.cuda.empty_cache()
                else:
                    frames_np = frames_tensor.numpy()
                
                # Torchcodec defaults to NCHW. Let's strictly permute.
                if frames_np.ndim == 4 and frames_np.shape[1] == 3:
                    frames_np = np.transpose(frames_np, (0, 2, 3, 1))

                for frame in frames_np:
                    h, w, c = frame.shape
                    
                    # Optional downscaling (disabled by default for forensic accuracy)
                    scale_dims = _calculate_scale(w, h)
                    if scale_dims:
                        frame = cv2.resize(frame, scale_dims, interpolation=cv2.INTER_AREA)
                        logger.warning(f"Frame downscaled to {scale_dims} — may reduce forensic accuracy.")
                    
                    # Enforce uint8 dtype (TorchCodec may return float [0,1])
                    if frame.dtype != np.uint8:
                        if frame.max() <= 1.0:
                            frame = (frame * 255.0)
                        frame = np.round(frame).clip(0, 255).astype(np.uint8)
                        
                    frames_list.append(frame)
                
            # Final cleanup
            del decoder
            if use_gpu_decode and torch and torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return frames_list
        except Exception as e:
            logger.warning(f"TorchCodec extraction failed: {e}. Falling back to OpenCV CPU decode.")

    # Fallback path: OpenCV CPU Decoding
    return _extract_cv2(video_path, max_frames, target_fps)

def _extract_cv2(video_path: str, max_frames: int, target_fps: int) -> List[np.ndarray]:
    """Fallback frame extraction method utilizing traditional OpenCV."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Could not open video file via cv2: {video_path}")
        return []
        
    source_fps = cap.get(cv2.CAP_PROP_FPS)
    if source_fps <= 0 or math.isnan(source_fps):
         source_fps = FALLBACK_FPS
         logger.warning(f"OpenCV returned invalid FPS ({source_fps}), using fallback {FALLBACK_FPS}")

    fps_ratio = source_fps / target_fps
    skip_interval = max(1.0, fps_ratio)
    
    frames_list = DiskBackedFrameList()
    frame_idx = 0
    current_target = 0.0
    
    while cap.isOpened() and len(frames_list) < max_frames:
        ret, frame_bgr = cap.read()
        if not ret:
            break
            
        if frame_idx >= int(round(current_target)):
            h, w = frame_bgr.shape[:2]
            
            # Optional downscaling (disabled by default for forensic accuracy)
            scale_dims = _calculate_scale(w, h)
            if scale_dims:
                frame_bgr = cv2.resize(frame_bgr, scale_dims, interpolation=cv2.INTER_AREA)
                logger.warning(f"Frame downscaled to {scale_dims} — may reduce forensic accuracy.")
                
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            
            if frame_rgb.dtype != np.uint8:
                frame_rgb = np.round(frame_rgb).clip(0, 255).astype(np.uint8)
                
            frames_list.append(frame_rgb)
            current_target += skip_interval
            
        frame_idx += 1
        
    cap.release()
    return frames_list

def get_video_duration(path: Path) -> float:
    """Gets the duration of the video file in seconds."""
    video_path_str = str(path)
    
    if TORCHCODEC_AVAILABLE:
        try:
            use_gpu = False
            if torch and torch.cuda.is_available():
                vram = _get_available_vram_gb()
                if vram >= VRAM_MODEL_LOAD_THRESHOLD:
                    use_gpu = True
            
            device = "cuda" if use_gpu else "cpu"
            decoder = VideoDecoder(video_path_str, device=device)
            
            duration = decoder.metadata.duration_seconds
            if duration is not None and duration > 0:
                return float(duration)
                
            num_frames = decoder.metadata.num_frames
            fps = decoder.metadata.average_fps
            if num_frames and fps and float(fps) > 0:
                return float(int(num_frames) / float(fps))
        except Exception:
            pass
            
    # OpenCV Fallback
    cap = cv2.VideoCapture(video_path_str)
    if not cap.isOpened():
        return 0.0
        
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    
    if fps > 0:
        return float(frame_count / fps)
        
    return 0.0