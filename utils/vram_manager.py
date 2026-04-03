"""VRAM Lifecycle Management.

Provides deterministic memory lifecycle management for hardware-accelerated tools.
Enforces absolute device purging via class-level locking and strict garbage collection
to stay within Aegis-X's 4GB VRAM ceiling (reserving ~3GB for models, ~1GB for context).

⚙️ HARDWARE THRESHOLDS — Synchronized with utils/video.py:
    Supports: TPU (Colab) → CUDA → MPS → CPU (priority order)
    
    TPU Note: Colab TPU v2/v3 has 8GB HBM, v4 has 32GB. Memory management differs from CUDA.
"""

from utils.thresholds import (
    VRAM_MODEL_LOAD_THRESHOLD,
    VRAM_RESERVED_BUFFER_GB,
    TPU_MEMORY_FRACTION,
    ENABLE_TPU_SUPPORT,
    VRAM_MIN_FOR_GPU,
    VRAM_RECOMMENDED
)

import threading
import gc
import time
import torch
from typing import Callable, Any, Optional

from utils.logger import setup_logger

logger = setup_logger(__name__)


def get_device() -> torch.device:
    """Auto-detects the best available accelerator hardware sequentially.
    
    Priority: TPU → CUDA → MPS → CPU
    
    TPU Detection:
        - Requires torch_xla package (pip install torch_xla)
        - Colab TPU: Runtime → Change runtime type → TPU
        - Verifies actual TPU backend (not XLA-CPU fallback)
    """
    # 1. TPU (Google Colab priority)
    if ENABLE_TPU_SUPPORT:
        try:
            import torch_xla.core.xla_model as xm
            
            # Fix: Explicitly verify a TPU is backing the XLA backend
            supported_devices = xm.get_xla_supported_devices()
            
            if supported_devices:
                # Check if any device is actually a TPU (not XLA-CPU fallback)
                tpu_devices = [d for d in supported_devices if "TPU" in d.upper()]
                
                if tpu_devices:
                    device = xm.xla_device()
                    logger.info(f"TPU detected: {tpu_devices[0]}. Using XLA accelerator.")
                    return device
                else:
                    logger.warning(
                        f"XLA available but no TPU found. Devices: {supported_devices}. "
                        f"Falling back to CUDA/CPU."
                    )
        except ImportError:
            pass  # torch_xla not installed
        except Exception as e:
            logger.warning(f"TPU detection failed: {e}. Falling back to CUDA/CPU.")

    # 2. CUDA (Standard GPU)
    if torch.cuda.is_available():
        # Force strict device pinning for VRAM safety
        device_id = 0
        device = torch.device(f"cuda:{device_id}")
        logger.info(f"CUDA detected: {torch.cuda.get_device_name(device_id)}. Using GPU.")
        return device

    # 3. MPS (Apple Silicon)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        logger.info("MPS detected (Apple Silicon). Using GPU.")
        return torch.device("mps")

    # 4. CPU (Universal fallback)
    logger.info("No accelerator found. Using CPU.")
    return torch.device("cpu")

def log_vram_status(tag: str, device_id: int = 0) -> None:
    """Log current VRAM usage for diagnostics."""
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info(device_id)
        free_gb, total_gb = free / (1024**3), total / (1024**3)
        used_gb = total_gb - free_gb
        logger.debug(f"[{tag}] VRAM: {used_gb:.2f} GB used / {free_gb:.2f} GB free / {total_gb:.2f} GB total (cuda:{device_id})")


def _get_available_vram_gb() -> float:
    """Safely detect total VRAM/HBM in GB. Returns 0.0 if no accelerator."""
    # TPU Memory Detection
    if ENABLE_TPU_SUPPORT:
        try:
            import torch_xla.core.xla_model as xm
            
            # TPU memory info via XLA
            device = xm.xla_device()
            memory_info = xm.get_memory_info(device)
            
            if memory_info and "bytes_limit" in memory_info:
                total_bytes = memory_info["bytes_limit"]
                total_gb = total_bytes / (1024 ** 3)
                logger.debug(f"TPU memory detected: {total_gb:.1f}GB")
                return total_gb
        except Exception:
            pass
    
    # CUDA Memory Detection
    if torch.cuda.is_available():
        try:
            props = torch.cuda.get_device_properties(0)
            return props.total_memory / (1024 ** 3)
        except Exception:
            return 0.0
    
    return 0.0


def _get_used_memory_gb() -> float:
    """Get currently allocated memory in GB (device-specific)."""
    device = get_device()
    
    # TPU Memory Usage
    if device.type == "xla":
        try:
            import torch_xla.core.xla_model as xm
            
            memory_info = xm.get_memory_info(device)
            if memory_info and "bytes_used" in memory_info:
                return memory_info["bytes_used"] / (1024 ** 3)
        except Exception:
            pass
        return 0.0
    
    # CUDA Memory Usage
    if device.type == "cuda":
        try:
            return torch.cuda.memory_allocated(0) / (1024 ** 3)
        except Exception:
            pass
        return 0.0
    
    return 0.0


def _check_available_vram(required_gb: float = 2.0) -> bool:
    """
    Check if sufficient memory is available for model loading.
    
    Args:
        required_gb: Estimated memory needed for the model (default: 2.0GB)
    
    Returns:
        bool: True if sufficient memory available, False otherwise
    """
    device = get_device()
    
    # CPU loading always OK
    if device.type == "cpu":
        return True
    
    # TPU Memory Check
    if device.type == "xla":
        try:
            import torch_xla.core.xla_model as xm
            
            total_gb = _get_available_vram_gb()
            
            if total_gb < VRAM_MODEL_LOAD_THRESHOLD:
                logger.warning(
                    f"Total TPU memory {total_gb:.1f}GB < threshold {VRAM_MODEL_LOAD_THRESHOLD}GB. "
                    f"Forcing CPU model loading."
                )
                return False
            
            # Check available (not just total)
            memory_info = xm.get_memory_info(device)
            if memory_info:
                bytes_limit = memory_info.get("bytes_limit", 0)
                bytes_used = memory_info.get("bytes_used", 0)
                bytes_free = bytes_limit - bytes_used
                
                # Apply TPU_MEMORY_FRACTION safety margin
                available_gb = (bytes_free / (1024 ** 3)) * TPU_MEMORY_FRACTION
                
                if available_gb < required_gb:
                    logger.warning(
                        f"Insufficient TPU memory: {available_gb:.1f}GB available, "
                        f"{required_gb:.1f}GB required. Forcing CPU."
                    )
                    return False
            
            return True
            
        except Exception as e:
            logger.warning(f"TPU memory check failed: {e}. Proceeding with caution.")
            return True
    
    # CUDA Memory Check
    if device.type == "cuda":
        total_vram = _get_available_vram_gb()
        
        if total_vram < VRAM_MODEL_LOAD_THRESHOLD:
            logger.warning(
                f"Total VRAM {total_vram:.1f}GB < threshold {VRAM_MODEL_LOAD_THRESHOLD}GB. "
                f"Forcing CPU model loading."
            )
            return False
        
        try:
            allocated_gb = torch.cuda.memory_allocated(0) / (1024 ** 3)
            reserved_gb = torch.cuda.memory_reserved(0) / (1024 ** 3)
            free_gb = total_vram - reserved_gb - VRAM_RESERVED_BUFFER_GB
            
            if free_gb < required_gb:
                logger.warning(
                    f"Insufficient free VRAM: {free_gb:.1f}GB available, "
                    f"{required_gb:.1f}GB required. Forcing CPU."
                )
                return False
        except Exception:
            pass
        
        return True
    
    return True


def _cleanup_device_memory(device: torch.device) -> None:
    """
    Device-specific memory cleanup.
    
    TPU: Uses xm.mark_step() + memory defragmentation
    CUDA: Uses torch.cuda.empty_cache()
    MPS: Uses torch.mps.empty_cache()
    """
    if device.type == "xla":
        # TPU-specific cleanup
        try:
            import torch_xla.core.xla_model as xm
            
            # Mark step to synchronize and free intermediate buffers
            xm.mark_step()
            
            # Force garbage collection
            gc.collect()
            
            logger.debug("TPU memory cleanup completed (xm.mark_step)")
        except Exception as e:
            logger.warning(f"TPU cleanup failed: {e}")
    
    elif device.type == "cuda":
        # CUDA-specific cleanup
        try:
            torch.cuda.empty_cache()
            gc.collect()
            logger.debug("CUDA memory cleanup completed (empty_cache)")
        except Exception as e:
            logger.warning(f"CUDA cleanup failed: {e}")
    
    elif device.type == "mps":
        # MPS-specific cleanup
        try:
            torch.mps.empty_cache()
            gc.collect()
            logger.debug("MPS memory cleanup completed (empty_cache)")
        except Exception as e:
            logger.warning(f"MPS cleanup failed: {e}")
    
    else:
        # CPU: Just garbage collection
        gc.collect()


class VRAMLifecycleManager:
    """Context manager that loads, yields, and ruthlessly purges models from accelerator memory.
    
    Usage:
        with VRAMLifecycleManager(load_clip_adapter, model_name="CLIP_Adapter") as model:
            with torch.no_grad():
                result = model(input_tensor)
    
    TPU Support:
        - Automatically detects TPU via torch_xla
        - Uses xm.mark_step() for memory synchronization
        - Falls back to CPU if TPU memory insufficient
    
    Thread Safety:
        Uses RLock with timeout to prevent deadlocks if a tool crashes hard.
        Only ONE accelerator model may be resident at a time (Spec Section 3).
    """
    
    # Class-level global lock to prevent concurrent requests from crashing the accelerator
    _accelerator_lock = threading.RLock()
    
    # Lock timeout to prevent permanent deadlock on hard crashes
    _LOCK_TIMEOUT_SECONDS = 30.0
    
    def __init__(self, model_loader_function, model_name="Unknown",
             required_vram_gb=2.0, loader_args=None, loader_kwargs=None):
        self.model_loader_function = model_loader_function
        self.model_name = model_name
        self.required_vram_gb = required_vram_gb
        self.args = loader_args or ()
        self.kwargs = loader_kwargs or {}
        self.device = get_device()
        self.model = None
        self._lock_acquired = False

    def __enter__(self) -> torch.nn.Module:
        # Acquire the global lock with timeout to prevent permanent deadlock
        start_time = time.time()
        acquired = self.__class__._accelerator_lock.acquire(
            timeout=self.__class__._LOCK_TIMEOUT_SECONDS
        )
        
        if not acquired:
            elapsed = time.time() - start_time
            raise RuntimeError(
                f"Failed to acquire accelerator lock after {elapsed:.1f}s. "
                f"Another tool may be stuck. Model: {self.model_name}"
            )
        
        self._lock_acquired = True
        
        # Check memory before attempting load
        use_cpu_fallback = False
        if self.device.type in ("cuda", "xla", "mps"):
            if not _check_available_vram(self.required_vram_gb):
                use_cpu_fallback = True
                self.device = torch.device("cpu")
                logger.warning(
                    f"Model '{self.model_name}': Insufficient accelerator memory. "
                    f"Falling back to CPU inference."
                )
        
        try:
            logger.info(
                f"Loading model '{self.model_name}' on {self.device.type.upper()}..."
            )
            
            # Execute the model loader function
            self.model = self.model_loader_function(*self.args, **self.kwargs)
            
            # Safely move model to device and set to eval
            if hasattr(self.model, "to"):
                self.model.to(self.device)
            if hasattr(self.model, "eval"):
                self.model.eval()
            
            # Log memory usage after load for debugging
            if self.device.type == "cuda":
                try:
                    allocated_mb = torch.cuda.memory_allocated(0) / (1024 ** 2)
                    logger.info(
                        f"Model '{self.model_name}' loaded. CUDA allocated: {allocated_mb:.0f}MB"
                    )
                except Exception:
                    pass
            
            elif self.device.type == "xla":
                try:
                    import torch_xla.core.xla_model as xm
                    
                    memory_info = xm.get_memory_info(self.device)
                    if memory_info and "bytes_used" in memory_info:
                        used_mb = memory_info["bytes_used"] / (1024 ** 2)
                        logger.info(
                            f"Model '{self.model_name}' loaded. TPU memory used: {used_mb:.0f}MB"
                        )
                except Exception:
                    pass
                
            return self.model
            
        except Exception as e:
            logger.error(
                f"Model '{self.model_name}' failed to load: {e}",
                exc_info=True
            )
            self._safe_cleanup()
            raise

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        # Log exception info for debugging
        if exc_type is not None:
            logger.warning(
                f"Model '{self.model_name}' context exited with exception: "
                f"{exc_type.__name__}: {exc_val}"
            )
        
        # Perform cleanup
        self._safe_cleanup()

    def _safe_cleanup(self) -> None:
        """
        EXACT INSTRUCTIONS MUST BE FOLLOWED IN THIS ORDER.
        Extracted to separate method for reuse in __enter__ error handling.
        """
        try:
            # 1. Unbind the model object from memory
            if self.model is not None:
                # Force the model's parameters to CPU to instantly drop accelerator memory
                if hasattr(self.model, "to"):
                    try:
                        self.model.to("cpu")
                    except Exception:
                        pass  # Ignore if this specific object rejects .to("cpu")
                        
                # Wait for pending async CUDA ops before dropping reference
                if torch.cuda.is_available():
                    try:
                        torch.cuda.synchronize()
                    except Exception:
                        pass
                
                del self.model
            self.model = None

            # 2. Device-specific memory     cleanup (TPU/CUDA/MPS)
            _cleanup_device_memory(self.device)
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
            
        finally:
            # 3. Release the global lock, no matter what crashed above
            if self._lock_acquired:
                self.__class__._accelerator_lock.release()
                self._lock_acquired = False


def run_with_vram_cleanup(
    model_loader: Callable,
    inference_fn: Callable,
    model_name: str = "Unknown",
    required_vram_gb: float = 2.0,
    *loader_args: Any,
    **loader_kwargs: Any
) -> Any:
    """
    Convenience function that wraps model loading + inference + cleanup in one call.
    
    Usage:
        result = run_with_vram_cleanup(
            load_clip_adapter,
            lambda model: model(input_tensor),
            model_name="CLIP_Adapter",
            required_vram_gb=0.6
        )
    
    This ensures torch.no_grad() is always applied and cleanup always runs.
    
    TPU Note:
        - Automatically uses TPU if available on Colab
        - xm.mark_step() called during cleanup for memory defragmentation
    """
    with VRAMLifecycleManager(
        model_loader,
        model_name=model_name,
        required_vram_gb=required_vram_gb,
        *loader_args,
        **loader_kwargs
    ) as model:
        with torch.no_grad():  # Ensure no gradient accumulation
            return inference_fn(model)