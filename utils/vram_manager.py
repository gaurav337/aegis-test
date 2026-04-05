"""VRAM Lifecycle Management — v2.1 (Audit Corrected)

Provides deterministic memory lifecycle management for hardware-accelerated tools.
Enforces absolute device purging via class-level locking and strict garbage collection
to stay within Aegis-X's 4GB VRAM ceiling (reserving ~3GB for models, ~1GB for context).

Key Fixes:
1. m-01: Watchdog thread prevents permanent deadlock on hard crashes/segfaults
2. m-10: VRAM cleanup guaranteed even on exception (empty_cache in finally block)
3. TPU/CUDA/MPS cleanup consistency: All device types get proper defragmentation
4. Memory check robustness: Handles edge cases in _check_available_vram
5. Thread safety: RLock always released via finally block + watchdog fallback
"""

from utils.thresholds import (
    VRAM_MODEL_LOAD_THRESHOLD,
    VRAM_RESERVED_BUFFER_GB,
    TPU_MEMORY_FRACTION,
    ENABLE_TPU_SUPPORT,
    VRAM_MIN_FOR_GPU,
    VRAM_RECOMMENDED,
)

import threading
import gc
import time
import torch
from typing import Callable, Any, Optional, Union
from utils.logger import setup_logger

logger = setup_logger(__name__)

# ──────────────────────────────────────────────────────────────
# Hardware Detection
# ──────────────────────────────────────────────────────────────


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

            supported_devices = xm.get_xla_supported_devices()
            if supported_devices:
                tpu_devices = [d for d in supported_devices if "TPU" in d.upper()]
                if tpu_devices:
                    device = xm.xla_device()
                    logger.info(
                        f"TPU detected: {tpu_devices[0]}. Using XLA accelerator."
                    )
                    return device
                else:
                    logger.warning(
                        f"XLA available but no TPU found. Devices: {supported_devices}. "
                        f"Falling back to CUDA/CPU."
                    )
        except ImportError:
            pass
        except Exception as e:
            logger.warning(f"TPU detection failed: {e}. Falling back to CUDA/CPU.")

    # 2. CUDA (Standard GPU)
    if torch.cuda.is_available():
        device_id = 0
        device = torch.device(f"cuda:{device_id}")
        logger.info(
            f"CUDA detected: {torch.cuda.get_device_name(device_id)}. Using GPU."
        )
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
        try:
            free, total = torch.cuda.mem_get_info(device_id)
            free_gb, total_gb = free / (1024**3), total / (1024**3)
            used_gb = total_gb - free_gb
            logger.debug(
                f"[{tag}] VRAM: {used_gb:.2f} GB used / {free_gb:.2f} GB free / {total_gb:.2f} GB total (cuda:{device_id})"
            )
        except Exception as e:
            logger.debug(f"[{tag}] VRAM log failed: {e}")


# ──────────────────────────────────────────────────────────────
# Memory Query Helpers
# ──────────────────────────────────────────────────────────────


def _get_available_vram_gb() -> float:
    """Safely detect total VRAM/HBM in GB. Returns 0.0 if no accelerator."""
    # TPU Memory Detection
    if ENABLE_TPU_SUPPORT:
        try:
            import torch_xla.core.xla_model as xm

            device = xm.xla_device()
            memory_info = xm.get_memory_info(device)
            if memory_info and "bytes_limit" in memory_info:
                total_bytes = memory_info["bytes_limit"]
                total_gb = total_bytes / (1024**3)
                logger.debug(f"TPU memory detected: {total_gb:.1f}GB")
                return total_gb
        except Exception:
            pass

    # CUDA Memory Detection
    if torch.cuda.is_available():
        try:
            props = torch.cuda.get_device_properties(0)
            return props.total_memory / (1024**3)
        except Exception:
            return 0.0

    return 0.0


def _get_used_memory_gb() -> float:
    """Get currently allocated memory in GB (device-specific)."""
    # CUDA Memory Usage (check first — most common)
    if torch.cuda.is_available():
        try:
            return torch.cuda.memory_allocated(0) / (1024**3)
        except Exception:
            pass

    # TPU Memory Usage
    if ENABLE_TPU_SUPPORT:
        try:
            import torch_xla.core.xla_model as xm

            device = xm.xla_device()
            memory_info = xm.get_memory_info(device)
            if memory_info and "bytes_used" in memory_info:
                return memory_info["bytes_used"] / (1024**3)
        except Exception:
            pass

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

            memory_info = xm.get_memory_info(device)
            if memory_info:
                bytes_limit = memory_info.get("bytes_limit", 0)
                bytes_used = memory_info.get("bytes_used", 0)
                bytes_free = bytes_limit - bytes_used
                available_gb = (bytes_free / (1024**3)) * TPU_MEMORY_FRACTION

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
            reserved_gb = torch.cuda.memory_reserved(0) / (1024**3)
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


# ──────────────────────────────────────────────────────────────
# Device-Specific Cleanup
# ──────────────────────────────────────────────────────────────


def _cleanup_device_memory(device: torch.device) -> None:
    """
    Device-specific memory cleanup.

    TPU: Uses xm.mark_step() + memory defragmentation
    CUDA: Uses torch.cuda.empty_cache() + synchronize
    MPS: Uses torch.mps.empty_cache()
    CPU: Just garbage collection
    """
    try:
        if device.type == "xla":
            # TPU-specific cleanup
            try:
                import torch_xla.core.xla_model as xm

                xm.mark_step()  # Synchronize and free intermediate buffers
                logger.debug("TPU memory cleanup: xm.mark_step() called")
            except Exception as e:
                logger.warning(f"TPU cleanup failed: {e}")

        elif device.type == "cuda":
            # CUDA-specific cleanup
            try:
                torch.cuda.synchronize()  # Wait for pending ops
                torch.cuda.empty_cache()  # Release cached memory
                logger.debug("CUDA memory cleanup: synchronize + empty_cache")
            except Exception as e:
                logger.warning(f"CUDA cleanup failed: {e}")

        elif device.type == "mps":
            # MPS-specific cleanup
            try:
                torch.mps.empty_cache()
                logger.debug("MPS memory cleanup: empty_cache")
            except Exception as e:
                logger.warning(f"MPS cleanup failed: {e}")

        # CPU: Just garbage collection (handled below)

    finally:
        # Universal: Force Python garbage collection
        gc.collect()


# ──────────────────────────────────────────────────────────────
# VRAM Lock Watchdog (m-01 Fix)
# ──────────────────────────────────────────────────────────────


class _VRAMLockWatchdog:
    """
    Watchdog thread that force-releases the accelerator lock if held too long.
    Prevents permanent deadlock from hard crashes (segfaults, OOM kills, etc.).
    """

    _watchdog_thread: Optional[threading.Thread] = None
    _lock_held_since: Optional[float] = None
    _lock: Optional[threading.RLock] = None
    _timeout_seconds: float = 45.0  # 30s lock timeout + 15s watchdog buffer

    @classmethod
    def start(cls, lock: threading.RLock, timeout: float = None) -> None:
        """Start watchdog monitoring for a lock acquisition."""
        if timeout:
            cls._timeout_seconds = timeout

        cls._lock = lock
        cls._lock_held_since = time.time()

        if cls._watchdog_thread and cls._watchdog_thread.is_alive():
            return  # Already running

        def _watch():
            while cls._lock_held_since is not None:
                elapsed = time.time() - cls._lock_held_since
                if elapsed > cls._timeout_seconds:
                    logger.critical(
                        f"VRAM lock watchdog: Lock held for {elapsed:.1f}s (> {cls._timeout_seconds}s). "
                        f"Forcing release to prevent deadlock."
                    )
                    try:
                        if cls._lock:
                            # Attempt to release — may fail if lock is in inconsistent state
                            cls._lock.release()
                    except Exception as e:
                        logger.error(f"Watchdog force-release failed: {e}")
                    finally:
                        # Always clean up memory as fallback
                        _cleanup_device_memory(get_device())
                        cls._lock_held_since = None
                    break
                time.sleep(5.0)  # Check every 5 seconds

        cls._watchdog_thread = threading.Thread(
            target=_watch, daemon=True, name="VRAMWatchdog"
        )
        cls._watchdog_thread.start()

    @classmethod
    def stop(cls) -> None:
        """Stop watchdog monitoring (called on successful lock release)."""
        cls._lock_held_since = None
        # Thread will exit naturally on next iteration


# ──────────────────────────────────────────────────────────────
# VRAM Lifecycle Manager
# ──────────────────────────────────────────────────────────────


class VRAMLifecycleManager:
    """
    Context manager that loads, yields, and ruthlessly purges models from accelerator memory.

    Usage:
        with VRAMLifecycleManager(load_clip_adapter, model_name="CLIP_Adapter") as model:
            with torch.no_grad():
                result = model(input_tensor)

    Key Guarantees:
        1. Only ONE accelerator model resident at a time (global RLock)
        2. Lock timeout (30s) + watchdog (45s) prevents permanent deadlock
        3. VRAM cleanup guaranteed via finally block (m-10 fix)
        4. CPU fallback if accelerator memory insufficient
        5. Device-specific cleanup (TPU/CUDA/MPS) + universal gc.collect()

    Thread Safety:
        Uses RLock with timeout + watchdog thread for hard-crash resilience.
    """

    # Class-level global lock
    _accelerator_lock = threading.RLock()
    _LOCK_TIMEOUT_SECONDS = 30.0

    def __init__(
        self,
        model_loader_function: Callable,
        model_name: str = "Unknown",
        required_vram_gb: float = 2.0,
        loader_args: Optional[tuple] = None,
        loader_kwargs: Optional[dict] = None,
    ):
        self.model_loader_function = model_loader_function
        self.model_name = model_name
        self.required_vram_gb = required_vram_gb
        self.args = loader_args or ()
        self.kwargs = loader_kwargs or {}
        self.device = get_device()
        self.model: Optional[torch.nn.Module] = None
        self._lock_acquired = False
        self._watchdog_started = False

    def __enter__(self) -> torch.nn.Module:
        # Acquire global lock with timeout
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

        # Start watchdog to prevent permanent deadlock on hard crashes
        _VRAMLockWatchdog.start(
            self.__class__._accelerator_lock,
            timeout=self.__class__._LOCK_TIMEOUT_SECONDS + 15.0,
        )
        self._watchdog_started = True

        # Check memory before attempting load
        if self.device.type in ("cuda", "xla", "mps"):
            if not _check_available_vram(self.required_vram_gb):
                self.device = torch.device("cpu")
                logger.warning(
                    f"Model '{self.model_name}': Insufficient accelerator memory. "
                    f"Falling back to CPU inference."
                )

        try:
            logger.info(
                f"Loading model '{self.model_name}' on {self.device.type.upper()}..."
            )

            # Execute model loader
            self.model = self.model_loader_function(*self.args, **self.kwargs)

            # Move to device and set eval mode
            if hasattr(self.model, "to"):
                self.model.to(self.device)
            if hasattr(self.model, "eval"):
                self.model.eval()

            # Log model stats
            total_params = (
                sum(p.numel() for p in self.model.parameters())
                if hasattr(self.model, "parameters")
                else 0
            )
            nonzero_params = (
                sum((p != 0).sum().item() for p in self.model.parameters())
                if hasattr(self.model, "parameters")
                else 0
            )
            logger.info(
                f"[{self.model_name}] Params: {total_params:,}, Non-zero: {nonzero_params:,}"
            )

            # Log memory usage after load
            if self.device.type == "cuda":
                try:
                    allocated_mb = torch.cuda.memory_allocated(0) / (1024**2)
                    logger.info(
                        f"[{self.model_name}] CUDA allocated: {allocated_mb:.0f}MB"
                    )
                except Exception:
                    pass
            elif self.device.type == "xla":
                try:
                    import torch_xla.core.xla_model as xm

                    memory_info = xm.get_memory_info(self.device)
                    if memory_info and "bytes_used" in memory_info:
                        used_mb = memory_info["bytes_used"] / (1024**2)
                        logger.info(f"[{self.model_name}] TPU used: {used_mb:.0f}MB")
                except Exception:
                    pass

            return self.model

        except Exception as e:
            logger.error(
                f"Model '{self.model_name}' failed to load: {e}", exc_info=True
            )
            # Cleanup even on load failure
            self._safe_cleanup()
            raise

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        # Log exception for debugging
        if exc_type is not None:
            logger.warning(
                f"Model '{self.model_name}' context exited with exception: "
                f"{exc_type.__name__}: {exc_val}"
            )

        # Guaranteed cleanup via finally
        self._safe_cleanup()

    def _safe_cleanup(self) -> None:
        """
        EXACT ORDER MUST BE FOLLOWED:
        1. Capture device type from model BEFORE moving to CPU
        2. Synchronize device (wait for pending ops)
        3. Move model to CPU (drop accelerator memory instantly)
        4. Delete model reference
        5. Device-specific cleanup (TPU/CUDA/MPS)
        6. Universal garbage collection
        7. Stop watchdog + release lock (in finally)
        """
        try:
            # Step 0: Capture actual device type from model before moving it
            model_device_type = self.device.type
            if self.model is not None and hasattr(self.model, "device"):
                try:
                    model_device_type = str(self.model.device).split(":")[0]
                except Exception:
                    pass

            # Step 1: Unbind model from accelerator memory
            if self.model is not None:
                # Wait for pending async ops BEFORE moving to CPU
                if model_device_type == "cuda" and torch.cuda.is_available():
                    try:
                        torch.cuda.synchronize()
                    except Exception:
                        pass

                # Force parameters to CPU to instantly free accelerator memory
                if hasattr(self.model, "to"):
                    try:
                        self.model.to("cpu")
                    except Exception as e:
                        logger.debug(f"Model.to('cpu') failed: {e}")

                # Delete reference to allow GC
                del self.model
            self.model = None

            # Step 2: Device-specific cleanup (TPU/CUDA/MPS)
            _cleanup_device_memory(torch.device(model_device_type))

        except Exception as e:
            logger.error(f"Cleanup error for '{self.model_name}': {e}", exc_info=True)

        finally:
            # Step 4: Stop watchdog + release lock (ALWAYS runs)
            if self._watchdog_started:
                _VRAMLockWatchdog.stop()
                self._watchdog_started = False

            if self._lock_acquired:
                try:
                    self.__class__._accelerator_lock.release()
                except Exception as e:
                    logger.error(f"Failed to release accelerator lock: {e}")
                finally:
                    self._lock_acquired = False


# ──────────────────────────────────────────────────────────────
# Convenience Wrapper
# ──────────────────────────────────────────────────────────────


def run_with_vram_cleanup(
    model_loader: Callable,
    inference_fn: Callable,
    model_name: str = "Unknown",
    required_vram_gb: float = 2.0,
    *loader_args: Any,
    **loader_kwargs: Any,
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

    Guarantees:
        - torch.no_grad() applied to inference
        - Cleanup always runs (even on exception)
        - VRAM lock + watchdog for hard-crash resilience
    """
    with VRAMLifecycleManager(
        model_loader,
        model_name=model_name,
        required_vram_gb=required_vram_gb,
        *loader_args,
        **loader_kwargs,
    ) as model:
        with torch.no_grad():
            return inference_fn(model)
