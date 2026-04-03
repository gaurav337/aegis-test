"""
Aegis-X Thresholds Configuration (v6.0 - Final Unified)
========================================================
Single source of truth for all numeric thresholds used across Aegis-X.

All tools import from this file. No hardcoded values elsewhere.
"""

# =============================================================================
# HARDWARE & VRAM THRESHOLDS
# =============================================================================

VRAM_MIN_FOR_GPU = 3.5              # Minimum VRAM (GB) to use GPU
VRAM_RECOMMENDED = 6.0              # Recommended VRAM for comfortable inference
VRAM_MODEL_LOAD_THRESHOLD = 3.5     # Threshold for model loading decision
VRAM_RESERVED_BUFFER_GB = 1.0       # Reserved VRAM buffer for safety

GPU_DECODE_BATCH_SIZE = 8           # Batch size for GPU video decoding
CPU_DECODE_BATCH_SIZE = 32          # Batch size for CPU video decoding

ENABLE_TPU_SUPPORT = False          # FIX: Missing constant for tools
TPU_MEMORY_FRACTION = 0.8           # FIX: Missing constant for SigLIP/SBI/FreqNet
GPU_MEMORY_FRACTION = 0.9
MAX_BATCH_SIZE = 64
MAX_FRAME_DIMENSION = 0             # 0 = disabled (preserve native res)
FALLBACK_FPS = 30.0                 # Fallback FPS for OpenCV

# =============================================================================
# ENSEMBLE WEIGHTS (Day 15)
# =============================================================================

WEIGHT_UNIVFD = 0.15
WEIGHT_SIGLIP = 0.15                # Alias for backward compatibility
WEIGHT_XCEPTION = 0.10
WEIGHT_SBI = 0.18
WEIGHT_FREQNET = 0.09
WEIGHT_RPPG = 0.06
WEIGHT_DCT = 0.07
WEIGHT_GEOMETRY = 0.18
WEIGHT_ILLUMINATION = 0.05
WEIGHT_CORNEAL = 0.07
WEIGHT_C2PA = 0.05

# =============================================================================
# ENSEMBLE DECISION THRESHOLDS (Day 16 Early Stopping)
# =============================================================================

REAL_THRESHOLD = 0.15               # FIX: For EarlyStoppingController
FAKE_THRESHOLD = 0.85               # FIX: For EarlyStoppingController
ENSEMBLE_REAL_THRESHOLD = 0.40
ENSEMBLE_FAKE_THRESHOLD = 0.60
ENSEMBLE_INCONCLUSIVE_WEIGHT = 0.50

# =============================================================================
# COMPRESSION DISCOUNTS (Cross-tool)
# =============================================================================

DCT_DOUBLE_QUANT_COMPRESSION_THRESHOLD = 0.80
SBI_COMPRESSION_DISCOUNT = 0.40
FREQNET_COMPRESSION_DISCOUNT = 0.50

# =============================================================================
# SBI THRESHOLDS
# =============================================================================

SBI_BLIND_SPOT_THRESHOLD = 0.50       # Only vote when actual blend boundary detected
SBI_HIGH_CONFIDENCE_THRESHOLD = 0.80
SBI_MID_BAND_BASE_WEIGHT = 0.03
SBI_MID_BAND_CLIP_MULTIPLIER = 0.12
SBI_SKIP_UNIVFD_THRESHOLD = 0.70
SBI_FAKE_THRESHOLD = 0.50
SBI_GRADCAM_REGION_THRESHOLD = 0.40
SBI_SCALE_115 = 1.15
SBI_SCALE_125 = 1.25
SBI_CONFIDENCE_THRESHOLD = 0.6      # FIX: Missing constant
SBI_BATCH_SIZE = 16                 # FIX: Missing constant
SBI_MODEL_PATH = "models/sbi_v4.pt" # FIX: Missing constant
SBI_THRESHOLD = 0.50

# =============================================================================
# FREQNET THRESHOLDS
# =============================================================================

FREQNET_Z_THRESHOLD = 1.5
FREQNET_HIGH_BAND_RATIO_THRESHOLD = 0.15
FREQNET_FAKE_THRESHOLD = 0.50
FREQNET_BLIND_SPOT_THRESHOLD = 0.45   # Only vote when actual frequency anomaly detected
FREQNET_CALIBRATION_PATH = "calibration/freqnet_fad_baseline.pt"
FREQNET_CONFIDENCE_THRESHOLD = 0.6  # FIX: Missing constant
FREQNET_DCT_COEFFICIENTS = 64       # FIX: Missing constant
FREQNET_MODEL_PATH = "models/freqnet_v4.pt" # FIX: Missing constant
FREQNET_THRESHOLD = 0.6

# FAD Fusion Thresholds
FREQNET_FAD_MID_EXCESS = 0.30
FREQNET_FAD_HIGH_EXCESS = 0.08
FREQNET_FAD_MID_MULTIPLIER = 3.0
FREQNET_FAD_HIGH_MULTIPLIER = 5.0
FREQNET_NEURAL_WEIGHT = 0.70
FREQNET_FAD_WEIGHT = 0.30

# =============================================================================
# RPPG THRESHOLDS
# =============================================================================

RPPG_PULSE_THRESHOLD_LOW = 0.30
RPPG_PULSE_THRESHOLD_HIGH = 0.70
RPPG_NO_PULSE_IMPLIED_PROB = 0.85
RPPG_MIN_FRAMES = 90
RPPG_SNR_THRESHOLD = 3.0
RPPG_FFT_NFFT = 2048
RPPG_CARDIAC_BAND_MIN_HZ = 0.7
RPPG_CARDIAC_BAND_MAX_HZ = 2.5
RPPG_HAIR_OCCLUSION_VARIANCE = 0.25 # FIX: Missing constant
RPPG_MIN_TEMPORAL_STD = 0.05        # FIX: Missing constant
RPPG_COHERENCE_THRESHOLD_HZ = 0.5
RPPG_HEART_RATE_MIN = 40            # FIX: Missing constant
RPPG_HEART_RATE_MAX = 180           # FIX: Missing constant
RPPG_SIGNAL_QUALITY_MIN = 0.6       # FIX: Missing constant
RPPG_CONFIDENCE_THRESHOLD = 0.7

# =============================================================================
# DCT THRESHOLDS
# =============================================================================

DCT_RATIO_THRESHOLD = 0.75
DCT_RATIO_SCALE = 0.15
DCT_CONFIDENCE_CAP = 0.9
DCT_CONFIDENCE_BUMP = 0.2
DCT_HISTOGRAM_BINS = 513
DCT_AC_MASK_SUM_MAX = 5
DCT_THRESHOLD = 0.55
DCT_CONFIDENCE_MIN = 0.5

# =============================================================================
# GEOMETRY THRESHOLDS
# =============================================================================

GEOMETRY_YAW_SKIP_THRESHOLD = 0.18
GEOMETRY_IPD_RATIO_MIN = 0.42
GEOMETRY_IPD_RATIO_MAX = 0.52
GEOMETRY_PHILTRUM_RATIO_MIN = 0.10
GEOMETRY_PHILTRUM_RATIO_MAX = 0.15
GEOMETRY_EYE_ASYMMETRY_MAX = 0.05
GEOMETRY_NOSE_WIDTH_RATIO_MIN = 0.55
GEOMETRY_NOSE_WIDTH_RATIO_MAX = 0.70
GEOMETRY_MOUTH_WIDTH_RATIO_MIN = 0.85
GEOMETRY_MOUTH_WIDTH_RATIO_MAX = 1.05
GEOMETRY_VERTICAL_THIRDS_MAX_DEVIATION = 0.15
GEOMETRY_LANDMARK_THRESHOLD = 0.8
GEOMETRY_SYMMETRY_THRESHOLD = 0.7
GEOMETRY_CONFIDENCE_MIN = 0.5

# =============================================================================
# ILLUMINATION THRESHOLDS
# =============================================================================

ILLUMINATION_DIFFUSE_THRESHOLD = 0.05
ILLUMINATION_GRADIENT_CONSISTENT_WEIGHT = 0.20
ILLUMINATION_GRADIENT_MISMATCH_WEIGHT = 0.70
ILLUMINATION_MISMATCH_BASE_PENALTY = 0.30
ILLUMINATION_CONSISTENCY_THRESHOLD = 0.7
ILLUMINATION_SHADOW_THRESHOLD = 0.6

# =============================================================================
# CORNEAL REFLECTION THRESHOLDS
# =============================================================================

CORNEAL_BOX_SIZE = 15
CORNEAL_MAX_DIVERGENCE = 0.5
CORNEAL_CONSISTENCY_THRESHOLD = 0.5
CORNEAL_REFLECTION_THRESHOLD = 0.75
CORNEAL_SYMMETRY_THRESHOLD = 0.7

# =============================================================================
# UNIVFD & XCEPTION THRESHOLDS
# =============================================================================

UNIVFD_FAKE_THRESHOLD = 0.50
UNIVFD_CONFIDENCE_MIN = 0.5
UNIVFD_TTA_ENABLED = True

XCEPTION_FAKE_THRESHOLD = 0.50
XCEPTION_CONFIDENCE_BASE = 0.40
XCEPTION_CONFIDENCE_MULTIPLIER = 1.6
XCEPTION_MATCH_RATIO_MIN = 0.70
XCEPTION_PARTIAL_LOAD_CAP = 0.30

# Backward compatibility for old configs
SIGLIP_THRESHOLD = 0.6
SIGLIP_CONFIDENCE_MIN = 0.5

# =============================================================================
# AGENT & ORCHESTRATION THRESHOLDS
# =============================================================================

EARLY_STOP_CONFIDENCE = 0.75
CONFIDENCE_GATE_THRESHOLD = 0.75
MIN_WEIGHT_FOR_STOP = 0.40
AGENT_MAX_RETRIES = 3
AGENT_LLM_TIMEOUT = 120
LLM_TEMPERATURE = 0.1
LLM_MAX_TOKENS = 512
LLM_RETRY_MAX = 2

# =============================================================================
# PREPROCESSING THRESHOLDS
# =============================================================================

FACE_DETECTION_CONFIDENCE = 0.80
LANDMARK_CONFIDENCE = 0.75
MIN_FACE_SIZE = 80
PREPROCESSING_MAX_SUBJECTS = 2
PREPROCESSING_MIN_FACE_RESOLUTION = 64
PREPROCESSING_FACE_CROP_SIZE = 224
PREPROCESSING_SBI_CROP_SIZE = 380
PREPROCESSING_MAX_VIDEO_FRAMES = 300
PREPROCESSING_MIN_VIDEO_FRAMES = 90
PREPROCESSING_EXTRACT_FPS = 30
PREPROCESSING_QUALITY_SNIPE_SAMPLES = 5

# =============================================================================
# C2PA THRESHOLDS
# =============================================================================

C2PA_SHORT_CIRCUIT_ENABLED = True
C2PA_VISUAL_CONTRADICTION_THRESHOLD = 0.80
C2PA_VISUAL_MIN_WEIGHT = 0.40
C2PA_VERIFICATION_STRICT = True
C2PA_CACHE_EXPIRY_SECONDS = 3600

# =============================================================================
# ENSEMBLE CONFLICT DETECTION
# =============================================================================

CONFLICT_STD_THRESHOLD = 0.20
SUSPICION_OVERRIDE_THRESHOLD = 0.50   # Max-pool fires when any specialist above this
EMA_SMOOTHING_ALPHA = 0.30
EMA_SMOOTHING_ENABLED = True

# =============================================================================
# THRESHOLD CONFIG DATACLASS (For EarlyStoppingController)
# =============================================================================

from dataclasses import dataclass


@dataclass(frozen=True)
class ThresholdConfig:
    """Immutable threshold configuration for EarlyStoppingController."""
    real_threshold: float = REAL_THRESHOLD
    fake_threshold: float = FAKE_THRESHOLD

    def __post_init__(self):
        if not (0.0 < self.real_threshold < self.fake_threshold < 1.0):
            raise ValueError(
                f"Thresholds must satisfy 0 < real ({self.real_threshold}) "
                f"< fake ({self.fake_threshold}) < 1"
            )

    def to_tuple(self) -> tuple:
        """Return thresholds as tuple for EarlyStoppingController."""
        return (self.real_threshold, self.fake_threshold)
RPPG_COHERENCE_THRESHOLD_HZ = 0.5  # RPPG coherence threshold
