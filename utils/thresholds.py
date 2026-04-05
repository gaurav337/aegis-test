"""
Aegis-X Thresholds Configuration (v6.1 - Audit Corrected)
Single source of truth for all numeric thresholds used across Aegis-X.
All tools import from this file. No hardcoded values elsewhere.
"""

# =============================================================================
# HARDWARE & VRAM THRESHOLDS
# =============================================================================
VRAM_MIN_FOR_GPU = 2.0  # Minimum VRAM (GB) to use GPU
VRAM_RECOMMENDED = 4.0
VRAM_MODEL_LOAD_THRESHOLD = 2.0
VRAM_RESERVED_BUFFER_GB = 0.5
GPU_DECODE_BATCH_SIZE = 8
CPU_DECODE_BATCH_SIZE = 32
ENABLE_TPU_SUPPORT = False
TPU_MEMORY_FRACTION = 0.8
GPU_MEMORY_FRACTION = 0.9
MAX_BATCH_SIZE = 64
MAX_FRAME_DIMENSION = 0  # 0 = disabled (preserve native res)
FALLBACK_FPS = 30.0

# =============================================================================
# ENSEMBLE WEIGHTS (Corrected to sum exactly to 1.0)
# =============================================================================
# AUDIT FIX C-03: Removed duplicate WEIGHT_SIGLIP alias, rebalanced to 1.0
WEIGHT_UNIVFD = 0.15
WEIGHT_XCEPTION = 0.15
WEIGHT_SBI = 0.15
WEIGHT_FREQNET = 0.15
WEIGHT_RPPG = 0.06
WEIGHT_DCT = 0.09
WEIGHT_GEOMETRY = 0.12
WEIGHT_ILLUMINATION = 0.04
WEIGHT_CORNEAL = 0.04
WEIGHT_C2PA = 0.05

# Backward compatibility alias (points to univfd)
WEIGHT_SIGLIP = WEIGHT_UNIVFD

# Validation guard
_ALL_WEIGHTS = [
    WEIGHT_UNIVFD,
    WEIGHT_XCEPTION,
    WEIGHT_SBI,
    WEIGHT_FREQNET,
    WEIGHT_RPPG,
    WEIGHT_DCT,
    WEIGHT_GEOMETRY,
    WEIGHT_ILLUMINATION,
    WEIGHT_CORNEAL,
    WEIGHT_C2PA,
]
assert abs(sum(_ALL_WEIGHTS) - 1.0) < 1e-6, (
    f"Weights sum to {sum(_ALL_WEIGHTS)}, must equal 1.0"
)

# =============================================================================
# ENSEMBLE DECISION THRESHOLDS
# =============================================================================
REAL_THRESHOLD = 0.15
FAKE_THRESHOLD = 0.85
ENSEMBLE_REAL_THRESHOLD = 0.42
ENSEMBLE_FAKE_THRESHOLD = 0.60
ENSEMBLE_INCONCLUSIVE_WEIGHT = 0.50
LOGIT_DAMPING_FACTOR = 0.45
NEUTRAL_DEADZONE_LOW = 0.42
NEUTRAL_DEADZONE_HIGH = 0.58
NEUTRAL_DEADZONE_CONFIDENCE_MIN = 0.85

# Borderline Consensus (AUDIT FIX S-02: widened to prevent fake bias)
BORDERLINE_CONSENSUS_LOW = 0.35
BORDERLINE_CONSENSUS_HIGH = 0.55
BORDERLINE_CONSENSUS_BOOST = 1.00  # Neutral: adaptive damping handles uncertainty now
GPU_COVERAGE_DEGRADATION_FACTOR = (
    0.02  # Halved: tools no longer fully abstain from borderline
)

# =============================================================================
# COMPRESSION DISCOUNTS
# =============================================================================
DCT_DOUBLE_QUANT_COMPRESSION_THRESHOLD = 0.80
SBI_COMPRESSION_DISCOUNT = 0.40
FREQNET_COMPRESSION_DISCOUNT = 0.50

# =============================================================================
# SBI THRESHOLDS
# =============================================================================
SBI_BLIND_SPOT_THRESHOLD = 0.50
SBI_HIGH_CONFIDENCE_THRESHOLD = 0.80
SBI_MID_BAND_BASE_WEIGHT = 0.03
SBI_MID_BAND_CLIP_MULTIPLIER = 0.12
SBI_SKIP_UNIVFD_THRESHOLD = (
    0.90  # AUDIT FIX S-09: Raised to prevent evidence suppression
)
SBI_FAKE_THRESHOLD = 0.50
SBI_GRADCAM_REGION_THRESHOLD = 0.40
SBI_SCALE_115 = 1.15
SBI_SCALE_125 = 1.25
SBI_CONFIDENCE_THRESHOLD = 0.6
SBI_BATCH_SIZE = 16
SBI_MODEL_PATH = "models/sbi_v4.pt"
SBI_THRESHOLD = 0.50

# =============================================================================
# FREQNET THRESHOLDS
# =============================================================================
FREQNET_Z_THRESHOLD = 1.5
FREQNET_HIGH_BAND_RATIO_THRESHOLD = 0.15
FREQNET_FAKE_THRESHOLD = 0.50
FREQNET_BLIND_SPOT_THRESHOLD = 0.45
FREQNET_CALIBRATION_PATH = "calibration/freqnet_fad_baseline.pt"
FREQNET_CONFIDENCE_THRESHOLD = 0.6
FREQNET_DCT_COEFFICIENTS = 64
FREQNET_MODEL_PATH = "models/freqnet_v4.pt"
FREQNET_THRESHOLD = 0.6
# FAD Fusion Thresholds
FREQNET_FAD_MID_EXCESS = 0.35
FREQNET_FAD_HIGH_EXCESS = 0.20
FREQNET_FAD_MID_MULTIPLIER = 3.0
FREQNET_FAD_HIGH_MULTIPLIER = 5.0
FREQNET_NEURAL_WEIGHT = 0.70
FREQNET_FAD_WEIGHT = 0.30

# =============================================================================
# RPPG THRESHOLDS (AUDIT FIX C-02: Added signal-quality gate constants)
# =============================================================================
RPPG_PULSE_THRESHOLD_LOW = 0.30
RPPG_PULSE_THRESHOLD_HIGH = 0.70
RPPG_NO_PULSE_IMPLIED_PROB = 0.85
RPPG_MIN_FRAMES = 90
RPPG_SNR_THRESHOLD = 3.0
RPPG_FFT_NFFT = 2048
RPPG_CARDIAC_BAND_MIN_HZ = 0.7
RPPG_CARDIAC_BAND_MAX_HZ = 2.5
RPPG_HAIR_OCCLUSION_VARIANCE = 250.0
RPPG_MIN_TEMPORAL_STD = 0.05
RPPG_COHERENCE_THRESHOLD_HZ = 0.5
RPPG_HEART_RATE_MIN = 40
RPPG_HEART_RATE_MAX = 180
RPPG_SIGNAL_QUALITY_MIN = 0.6
RPPG_CONFIDENCE_THRESHOLD = 0.7

# =============================================================================
# DCT THRESHOLDS (AUDIT FIX C-04 & S-01)
# =============================================================================
DCT_RATIO_THRESHOLD = 0.88
DCT_RATIO_SCALE = 0.10
DCT_CONFIDENCE_CAP = 0.7
DCT_CONFIDENCE_BUMP = 0.15
DCT_HISTOGRAM_BINS = 513
DCT_AC_MASK_SUM_MAX = 5
DCT_THRESHOLD = 0.55
DCT_CONFIDENCE_MIN = 0.5
DCT_MIN_CROP_SIZE = 224
DCT_HIGH_CONFIDENCE_RATIO = 0.95
DCT_CORRECTED_Z_THRESHOLD = 4.15  # AUDIT FIX S-01: Multiple testing correction

# =============================================================================
# GEOMETRY THRESHOLDS
# =============================================================================
GEOMETRY_YAW_SKIP_THRESHOLD = 0.25
GEOMETRY_IPD_RATIO_MIN = 0.35
GEOMETRY_IPD_RATIO_MAX = 0.60
GEOMETRY_PHILTRUM_RATIO_MIN = 0.06
GEOMETRY_PHILTRUM_RATIO_MAX = 0.20
GEOMETRY_EYE_ASYMMETRY_MAX = 0.08
GEOMETRY_NOSE_WIDTH_RATIO_MIN = 0.45
GEOMETRY_NOSE_WIDTH_RATIO_MAX = 0.80
GEOMETRY_MOUTH_WIDTH_RATIO_MIN = 0.75
GEOMETRY_MOUTH_WIDTH_RATIO_MAX = 1.15
GEOMETRY_VERTICAL_THIRDS_MAX_DEVIATION = 0.26
GEOMETRY_LANDMARK_THRESHOLD = 0.8
GEOMETRY_SYMMETRY_THRESHOLD = 0.7
GEOMETRY_CONFIDENCE_MIN = 0.5

# =============================================================================
# ILLUMINATION THRESHOLDS
# =============================================================================
ILLUMINATION_DIFFUSE_THRESHOLD = 0.08
ILLUMINATION_GRADIENT_CONSISTENT_WEIGHT = 0.15
ILLUMINATION_GRADIENT_MISMATCH_WEIGHT = 0.50
ILLUMINATION_MISMATCH_BASE_PENALTY = 0.20
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
CORNEAL_BLIND_SPOT_THRESHOLD = 0.30  # Added for ensemble routing
CORNEAL_SIGMA_FLOOR = 5.0  # AUDIT FIX M-01: Prevent threshold collapse on dark irises

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
# C2PA THRESHOLDS (AUDIT FIX S-03: Lowered contradiction threshold)
# =============================================================================
C2PA_SHORT_CIRCUIT_ENABLED = True
C2PA_VISUAL_CONTRADICTION_THRESHOLD = 0.55  # Was 0.92
C2PA_VISUAL_MIN_WEIGHT = 0.40
C2PA_VERIFICATION_STRICT = True
C2PA_CACHE_EXPIRY_SECONDS = 3600

# =============================================================================
# ENSEMBLE CONFLICT DETECTION
# =============================================================================
# Specialist Dominance (AUDIT FIX S-16: GPU tool priority)
ENSEMBLE_SPECIALIST_DOMINANCE_FACTOR = 2.5
ENSEMBLE_SPECIALIST_MIN_CONFIDENCE = 0.40
ENSEMBLE_SPECIALIST_FAKE_THRESHOLD = 0.50
GPU_SINGLE_DETECT_HIGH_CONFIDENCE = 0.80
GPU_MULTI_DETECT_MIN_CONFIDENCE = 0.55
GPU_SINGLE_DETECT_BOOST = 0.12
GPU_MULTI_DETECT_BOOST = 0.10
CPU_SUPPORT_WHEN_GPU_AVAILABLE = 0.15

# Conflict & Smoothing
CONFLICT_STD_THRESHOLD = 0.20
SUSPICION_OVERRIDE_THRESHOLD = 0.90
OVERRIDE_AGREEMENT_THRESHOLD = 0.80
OVERRIDE_MIN_AGREEMENT = 2
EMA_SMOOTHING_ALPHA = 0.30
EMA_SMOOTHING_ENABLED = True

# =============================================================================
# ENCORE CORROBORATION (NEAR-MISS) THRESHOLDS
# =============================================================================
ENCORE_NEAR_MISS_THRESHOLD = 0.42
ENCORE_CORROBORATION_SENSITIVITY = 0.50  # Partial mass for near-miss suspects

# =============================================================================
# THRESHOLD CONFIG DATACLASS
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
        return (self.real_threshold, self.fake_threshold)
