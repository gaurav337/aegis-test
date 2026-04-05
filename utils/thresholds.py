"""
Aegis-X Thresholds Configuration (v6.0 - Final Unified)
========================================================
Single source of truth for all numeric thresholds used across Aegis-X.

All tools import from this file. No hardcoded values elsewhere.
"""

# =============================================================================
# HARDWARE & VRAM THRESHOLDS
# =============================================================================

VRAM_MIN_FOR_GPU = (
    2.0  # Minimum VRAM (GB) to use GPU (lowered for 4GB GPUs like RTX 3050)
)
VRAM_RECOMMENDED = 4.0  # Recommended VRAM for comfortable inference
VRAM_MODEL_LOAD_THRESHOLD = (
    2.0  # Threshold for model loading decision (lowered for 4GB GPUs)
)
VRAM_RESERVED_BUFFER_GB = (
    0.5  # Reserved VRAM buffer for safety (reduced for smaller GPUs)
)

GPU_DECODE_BATCH_SIZE = 8  # Batch size for GPU video decoding
CPU_DECODE_BATCH_SIZE = 32  # Batch size for CPU video decoding

ENABLE_TPU_SUPPORT = False  # FIX: Missing constant for tools
TPU_MEMORY_FRACTION = 0.8  # FIX: Missing constant for SigLIP/SBI/FreqNet
GPU_MEMORY_FRACTION = 0.9
MAX_BATCH_SIZE = 64
MAX_FRAME_DIMENSION = 0  # 0 = disabled (preserve native res)
FALLBACK_FPS = 30.0  # Fallback FPS for OpenCV

# =============================================================================
# ENSEMBLE WEIGHTS (Day 15)
# =============================================================================

# Working GPU tools (calibrated):
WEIGHT_UNIVFD = 0.05  # Reduced: 4KB probe is broken, returns ~0.41 for everything
WEIGHT_SIGLIP = 0.05  # Alias for backward compatibility
WEIGHT_XCEPTION = 0.25  # Increased: working correctly, gives discriminative scores
WEIGHT_SBI = (
    0.20  # Reduced from 0.25: temperature scaled, still useful but less dominant
)
WEIGHT_FREQNET = 0.05  # Reduced: neural model barely works, FAD fallback noisy
# CPU tools:
WEIGHT_RPPG = 0.06
WEIGHT_DCT = 0.10  # Increased: working reasonably well
WEIGHT_GEOMETRY = 0.12  # Increased: reliable structural analysis
WEIGHT_ILLUMINATION = 0.04
WEIGHT_CORNEAL = 0.04
WEIGHT_C2PA = 0.05

# =============================================================================
# ENSEMBLE DECISION THRESHOLDS (Day 16 Early Stopping)
# =============================================================================

REAL_THRESHOLD = 0.15  # FIX: For EarlyStoppingController
FAKE_THRESHOLD = 0.85  # FIX: For EarlyStoppingController
ENSEMBLE_REAL_THRESHOLD = 0.50
ENSEMBLE_FAKE_THRESHOLD = 0.60
ENSEMBLE_INCONCLUSIVE_WEIGHT = 0.50

# Borderline Consensus: when multiple GPU specialists cluster near 50%,
# their agreement is itself a corroborating signal of manipulation.
BORDERLINE_CONSENSUS_LOW = 0.35  # Lower bound of the "borderline" zone
BORDERLINE_CONSENSUS_HIGH = 0.55  # Upper bound of the "borderline" zone
BORDERLINE_CONSENSUS_BOOST = 1.10  # Reduced from 1.25 to prevent over-penalizing noise
GPU_COVERAGE_DEGRADATION_FACTOR = 0.05  # Reduced from 0.10 per-abstained specialist

# =============================================================================
# COMPRESSION DISCOUNTS (Cross-tool)
# =============================================================================

DCT_DOUBLE_QUANT_COMPRESSION_THRESHOLD = 0.80
SBI_COMPRESSION_DISCOUNT = 0.40
FREQNET_COMPRESSION_DISCOUNT = 0.50

# =============================================================================
# SBI THRESHOLDS
# =============================================================================

SBI_BLIND_SPOT_THRESHOLD = 0.50  # Only vote when actual blend boundary detected
SBI_HIGH_CONFIDENCE_THRESHOLD = 0.80
SBI_MID_BAND_BASE_WEIGHT = 0.03
SBI_MID_BAND_CLIP_MULTIPLIER = 0.12
SBI_SKIP_UNIVFD_THRESHOLD = 0.70
SBI_FAKE_THRESHOLD = 0.50
SBI_GRADCAM_REGION_THRESHOLD = 0.40
SBI_SCALE_115 = 1.15
SBI_SCALE_125 = 1.25
SBI_CONFIDENCE_THRESHOLD = 0.6  # FIX: Missing constant
SBI_BATCH_SIZE = 16  # FIX: Missing constant
SBI_MODEL_PATH = "models/sbi_v4.pt"  # FIX: Missing constant
SBI_THRESHOLD = 0.50

# =============================================================================
# FREQNET THRESHOLDS
# =============================================================================

FREQNET_Z_THRESHOLD = 1.5
FREQNET_HIGH_BAND_RATIO_THRESHOLD = 0.15
FREQNET_FAKE_THRESHOLD = 0.50
FREQNET_BLIND_SPOT_THRESHOLD = 0.45  # Only vote when actual frequency anomaly detected
FREQNET_CALIBRATION_PATH = "calibration/freqnet_fad_baseline.pt"
FREQNET_CONFIDENCE_THRESHOLD = 0.6  # FIX: Missing constant
FREQNET_DCT_COEFFICIENTS = 64  # FIX: Missing constant
FREQNET_MODEL_PATH = "models/freqnet_v4.pt"  # FIX: Missing constant
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
RPPG_HAIR_OCCLUSION_VARIANCE = 0.25  # FIX: Missing constant
RPPG_MIN_TEMPORAL_STD = 0.05  # FIX: Missing constant
RPPG_COHERENCE_THRESHOLD_HZ = 0.5
RPPG_HEART_RATE_MIN = 40  # FIX: Missing constant
RPPG_HEART_RATE_MAX = 180  # FIX: Missing constant
RPPG_SIGNAL_QUALITY_MIN = 0.6  # FIX: Missing constant
RPPG_CONFIDENCE_THRESHOLD = 0.7

# =============================================================================
# DCT THRESHOLDS
# =============================================================================
# NOTE: DCT analysis on 224x224 face crops (~784 8x8 blocks) is statistically
# unreliable. Phone camera JPEGs naturally have DCT block patterns.
# These thresholds are calibrated to avoid false positives on real JPEG images.

DCT_RATIO_THRESHOLD = 0.88  # Raised from 0.75 — phone JPEGs naturally hit 0.80-0.90
DCT_RATIO_SCALE = 0.10  # Narrowed scale for sharper discrimination
DCT_CONFIDENCE_CAP = 0.7  # Lowered from 0.9 — small crops are inherently noisy
DCT_CONFIDENCE_BUMP = 0.15  # Lowered from 0.2
DCT_HISTOGRAM_BINS = 513
DCT_AC_MASK_SUM_MAX = 5
DCT_THRESHOLD = 0.55
DCT_CONFIDENCE_MIN = 0.5
# Minimum crop size for reliable DCT analysis (224x224 = 784 blocks is borderline)
DCT_MIN_CROP_SIZE = 224
# Peak ratio above which we're confident of double-quantization (not just JPEG)
DCT_HIGH_CONFIDENCE_RATIO = 0.95

# =============================================================================
# GEOMETRY THRESHOLDS
# =============================================================================
# NOTE: Human facial anatomy has natural variation. These ranges are calibrated
# to avoid false positives on real faces with natural proportions.

GEOMETRY_YAW_SKIP_THRESHOLD = 0.25  # Raised from 0.22 — more tolerant of head turns
GEOMETRY_IPD_RATIO_MIN = 0.35  # Widened from 0.38
GEOMETRY_IPD_RATIO_MAX = 0.60  # Widened from 0.56
GEOMETRY_PHILTRUM_RATIO_MIN = 0.06  # Widened from 0.08
GEOMETRY_PHILTRUM_RATIO_MAX = 0.20  # Widened from 0.18
GEOMETRY_EYE_ASYMMETRY_MAX = 0.08  # Widened from 0.06
GEOMETRY_NOSE_WIDTH_RATIO_MIN = 0.45  # Widened from 0.50
GEOMETRY_NOSE_WIDTH_RATIO_MAX = 0.80  # Widened from 0.75
GEOMETRY_MOUTH_WIDTH_RATIO_MIN = 0.75  # Widened from 0.80
GEOMETRY_MOUTH_WIDTH_RATIO_MAX = 1.15  # Widened from 1.10
GEOMETRY_VERTICAL_THIRDS_MAX_DEVIATION = (
    0.26  # Raised from 0.22 — more tolerant of selfie camera distortion
)
GEOMETRY_LANDMARK_THRESHOLD = 0.8
GEOMETRY_SYMMETRY_THRESHOLD = 0.7
GEOMETRY_CONFIDENCE_MIN = 0.5

# =============================================================================
# ILLUMINATION THRESHOLDS
# =============================================================================
# NOTE: Outdoor/open-place lighting is complex (sky above, ground reflections,
# multiple light sources). The tool should not flag natural lighting variations.

ILLUMINATION_DIFFUSE_THRESHOLD = (
    0.08  # Raised from 0.05 — more tolerant of subtle gradients
)
ILLUMINATION_GRADIENT_CONSISTENT_WEIGHT = (
    0.15  # Lowered from 0.20 — consistent lighting shouldn't penalize
)
ILLUMINATION_GRADIENT_MISMATCH_WEIGHT = (
    0.50  # Lowered from 0.70 — outdoor scenes naturally have complex lighting
)
ILLUMINATION_MISMATCH_BASE_PENALTY = (
    0.20  # Lowered from 0.30 — base penalty too aggressive
)
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
C2PA_VISUAL_CONTRADICTION_THRESHOLD = 0.92  # Raised from 0.80 — must be near-certain FAKE to override a cryptographic signature
C2PA_VISUAL_MIN_WEIGHT = 0.40
C2PA_VERIFICATION_STRICT = True
C2PA_CACHE_EXPIRY_SECONDS = 3600

# =============================================================================
# ENSEMBLE CONFLICT DETECTION
# =============================================================================

CONFLICT_STD_THRESHOLD = 0.20
SUSPICION_OVERRIDE_THRESHOLD = (
    0.90  # Raised from 0.70 — requires near-certain single-tool evidence
)
OVERRIDE_AGREEMENT_THRESHOLD = (
    0.80  # Individual tool must exceed this to count toward agreement
)
OVERRIDE_MIN_AGREEMENT = (
    2  # At least N working GPU tools must agree to trigger overdrive
)
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
