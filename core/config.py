"""Configuration module housing all application configuration state.

This module defines standard typed configuration classes that serve
as the single source of truth for configuration across Aegis-X.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from dotenv import load_dotenv
from typing import Optional  

from utils.thresholds import (
    WEIGHT_SIGLIP, WEIGHT_UNIVFD, WEIGHT_XCEPTION, WEIGHT_SBI, WEIGHT_FREQNET, WEIGHT_RPPG,
    WEIGHT_DCT, WEIGHT_GEOMETRY, WEIGHT_ILLUMINATION,
    REAL_THRESHOLD, FAKE_THRESHOLD, EARLY_STOP_CONFIDENCE,
    SBI_SKIP_UNIVFD_THRESHOLD,
    XCEPTION_CONFIDENCE_BASE, XCEPTION_CONFIDENCE_MULTIPLIER, XCEPTION_MATCH_RATIO_MIN, XCEPTION_PARTIAL_LOAD_CAP,
    FREQNET_FAD_MID_EXCESS, FREQNET_FAD_HIGH_EXCESS, FREQNET_FAD_MID_MULTIPLIER, FREQNET_FAD_HIGH_MULTIPLIER,
    FREQNET_NEURAL_WEIGHT, FREQNET_FAD_WEIGHT
)

# Load environment variables early
load_dotenv()

@dataclass
class ModelPaths:
    """Paths to models and weights used by the system."""
    # Constructed using pathlib but cast to str to match specifications
    phi3_model: str = str(Path(os.getenv("AEGIS_MODEL_DIR", "models/")) / "phi3")
    univfd_backbone_dir: str = str(Path(os.getenv("AEGIS_MODEL_DIR", "models/")) / "clip-vit-large-patch14")
    univfd_probe_path: str = str(Path(os.getenv("AEGIS_MODEL_DIR", "models/")) / "univfd" / "probe.pth")
    xception_weights: str = str(Path(os.getenv("AEGIS_MODEL_DIR", "models/")) / "xception" / "xception_deepfake.pth")
    sbi_weights: str = str(Path(os.getenv("AEGIS_MODEL_DIR", "models/")) / "sbi" / "efficientnet_b4.pth")
    freqnet_weights: str = str(Path(os.getenv("AEGIS_MODEL_DIR", "models/")) / "freqnet" / "cnndetect_resnet50.pth")
    clip_adapter_weights: str = str(Path(os.getenv("AEGIS_MODEL_DIR", "models/")) / "clip_adapter.pt")

@dataclass
class AgentConfig:
    """Configuration for the local LLM agent and orchestration."""
    max_retries: int = int(os.getenv("AGENT_MAX_RETRIES", "2"))
    llm_timeout: int = int(os.getenv("AGENT_LLM_TIMEOUT", "120"))
    ollama_endpoint: str = os.getenv("OLLAMA_ENDPOINT", "http://localhost:11434")
    
    # === OLLAMA SPECIFIC ===
    ollama_model_name: str = os.getenv("OLLAMA_MODEL", "phi3:mini")
    ollama_timeout: int = int(os.getenv("OLLAMA_TIMEOUT", "120"))
    # C1: Omit for Ollama default (5min) - best for batch scanning
    ollama_keep_alive: Optional[int] = None
    
    # === OPENROUTER SPECIFIC ===
    use_openrouter: bool = False  # Runtime flag set by API endpoint
    openrouter_api_key: str = os.getenv("OPENROUTER_API_KEY", "")
    openrouter_model: str = os.getenv("OPENROUTER_MODEL", "qwen/qwen3.6-plus:free")

    
    # === GENERATION SETTINGS ===
    llm_temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.1"))
    llm_seed: int = int(os.getenv("LLM_SEED", "42"))
    llm_max_tokens: int = int(os.getenv("LLM_MAX_TOKENS", "1024"))
    llm_context_window: int = int(os.getenv("LLM_CONTEXT_WINDOW", "4096"))

@dataclass
class EnsembleWeights:
    """Weights for the final ensemble verdict calculation."""
    clip_adapter: float = WEIGHT_UNIVFD
    univfd: float = WEIGHT_UNIVFD
    xception: float = WEIGHT_XCEPTION
    sbi: float = WEIGHT_SBI
    freqnet: float = WEIGHT_FREQNET
    rppg: float = WEIGHT_RPPG
    dct: float = WEIGHT_DCT
    geometry: float = WEIGHT_GEOMETRY
    illumination: float = WEIGHT_ILLUMINATION

@dataclass
class ThresholdConfig:
    """Thresholds for verdicts and tool routing."""
    real_threshold: float = REAL_THRESHOLD
    fake_threshold: float = FAKE_THRESHOLD
    early_stop_confidence: float = EARLY_STOP_CONFIDENCE
    sbi_skip_univfd_threshold: float = SBI_SKIP_UNIVFD_THRESHOLD

@dataclass
class PreprocessingConfig:
    """Configuration for media preprocessing and patching."""
    face_crop_size: int = 224
    sbi_crop_size: int = 380
    native_patch_size: int = 224
    max_video_frames: int = 300
    min_video_frames: int = 90
    extract_fps: int = 30
    video_backend: str = "auto"
    quality_snipe_enabled: bool = True
    quality_snipe_samples: int = 5

@dataclass
class XceptionConfig:
    confidence_base: float = XCEPTION_CONFIDENCE_BASE
    confidence_multiplier: float = XCEPTION_CONFIDENCE_MULTIPLIER
    match_ratio_min: float = XCEPTION_MATCH_RATIO_MIN
    partial_load_cap: float = XCEPTION_PARTIAL_LOAD_CAP

@dataclass
class FreqNetFusionConfig:
    mid_excess_threshold: float = FREQNET_FAD_MID_EXCESS
    high_excess_threshold: float = FREQNET_FAD_HIGH_EXCESS
    mid_multiplier: float = FREQNET_FAD_MID_MULTIPLIER
    high_multiplier: float = FREQNET_FAD_HIGH_MULTIPLIER
    neural_weight: float = FREQNET_NEURAL_WEIGHT
    fad_weight: float = FREQNET_FAD_WEIGHT

@dataclass
class AegisConfig:
    """Master configuration class grouping all subsystem configs."""
    models: ModelPaths = field(default_factory=ModelPaths)
    agent: AgentConfig = field(default_factory=AgentConfig)
    weights: EnsembleWeights = field(default_factory=EnsembleWeights)
    thresholds: ThresholdConfig = field(default_factory=ThresholdConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    xception: XceptionConfig = field(default_factory=XceptionConfig)
    freqnet_fusion: FreqNetFusionConfig = field(default_factory=FreqNetFusionConfig)

