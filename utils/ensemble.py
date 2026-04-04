"""
Aegis-X Ensemble Scorer (v4.0 - Production Final)
==================================================
Aggregates ToolResults into single probability.

VRAM Guarantees:
  - 0 MB GPU memory (CPU-only computation)
  - No tensor caching (results are plain dataclasses)
  - Subject-aware stream processing with temporal EMA smoothing

Memory Budget:
  - CPU RAM: ~6 KB per ensemble call + (~8 bytes per tracked subject)
  - VRAM: 0 MB

Addresses: 69 Total Issues (Including Temporal Ghosting & Float Precision)
"""

import logging
from typing import List, Dict, Tuple, Optional, Iterator

from core.data_types import ToolResult

from utils.thresholds import (
    # Tool weights
    WEIGHT_UNIVFD, WEIGHT_XCEPTION, WEIGHT_SIGLIP, WEIGHT_SBI, WEIGHT_FREQNET,
    WEIGHT_RPPG, WEIGHT_DCT, WEIGHT_GEOMETRY,
    WEIGHT_ILLUMINATION, WEIGHT_CORNEAL,
    
    # Ensemble decision thresholds
    ENSEMBLE_REAL_THRESHOLD,
    ENSEMBLE_FAKE_THRESHOLD,
    ENSEMBLE_INCONCLUSIVE_WEIGHT,
    
    # Compression discounts
    DCT_DOUBLE_QUANT_COMPRESSION_THRESHOLD,
    SBI_COMPRESSION_DISCOUNT,
    FREQNET_COMPRESSION_DISCOUNT,
    
    # Tool thresholds
    SBI_BLIND_SPOT_THRESHOLD,
    SBI_HIGH_CONFIDENCE_THRESHOLD,
    SBI_MID_BAND_BASE_WEIGHT,
    SBI_MID_BAND_CLIP_MULTIPLIER,
    FREQNET_BLIND_SPOT_THRESHOLD,
    RPPG_PULSE_THRESHOLD_LOW,
    RPPG_PULSE_THRESHOLD_HIGH,
    RPPG_NO_PULSE_IMPLIED_PROB,
    
    # C2PA thresholds
    C2PA_VISUAL_CONTRADICTION_THRESHOLD,
    C2PA_VISUAL_MIN_WEIGHT,
    
    # Conflict detection
    CONFLICT_STD_THRESHOLD,
    SUSPICION_OVERRIDE_THRESHOLD,
    
    # Borderline consensus & GPU coverage
    BORDERLINE_CONSENSUS_LOW,
    BORDERLINE_CONSENSUS_HIGH,
    BORDERLINE_CONSENSUS_BOOST,
    GPU_COVERAGE_DEGRADATION_FACTOR,
    
    # EMA smoothing (2026 Edge Standard)
    EMA_SMOOTHING_ALPHA,
    EMA_SMOOTHING_ENABLED,
)

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────
# Tool name mapping
# ──────────────────────────────────────────────────────────────
TOOL_NAME_MAP = {
    "run_clip_adapter": "run_siglip_adapter",
    "run_siglip_adapter": "run_siglip_adapter",
    "run_univfd": "run_univfd",
    "run_xception": "run_xception",
    "run_sbi": "run_sbi",
    "run_freqnet": "run_freqnet",
    "run_dct": "run_dct",
    "run_geometry": "run_geometry",
    "run_illumination": "run_illumination",
    "run_corneal": "run_corneal",
    "run_rppg": "run_rppg",
    "check_c2pa": "check_c2pa",
}

# Weight mapping
WEIGHT_MAP = {
    "run_siglip_adapter": WEIGHT_SIGLIP,
    "run_clip_adapter": WEIGHT_SIGLIP,
    "run_univfd": WEIGHT_UNIVFD,
    "run_xception": WEIGHT_XCEPTION,
    "run_sbi": WEIGHT_SBI,
    "run_freqnet": WEIGHT_FREQNET,
    "run_rppg": WEIGHT_RPPG,
    "run_dct": WEIGHT_DCT,
    "run_geometry": WEIGHT_GEOMETRY,
    "run_illumination": WEIGHT_ILLUMINATION,
    "run_corneal": WEIGHT_CORNEAL,
}


def _get_base_schema() -> Dict:
    """Generate fresh schema dict to prevent state mutation."""
    return {
        "ensemble_score": 0.0,
        "is_c2pa_override": False,
        "is_inconclusive": False,
        "total_weight": 0.0,
        "weight_breakdown": {},
        "tools_ran": [],
        "context": {},
        "abstentions": [],
        "conflict_std": 0.0,
        "has_conflict": False,
    }


def _normalize_tool_name(name: str) -> str:
    return TOOL_NAME_MAP.get(name, name)


def _deduplicate_results(tool_results: List[ToolResult]) -> List[ToolResult]:
    seen: Dict[str, ToolResult] = {}
    for r in tool_results:
        tool_name = _normalize_tool_name(r.tool_name)
        existing = seen.get(tool_name)
        if existing is None or r.success or not existing.success:
            seen[tool_name] = r
    return list(seen.values())


def _safe_get_details(result: ToolResult, key: str, default=None):
    return (result.details or {}).get(key, default)


def _extract_context(tool_results: List[ToolResult]) -> Dict[str, float]:
    """FIX #68: Single-pass O(N) context extraction."""
    context = {
        "dct_peak_ratio": 0.0,
        "siglip_score": 0.5,
        "siglip_available": False,
        "compression_detected": False,
    }
    
    for result in tool_results:
        if not result.success:
            continue
        
        tool_name = _normalize_tool_name(result.tool_name)
        
        if tool_name == "run_dct":
            context["dct_peak_ratio"] = _safe_get_details(result, "peak_ratio", 0.0)
            context["compression_detected"] = (
                context["dct_peak_ratio"] > DCT_DOUBLE_QUANT_COMPRESSION_THRESHOLD
            )
        elif tool_name in ["run_univfd", "run_siglip_adapter", "run_clip_adapter"]:
            context["siglip_score"] = result.fake_score
            context["siglip_available"] = True
            
    return context


def _compute_conflict_std(implied_probs: List[float]) -> float:
    if len(implied_probs) < 2:
        return 0.0
    mean = sum(implied_probs) / len(implied_probs)
    variance = sum((p - mean) ** 2 for p in implied_probs) / len(implied_probs)
    return variance ** 0.5


def _route(result: ToolResult, context: Dict[str, float],
           use_confidence_weighting: bool = False) -> Tuple[float, float]:
    
    if not result.success or getattr(result, "error_msg", None):
        return (0.0, 0.0)
    if result.confidence <= 0.0:
        return (0.0, 0.0)
    
    score = max(0.0, min(1.0, result.fake_score))
    tool_name = _normalize_tool_name(result.tool_name)
    base_weight = WEIGHT_MAP.get(tool_name, 0.0)
    
    if base_weight == 0.0 and tool_name != "check_c2pa":
        logger.warning(f"Unknown tool '{tool_name}' has no weight mapping — abstaining.")
        return (0.0, 0.0)
    if tool_name == "check_c2pa":
        return (0.0, 0.0)
    
    # rPPG
    if tool_name == "run_rppg":
        if score < RPPG_PULSE_THRESHOLD_LOW:
            contribution, weight = 0.0, WEIGHT_RPPG
        elif score > RPPG_PULSE_THRESHOLD_HIGH:
            contribution, weight = WEIGHT_RPPG * RPPG_NO_PULSE_IMPLIED_PROB, WEIGHT_RPPG
        else:
            return (0.0, 0.0)
            
        if use_confidence_weighting:
            weight *= result.confidence
            contribution = (contribution / (WEIGHT_RPPG + 1e-10)) * weight
        return (contribution, weight)
    
    # UnivFD & Xception
    if tool_name in ("run_univfd", "run_xception", "run_siglip_adapter"):
        contribution, weight = score * base_weight, base_weight
        if use_confidence_weighting:
            weight *= result.confidence
            contribution = score * weight
        return (contribution, weight)
    
    # SBI
    if tool_name == "run_sbi":
        if score < SBI_BLIND_SPOT_THRESHOLD:
            return (0.0, 0.0)
        
        if score >= SBI_HIGH_CONFIDENCE_THRESHOLD:
            sbi_weight = WEIGHT_SBI
            if context.get("compression_detected", False):
                sbi_weight *= SBI_COMPRESSION_DISCOUNT
            
            contribution, weight = score * sbi_weight, sbi_weight
            if use_confidence_weighting:
                weight *= result.confidence
                contribution = (contribution / (sbi_weight + 1e-10)) * weight
            return (contribution, weight)
        
        # Mid-band
        clip_score = context.get("siglip_score", 0.5) if context.get("siglip_available", False) else 0.5
        sbi_weight = SBI_MID_BAND_BASE_WEIGHT + (SBI_MID_BAND_CLIP_MULTIPLIER * clip_score)
        
        if context.get("compression_detected", False):
            sbi_weight *= SBI_COMPRESSION_DISCOUNT
            
        contribution, weight = score * sbi_weight, sbi_weight
        if use_confidence_weighting:
            weight *= result.confidence
            contribution = (contribution / (sbi_weight + 1e-10)) * weight
        return (contribution, weight)
    
    # FreqNet
    if tool_name == "run_freqnet":
        if score < FREQNET_BLIND_SPOT_THRESHOLD:
            return (0.0, 0.0)
        
        freqnet_weight = WEIGHT_FREQNET
        if context.get("compression_detected", False):
            freqnet_weight *= FREQNET_COMPRESSION_DISCOUNT
            
        contribution, weight = score * freqnet_weight, freqnet_weight
        if use_confidence_weighting:
            weight *= result.confidence
            contribution = (contribution / (freqnet_weight + 1e-10)) * weight
        return (contribution, weight)
    
    # CPU Tools
    contribution, weight = score * base_weight, base_weight
    if use_confidence_weighting:
        weight *= result.confidence
        contribution = score * weight
    return (contribution, weight)


def calculate_ensemble_score(
    tool_results: List[ToolResult],
    return_metadata: bool = False,
    use_confidence_weighting: bool = False,
) -> Dict:
    tool_results = _deduplicate_results(tool_results)
    
    # ──────────────────────────────────────────────────────────────
    # Step 1: Extract Context (FIX #68: Single Pass O(N))
    # ──────────────────────────────────────────────────────────────
    context = _extract_context(tool_results)
    
    # ──────────────────────────────────────────────────────────────
    # Step 2: C2PA Override Check with Visual Corroboration
    # ──────────────────────────────────────────────────────────────
    c2pa_result = next((r for r in tool_results if _normalize_tool_name(r.tool_name) == "check_c2pa"), None)
    
    if c2pa_result and c2pa_result.success and _safe_get_details(c2pa_result, "c2pa_verified", False):
        visual_tools_run = [
            r for r in tool_results 
            if _normalize_tool_name(r.tool_name) in ["run_univfd", "run_xception", "run_siglip_adapter", "run_sbi", "run_freqnet"]
            and r.success and r.confidence > 0.5
        ]
        
        if visual_tools_run:
            visual_contributions, visual_weights = [], []
            for r in visual_tools_run:
                contrib, eff_weight = _route(r, context, use_confidence_weighting=False)
                # FIX #67: Micro-float precision guard
                if eff_weight > 1e-9:
                    visual_contributions.append(contrib)
                    visual_weights.append(eff_weight)
            
            total_visual_weight = sum(visual_weights)
            visual_avg = sum(visual_contributions) / total_visual_weight if total_visual_weight > 0 else 0.0
            
            if visual_avg > C2PA_VISUAL_CONTRADICTION_THRESHOLD and total_visual_weight > C2PA_VISUAL_MIN_WEIGHT:
                logger.warning("C2PA verified but visual models scream FAKE. Possible spoofing.")
            else:
                return {**_get_base_schema(), "is_c2pa_override": True, "override_reason": "C2PA verified"}
        else:
            return {**_get_base_schema(), "is_c2pa_override": True, "override_reason": "C2PA verified"}
    
    # ──────────────────────────────────────────────────────────────
    # Step 3: Route Each Tool
    # ──────────────────────────────────────────────────────────────
    total_contribution, total_weight = 0.0, 0.0
    weight_breakdown = {} if return_metadata else None
    tools_ran, abstentions, implied_probs, gpu_specialist_probs = [], [], [], []
    
    GPU_SPECIALISTS = {"run_univfd", "run_xception", "run_sbi", "run_freqnet"}
    
    for result in tool_results:
        tool_name = _normalize_tool_name(result.tool_name)
        
        if not result.success:
            abstentions.append({"tool_name": tool_name, "fake_score": result.fake_score, "reason": "tool_failed"})
            continue
        
        contribution, effective_weight = _route(result, context, use_confidence_weighting)
        
        # FIX #67: Micro-float precision guard for division
        if effective_weight < 1e-9:
            abstentions.append({"tool_name": tool_name, "fake_score": result.fake_score, "reason": "blind_spot"})
        else:
            tools_ran.append(tool_name)
            implied = contribution / effective_weight
            implied_probs.append(implied)
            if tool_name in GPU_SPECIALISTS:
                gpu_specialist_probs.append(implied)
        
        total_contribution += contribution
        total_weight += effective_weight
        
        if return_metadata:
            weight_breakdown[tool_name] = {
                "fake_score": result.fake_score, "confidence": result.confidence,
                "contribution": contribution, "effective_weight": effective_weight
            }
    
    # ──────────────────────────────────────────────────────────────
    # Step 4: Calculate Final Outputs
    # ──────────────────────────────────────────────────────────────
    if total_weight < 1e-9:
        return {**_get_base_schema(), "ensemble_score": 0.5, "is_inconclusive": True, "abstentions": abstentions}
    
    # Base weighted average combines both GPU specialists and CPU supporters
    base_ensemble = total_contribution / total_weight
    
    # ── PRONG 1: Suspicion Overdrive (Hard Max-Pool) ──
    max_gpu_prob = max(gpu_specialist_probs) if gpu_specialist_probs else 0.0
    highest_suspicion = max(implied_probs) if implied_probs else 0.0
    
    # Debug logging
    logger.debug("Ensemble trace: implied_probs=%s, max_gpu_prob=%.4f, highest_suspicion=%.4f, base_ensemble=%.4f", 
                 implied_probs, max_gpu_prob, highest_suspicion, base_ensemble)
    logger.debug("Ensemble trace: tools_ran=%s, abstentions=%s", tools_ran, [a['tool_name'] for a in abstentions])
    
    if max_gpu_prob > SUSPICION_OVERRIDE_THRESHOLD:
        # Forensic tools detect complementary failure modes (e.g. face-swap vs pure generative).
        # A high score from ONE reliable GPU specialist is sufficient evidence of manipulation.
        # Hard max-pooling for highest GPU certainty.
        fake_score = round(max_gpu_prob, 4)
        logger.info("Suspicion Overdrive FIRED: max_gpu_prob=%.4f > threshold=%.2f. Complementary tools OR logic active.", max_gpu_prob, SUSPICION_OVERRIDE_THRESHOLD)
    else:
        # ── PRONG 2: Borderline Consensus Detection ──
        # When ≥2 GPU specialists independently cluster near 50% (borderline zone),
        # their joint uncertainty is itself a corroborating signal of manipulation.
        # A single tool at 49% is a coin-flip; TWO tools at ~47% is a pattern.
        borderline_gpu_probs = [
            p for p in gpu_specialist_probs 
            if BORDERLINE_CONSENSUS_LOW <= p <= BORDERLINE_CONSENSUS_HIGH
        ]
        
        consensus_anchor = 0.0
        if len(borderline_gpu_probs) >= 2:
            consensus_mean = sum(borderline_gpu_probs) / len(borderline_gpu_probs)
            consensus_anchor = min(1.0, consensus_mean * BORDERLINE_CONSENSUS_BOOST)
            logger.info("Borderline Consensus FIRED: %d GPU specialists in [%.2f, %.2f] zone, "
                       "mean=%.4f, boosted=%.4f",
                       len(borderline_gpu_probs), BORDERLINE_CONSENSUS_LOW, 
                       BORDERLINE_CONSENSUS_HIGH, consensus_mean, consensus_anchor)
        
        # ── PRONG 3: GPU Coverage Degradation ──
        # When GPU specialists blind-spot out, the system has less evidence.
        # It should NOT confidently declare REAL on thin evidence.
        gpu_abstained = [a for a in abstentions if a["tool_name"] in GPU_SPECIALISTS]
        total_gpu_expected = len(GPU_SPECIALISTS)
        gpu_degradation_boost = 1.0
        if len(gpu_abstained) > 0:
            gpu_degradation_boost = 1.0 + (GPU_COVERAGE_DEGRADATION_FACTOR * len(gpu_abstained))
            logger.info("GPU Coverage Degradation: %d/%d specialists abstained, boost=%.2f",
                       len(gpu_abstained), total_gpu_expected, gpu_degradation_boost)
        
        # Anomaly anchor uses ONLY GPU specialist scores.
        # CPU tools (corneal, geometry, DCT) are noisy supporters — they participate
        # via the base weighted average but cannot unilaterally anchor the score.
        anomaly_anchor = max_gpu_prob
        
        # We no longer disable anomaly anchors due to "conflict". GPU specialists detect
        # orthogonal features (e.g., SBI for boundaries, UnivFD for generative GANs).
        # One tool firing while others are silent is an expected behavioral pattern.
        
        # Pick the strongest signal from all three sources
        candidate_score = max(base_ensemble, anomaly_anchor, consensus_anchor)
        
        # Apply GPU degradation boost (pushes fake_score UP when coverage is thin)
        candidate_score = min(1.0, candidate_score * gpu_degradation_boost)
        
        fake_score = round(max(0.0, min(1.0, candidate_score)), 4)
        
        if candidate_score > base_ensemble:
            logger.info("Score lifted from base=%.4f to %.4f (anchor=%.4f, consensus=%.4f, degradation=%.2f)",
                       base_ensemble, candidate_score, anomaly_anchor, consensus_anchor, gpu_degradation_boost)
    
    # Convert to REAL probability: how likely is this media authentic?
    # 0.0 = definitely fake, 1.0 = definitely real
    ensemble_score = round(1.0 - fake_score, 4)
        
    conflict_std = _compute_conflict_std(implied_probs)
    has_conflict = conflict_std > CONFLICT_STD_THRESHOLD
    
    output = {
        **_get_base_schema(),
        "ensemble_score": ensemble_score,  # Real probability (1.0 = authentic)
        "fake_score": fake_score,          # Preserved for internal reference
        "is_inconclusive": total_weight < ENSEMBLE_INCONCLUSIVE_WEIGHT,
        "total_weight": round(total_weight, 4),
        "tools_ran": tools_ran,
        "abstentions": abstentions,
        "conflict_std": round(conflict_std, 4),
        "has_conflict": has_conflict,
    }
    
    if return_metadata:
        output["weight_breakdown"] = weight_breakdown
        output["context"] = context
        
    return output


def stream_ensemble_score(
    frame_results_iterator: Iterator[Tuple[str, List[ToolResult]]],
    return_metadata: bool = False,
    apply_ema_smoothing: bool = True,
    ema_alpha: float = None,
) -> Iterator[Tuple[str, Dict]]:
    """
    FIX #69: Subject-Aware State Management & Scene Cut Hard-Reset
    Processes multiple faces dynamically without memory bloat or ghosting.
    """
    subject_states: Dict[str, float] = {}
    ema_alpha = ema_alpha if ema_alpha is not None else EMA_SMOOTHING_ALPHA
    
    for subject_id, frame_results in frame_results_iterator:
        output = calculate_ensemble_score(frame_results, return_metadata)
        
        if apply_ema_smoothing and EMA_SMOOTHING_ENABLED:
            if output["is_inconclusive"]:
                # Hard reset: Break temporal chain on scene cut or tracking loss
                subject_states.pop(subject_id, None)
            else:
                current_raw = output["ensemble_score"]
                prev_score = subject_states.get(subject_id)
                
                if prev_score is None:
                    subject_states[subject_id] = current_raw
                else:
                    smoothed = (ema_alpha * current_raw) + ((1.0 - ema_alpha) * prev_score)
                    output["ensemble_score"] = round(smoothed, 4)
                    subject_states[subject_id] = smoothed
                    
        yield subject_id, output

class EnsembleAggregator:
    """Stateful wrapper for ensemble calculations, used by the Agent."""
    def __init__(self):
        self.tool_results: Dict[str, ToolResult] = {}
        
    def add_result(self, result: ToolResult):
        self.tool_results[result.tool_name] = result
        
    def get_final_score(self) -> float:
        res = calculate_ensemble_score(list(self.tool_results.values()))
        return float(res.get("ensemble_score", 0.0))
        
    def get_verdict(self) -> str:
        score = self.get_final_score()
        # ensemble_score is now REAL probability (1.0 = authentic, 0.0 = fake)
        # FAKE if real probability is below the real threshold
        return "FAKE" if score <= ENSEMBLE_REAL_THRESHOLD else "REAL"
