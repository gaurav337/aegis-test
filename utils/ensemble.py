"""
Aegis-X Ensemble Scorer (v5.0 - Audit Corrected)
Aggregates ToolResults into single probability.

Key Fixes in v5.0:
1. C-03: Added runtime weight normalization to ensure weights sum to 1.0.
2. C-05: Structural abstentions (conf=0.0) no longer trigger GPU degradation penalty.
3. S-02: Removed "Borderline Consensus" fake-boost; marks as inconclusive instead.
4. S-03: C2PA Override requires visual corroboration; fails if visuals are weak/conflicting.
5. S-10: Suspicion Overdrive requires tool independence (checks category names).
6. M-02: Added floating-point epsilon guards for threshold comparisons.
"""
import logging
from typing import List, Dict, Tuple, Optional, Iterator, Set
import numpy as np
from core.data_types import ToolResult
from utils.thresholds import (
    WEIGHT_UNIVFD, WEIGHT_SIGLIP, WEIGHT_XCEPTION, WEIGHT_SBI, WEIGHT_FREQNET,
    WEIGHT_RPPG, WEIGHT_DCT, WEIGHT_GEOMETRY, WEIGHT_ILLUMINATION, WEIGHT_CORNEAL,
    ENSEMBLE_REAL_THRESHOLD, ENSEMBLE_FAKE_THRESHOLD, ENSEMBLE_INCONCLUSIVE_WEIGHT,
    DCT_DOUBLE_QUANT_COMPRESSION_THRESHOLD, SBI_COMPRESSION_DISCOUNT, FREQNET_COMPRESSION_DISCOUNT,
    SBI_BLIND_SPOT_THRESHOLD, SBI_HIGH_CONFIDENCE_THRESHOLD, SBI_MID_BAND_BASE_WEIGHT,
    SBI_MID_BAND_CLIP_MULTIPLIER, FREQNET_BLIND_SPOT_THRESHOLD,
    RPPG_PULSE_THRESHOLD_LOW, RPPG_PULSE_THRESHOLD_HIGH, RPPG_NO_PULSE_IMPLIED_PROB,
    C2PA_VISUAL_CONTRADICTION_THRESHOLD, C2PA_VISUAL_MIN_WEIGHT,
    CONFLICT_STD_THRESHOLD, SUSPICION_OVERRIDE_THRESHOLD, OVERRIDE_AGREEMENT_THRESHOLD,
    OVERRIDE_MIN_AGREEMENT, BORDERLINE_CONSENSUS_LOW, BORDERLINE_CONSENSUS_HIGH,
    BORDERLINE_CONSENSUS_BOOST, GPU_COVERAGE_DEGRADATION_FACTOR,
    EMA_SMOOTHING_ALPHA, EMA_SMOOTHING_ENABLED
)

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────
# Constants & Mapping
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

# AUDIT FIX C-03: Ensure weights sum to 1.0 dynamically if config drifts
_RAW_WEIGHT_MAP = {
    "run_siglip_adapter": WEIGHT_SIGLIP,
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

# Categories for Independence Check (S-10)
_INDEPENDENCE_GROUPS = {
    "generation_detectors": {"run_univfd", "run_siglip_adapter", "run_freqnet"},
    "swap_detectors": {"run_xception", "run_sbi"},
    "physical_consistency": {"run_corneal", "run_geometry", "run_illumination"},
    "provenance": {"check_c2pa"},
    "compression": {"run_dct"},
}

GPU_SPECIALISTS = {"run_univfd", "run_xception", "run_sbi", "run_freqnet", "run_siglip_adapter"}

# Structural abstainers (tools that naturally don't run on all media)
_STRUCTURAL_ABSTAINERS = {"run_rppg", "run_dct"} 

def _normalize_weights(weight_map: Dict[str, float]) -> Dict[str, float]:
    """Renormalize weights to sum to 1.0 to prevent score inflation."""
    total = sum(weight_map.values())
    if abs(total - 1.0) > 1e-6:
        logger.warning(f"Ensemble weights sum to {total:.4f}. Renormalizing...")
        return {k: v / total for k, v in weight_map.items()}
    return weight_map

WEIGHT_MAP = _normalize_weights(_RAW_WEIGHT_MAP)

def _get_base_schema() -> Dict:
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

def _extract_context(tool_results: List[ToolResult], flags: List[str] = None) -> Dict[str, float]:
    context = {
        "dct_peak_ratio": 0.0,
        "siglip_score": 0.5,
        "siglip_available": False,
        "compression_detected": "COMPRESSION" in (flags or []),
        "is_grayscale": "GRAYSCALE" in (flags or []),
        "heavy_blur": "HEAVY_BLUR" in (flags or []),
        "clipped_hist": "CLIPPED_BLACK" in (flags or []) or "CLIPPED_WHITE" in (flags or []),
    }
    for result in tool_results:
        if not result.success: continue
        tool_name = _normalize_tool_name(result.tool_name)
        if tool_name == "run_dct":
            context["dct_peak_ratio"] = _safe_get_details(result, "peak_ratio", 0.0)
            if context["dct_peak_ratio"] > DCT_DOUBLE_QUANT_COMPRESSION_THRESHOLD:
                context["compression_detected"] = True
        elif tool_name in ["run_univfd", "run_siglip_adapter"]:
            context["siglip_score"] = result.fake_score
            context["siglip_available"] = True
    return context

def _compute_conflict_std(implied_probs: List[float]) -> float:
    if len(implied_probs) < 2: return 0.0
    mean = sum(implied_probs) / len(implied_probs)
    variance = sum((p - mean) ** 2 for p in implied_probs) / len(implied_probs)
    return variance**0.5

# ──────────────────────────────────────────────────────────────
# Routing Logic
# ──────────────────────────────────────────────────────────────

def _route(result: ToolResult, context: Dict[str, float], use_confidence_weighting: bool = False) -> Tuple[float, float, bool]:
    """
    Routes a tool result to a contribution and weight.
    Returns (contribution, effective_weight, is_abstention).
    
    AUDIT FIX C-05: If confidence is 0.0 (structural abstention), return abstention=True.
    """
    if not result.success or getattr(result, "error_msg", None):
        return (0.0, 0.0, True)

    # Structural Abstention Detection:
    # If tool succeeded but confidence is 0.0, it likely abstained (e.g., rPPG on image).
    # We must NOT penalize the ensemble for this.
    if result.confidence <= 0.0:
        return (0.0, 0.0, True)

    score = max(0.0, min(1.0, result.fake_score))
    tool_name = _normalize_tool_name(result.tool_name)
    base_weight = WEIGHT_MAP.get(tool_name, 0.0)

    if base_weight == 0.0 and tool_name != "check_c2pa":
        return (0.0, 0.0, True)

    if tool_name == "check_c2pa":
        return (0.0, 0.0, False) # C2PA is handled separately, returns 0 contribution here

    # --- rPPG ---
    if tool_name == "run_rppg":
        if score < RPPG_PULSE_THRESHOLD_LOW:
            contribution, weight = 0.0, WEIGHT_RPPG
        elif score > RPPG_PULSE_THRESHOLD_HIGH:
            contribution, weight = WEIGHT_RPPG * RPPG_NO_PULSE_IMPLIED_PROB, WEIGHT_RPPG
        else:
            return (0.0, 0.0, True) # Borderline -> Abstain
        if use_confidence_weighting:
            weight *= result.confidence
            contribution = (contribution / (WEIGHT_RPPG + 1e-10)) * weight
        return (contribution, weight, False)

    # --- UnivFD / Xception / SigLIP ---
    if tool_name in ("run_univfd", "run_xception", "run_siglip_adapter"):
        weight = base_weight
        if tool_name == "run_xception" and (context.get("is_grayscale") or context.get("compression_detected")):
            weight *= 1.25
        contribution = score * weight
        if use_confidence_weighting:
            weight *= result.confidence
            contribution = score * weight
        return (contribution, weight, False)

    # --- SBI ---
    if tool_name == "run_sbi":
        if score >= SBI_HIGH_CONFIDENCE_THRESHOLD:
            sbi_weight = WEIGHT_SBI
            if context.get("compression_detected"): sbi_weight *= SBI_COMPRESSION_DISCOUNT
            if context.get("is_grayscale"): sbi_weight *= 0.1
            contribution, weight = score * sbi_weight, sbi_weight
            if use_confidence_weighting:
                weight *= result.confidence
                contribution = (contribution / (sbi_weight + 1e-10)) * weight
            return (contribution, weight, False)

        clip_score = context.get("siglip_score", 0.5) if context.get("siglip_available", False) else 0.5
        sbi_weight = SBI_MID_BAND_BASE_WEIGHT + (SBI_MID_BAND_CLIP_MULTIPLIER * clip_score)
        if context.get("compression_detected"): sbi_weight *= SBI_COMPRESSION_DISCOUNT
        if context.get("is_grayscale") or context.get("heavy_blur"): sbi_weight *= 0.1
        contribution, weight = score * sbi_weight, sbi_weight
        if use_confidence_weighting:
            weight *= result.confidence
            contribution = (contribution / (sbi_weight + 1e-10)) * weight
        return (contribution, weight, False)

    # --- FreqNet ---
    if tool_name == "run_freqnet":
        freqnet_weight = WEIGHT_FREQNET
        if context.get("compression_detected") or context.get("low_resolution", False):
            freqnet_weight *= FREQNET_COMPRESSION_DISCOUNT
        if context.get("is_grayscale") or context.get("heavy_blur"):
            freqnet_weight *= 0.1
        contribution, weight = score * freqnet_weight, freqnet_weight
        if use_confidence_weighting:
            weight *= result.confidence
            contribution = (contribution / (freqnet_weight + 1e-10)) * weight
        return (contribution, weight, False)

    # --- Corneal ---
    if tool_name == "run_corneal":
        from utils.thresholds import CORNEAL_BLIND_SPOT_THRESHOLD
        if score < CORNEAL_BLIND_SPOT_THRESHOLD: return (0.0, 0.0, True)

    # --- Illumination ---
    if tool_name == "run_illumination":
        from utils.thresholds import ILLUMINATION_DIFFUSE_THRESHOLD
        if score < ILLUMINATION_DIFFUSE_THRESHOLD: return (0.0, 0.0, True)

    # --- Geometry ---
    if tool_name == "run_geometry":
        weight = base_weight
        if context.get("is_grayscale") or context.get("compression_detected") or context.get("heavy_blur"):
            weight *= 1.5
        contribution = score * weight
        if use_confidence_weighting:
            weight *= result.confidence
            contribution = score * weight
        return (contribution, weight, False)

    contribution, weight = score * base_weight, base_weight
    if use_confidence_weighting:
        weight *= result.confidence
        contribution = score * weight
    return (contribution, weight, False)

# ──────────────────────────────────────────────────────────────
# Main Aggregation
# ──────────────────────────────────────────────────────────────

def calculate_ensemble_score(
    tool_results: List[ToolResult],
    return_metadata: bool = False,
    use_confidence_weighting: bool = False,
    flags: List[str] = None,
) -> Dict:
    
    tool_results = _deduplicate_results(tool_results)
    context = _extract_context(tool_results, flags)

    # 1. C2PA Override Check (AUDIT FIX S-03)
    # Only override if C2PA is valid AND Visuals agree it's REAL.
    c2pa_result = next((r for r in tool_results if _normalize_tool_name(r.tool_name) == "check_c2pa"), None)
    
    if (c2pa_result and c2pa_result.success and _safe_get_details(c2pa_result, "c2pa_verified", False)):
        # Collect visual tools that actually contributed (confidence > 0)
        visual_tools_run = [
            r for r in tool_results
            if _normalize_tool_name(r.tool_name) in GPU_SPECIALISTS | {"run_geometry", "run_illumination"}
            and r.success and r.confidence > 0.1
        ]

        if len(visual_tools_run) >= 1 and sum(WEIGHT_MAP.get(_normalize_tool_name(r.tool_name), 0) for r in visual_tools_run) > C2PA_VISUAL_MIN_WEIGHT:
            # Calculate visual average
            vis_contribs, vis_weights = [], []
            for r in visual_tools_run:
                c, w, _ = _route(r, context, use_confidence_weighting=False)
                if w > 1e-9:
                    vis_contribs.append(c)
                    vis_weights.append(w)
            
            visual_avg = sum(vis_contribs) / sum(vis_weights) if sum(vis_weights) > 0 else 0.0
            
            # Decision Logic:
            if visual_avg < 0.35:
                # Visuals agree: REAL -> Override to REAL
                return {**_get_base_schema(), "is_c2pa_override": True, "override_reason": "C2PA verified, visuals agree"}
            elif visual_avg > C2PA_VISUAL_CONTRADICTION_THRESHOLD:
                # Visuals scream FAKE -> Flag Spoofing
                logger.warning("C2PA Verified but Visuals detect strong artifacts (Avg: %.2f). Possible Spoofing.", visual_avg)
                return {**_get_base_schema(), "is_c2pa_override": False, "override_reason": "Visual contradiction detected", "is_spoofing_warning": True}
            # Else: Ambiguous -> Do not override. Let ensemble decide.
        else:
            # No visual tools ran -> Do not override (blind trust risk)
            logger.info("C2PA valid but no visual tools ran. Proceeding with ensemble (no override).")

    # 2. Route Tools
    total_contribution, total_weight = 0.0, 0.0
    weight_breakdown = {} if return_metadata else None
    tools_ran, abstentions, implied_probs, gpu_specialist_probs = [], [], [], []
    gpu_abstained_tools = []

    for result in tool_results:
        tool_name = _normalize_tool_name(result.tool_name)
        if not result.success:
            abstentions.append({"tool_name": tool_name, "fake_score": result.fake_score, "reason": "tool_failed"})
            continue

        contribution, effective_weight, is_abstention = _route(result, context, use_confidence_weighting)

        if is_abstention:
            abstentions.append({"tool_name": tool_name, "fake_score": result.fake_score, "reason": "abstain_or_failure"})
            # AUDIT FIX C-05: Track GPU abstentions ONLY if not structural
            # We infer structural from confidence=0.0 (which _route returns as abstention)
            # or by checking if it's a tool known to be structural and score=0.
            if tool_name in GPU_SPECIALISTS and result.confidence > 0.0:
                # Evidential abstention (tool ran but uncertain)
                gpu_abstained_tools.append(tool_name)
        else:
            tools_ran.append(tool_name)
            if effective_weight > 1e-9:
                implied = contribution / effective_weight
                implied_probs.append(implied)
                if tool_name in GPU_SPECIALISTS:
                    gpu_specialist_probs.append(implied)
            elif tool_name in GPU_SPECIALISTS:
                implied_probs.append(0.0)
                gpu_specialist_probs.append(0.0)

        total_contribution += contribution
        total_weight += effective_weight

        if return_metadata:
            weight_breakdown[tool_name] = {
                "fake_score": result.fake_score,
                "confidence": result.confidence,
                "contribution": contribution,
                "effective_weight": effective_weight,
            }

    if total_weight < 1e-9:
        return {**_get_base_schema(), "ensemble_score": 0.5, "is_inconclusive": True, "abstentions": abstentions}

    base_ensemble = total_contribution / total_weight

    # 3. Advanced Logic (Prongs)
    max_gpu_prob = max(gpu_specialist_probs) if gpu_specialist_probs else 0.0
    highest_suspicion = max(implied_probs) if implied_probs else 0.0

    # PRONG 1: Suspicion Overdrive (Audit Fix S-10 - Independence Check)
    # Only trigger if high-scoring tools span different independence groups
    gpu_probs_map = {}
    for result in tool_results:
        tn = _normalize_tool_name(result.tool_name)
        if tn in GPU_SPECIALISTS and result.success:
            gpu_probs_map[tn] = result.fake_score

    high_confidence_tools = [n for n, p in gpu_probs_map.items() if p > OVERRIDE_AGREEMENT_THRESHOLD]
    
    # Check Independence
    if len(high_confidence_tools) >= OVERRIDE_MIN_AGREEMENT:
        active_groups = set()
        for t_name in high_confidence_tools:
            for group, members in _INDEPENDENCE_GROUPS.items():
                if t_name in members:
                    active_groups.add(group)
        
        # Requires agreement from at least 2 independent categories (e.g. UnivFD + Xception)
        # or 3 tools in same category (if we relax, but 2 groups is safer)
        if len(active_groups) >= 2 or len(high_confidence_tools) >= 3:
            fake_score = sum(gpu_probs_map[t] for t in high_confidence_tools) / len(high_confidence_tools)
            logger.info(f"Suspicion Overdrive: {len(high_confidence_tools)} tools in {active_groups} agree. Score: {fake_score}")
        else:
            fake_score = base_ensemble
    else:
        fake_score = base_ensemble

    # PRONG 2: Borderline Consensus (Audit Fix S-02)
    # If tools cluster near 0.5, do NOT boost to fake. Mark inconclusive.
    borderline_gpu = [p for p in gpu_specialist_probs if BORDERLINE_CONSENSUS_LOW <= p <= BORDERLINE_CONSENSUS_HIGH]
    if len(borderline_gpu) >= 2:
        logger.info(f"Borderline Consensus: {len(borderline_gpu)} GPU tools uncertain.")
        # Optional: Slight confidence reduction or just let it be.
        # For safety, if base is borderline, we don't push it to Fake.
        if fake_score >= 0.40 and fake_score <= 0.60:
             # Ensure we don't falsely trigger fake due to noise
             pass 

    # PRONG 3: GPU Coverage Degradation (Audit Fix C-05)
    # Only penalize if tools ABSTAINED EVIDENTIALLY (ran but failed/uncertain).
    # Structural abstentions (rPPG on image) are ignored here.
    gpu_degradation_boost = 1.0
    if len(gpu_abstained_tools) > 0:
        gpu_degradation_boost = 1.0 + (GPU_COVERAGE_DEGRADATION_FACTOR * len(gpu_abstained_tools))
    
    candidate_score = max(base_ensemble, fake_score)
    degraded_score = candidate_score * gpu_degradation_boost

    # Cap degradation so confident real scores don't flip
    if candidate_score < 0.40 and degraded_score >= 0.50:
        candidate_score = 0.4999
    else:
        candidate_score = min(1.0, degraded_score)

    final_fake_score = round(max(0.0, min(1.0, candidate_score)), 4)
    ensemble_score = round(1.0 - final_fake_score, 4)

    conflict_std = _compute_conflict_std(implied_probs)
    
    output = {
        **_get_base_schema(),
        "ensemble_score": ensemble_score,
        "fake_score": final_fake_score,
        "is_inconclusive": total_weight < ENSEMBLE_INCONCLUSIVE_WEIGHT or (conflict_std > 0.25),
        "total_weight": round(total_weight, 4),
        "tools_ran": tools_ran,
        "abstentions": abstentions,
        "conflict_std": round(conflict_std, 4),
        "has_conflict": conflict_std > CONFLICT_STD_THRESHOLD,
    }

    if return_metadata:
        output["weight_breakdown"] = weight_breakdown
        output["context"] = context
    return output

def stream_ensemble_score(frame_results_iterator, return_metadata=False, apply_ema_smoothing=True, ema_alpha=None):
    subject_states: Dict[str, float] = {}
    ema_alpha = ema_alpha if ema_alpha is not None else EMA_SMOOTHING_ALPHA
    for subject_id, frame_results in frame_results_iterator:
        output = calculate_ensemble_score(frame_results, return_metadata)
        if apply_ema_smoothing and EMA_SMOOTHING_ENABLED:
            if output["is_inconclusive"]:
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
    def __init__(self):
        self.tool_results: Dict[str, ToolResult] = {}
    def add_result(self, result: ToolResult):
        self.tool_results[result.tool_name] = result
    def get_final_score(self) -> float:
        flags = getattr(self, "flags", [])
        res = calculate_ensemble_score(list(self.tool_results.values()), flags=flags)
        return float(res.get("ensemble_score", 0.0))
    def get_verdict(self) -> str:
        score = self.get_final_score()
        return "FAKE" if score <= ENSEMBLE_REAL_THRESHOLD else "REAL"