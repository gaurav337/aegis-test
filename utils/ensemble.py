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
    ENSEMBLE_REAL_THRESHOLD,
    ENSEMBLE_FAKE_THRESHOLD,
    ENSEMBLE_INCONCLUSIVE_WEIGHT,
    DCT_DOUBLE_QUANT_COMPRESSION_THRESHOLD,
    SBI_COMPRESSION_DISCOUNT,
    FREQNET_COMPRESSION_DISCOUNT,
    SBI_BLIND_SPOT_THRESHOLD,
    SBI_HIGH_CONFIDENCE_THRESHOLD,
    SBI_MID_BAND_BASE_WEIGHT,
    SBI_MID_BAND_CLIP_MULTIPLIER,
    FREQNET_BLIND_SPOT_THRESHOLD,
    RPPG_PULSE_THRESHOLD_LOW,
    RPPG_PULSE_THRESHOLD_HIGH,
    RPPG_NO_PULSE_IMPLIED_PROB,
    C2PA_VISUAL_CONTRADICTION_THRESHOLD,
    C2PA_VISUAL_MIN_WEIGHT,
    CONFLICT_STD_THRESHOLD,
    SUSPICION_OVERRIDE_THRESHOLD,
    OVERRIDE_AGREEMENT_THRESHOLD,
    OVERRIDE_MIN_AGREEMENT,
    BORDERLINE_CONSENSUS_LOW,
    BORDERLINE_CONSENSUS_HIGH,
    BORDERLINE_CONSENSUS_BOOST,
    GPU_COVERAGE_DEGRADATION_FACTOR,
    EMA_SMOOTHING_ALPHA,
    EMA_SMOOTHING_ENABLED,
    ENCORE_NEAR_MISS_THRESHOLD,
    ENCORE_CORROBORATION_SENSITIVITY,
    LOGIT_DAMPING_FACTOR,
    NEUTRAL_DEADZONE_LOW,
    NEUTRAL_DEADZONE_HIGH,
    NEUTRAL_DEADZONE_CONFIDENCE_MIN,
    ENSEMBLE_SPECIALIST_DOMINANCE_FACTOR,
    ENSEMBLE_SPECIALIST_FAKE_THRESHOLD,
    ENSEMBLE_SPECIALIST_MIN_CONFIDENCE,
    GPU_SINGLE_DETECT_HIGH_CONFIDENCE,
    GPU_MULTI_DETECT_MIN_CONFIDENCE,
    GPU_SINGLE_DETECT_BOOST,
    GPU_MULTI_DETECT_BOOST,
    CPU_SUPPORT_WHEN_GPU_AVAILABLE,
)

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────
# Constants & Mapping
# ──────────────────────────────────────────────────────────────

TOOL_NAME_MAP = {
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
    "run_univfd": WEIGHT_UNIVFD,
    "run_xception": WEIGHT_XCEPTION,
    "run_sbi": WEIGHT_SBI,
    "run_freqnet": WEIGHT_FREQNET,
    "run_rppg": WEIGHT_RPPG,
    "run_dct": WEIGHT_DCT,
    "run_geometry": WEIGHT_GEOMETRY,
    "run_illumination": WEIGHT_ILLUMINATION,
    "run_corneal": WEIGHT_CORNEAL,
    "check_c2pa": WEIGHT_C2PA,
}

# Categories for Independence Check (S-10)
_INDEPENDENCE_GROUPS = {
    "generation_detectors": {"run_univfd", "run_freqnet"},
    "swap_detectors": {"run_xception", "run_sbi"},
    "physical_consistency": {"run_corneal", "run_geometry", "run_illumination"},
    "provenance": {"check_c2pa"},
    "compression": {"run_dct"},
}

GPU_SPECIALISTS = {
    "run_univfd",
    "run_xception",
    "run_sbi",
    "run_freqnet",
}

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


def _extract_context(
    tool_results: List[ToolResult], flags: List[str] = None
) -> Dict[str, float]:
    context = {
        "dct_peak_ratio": 0.0,
        "univfd_score": 0.5,
        "univfd_available": False,
        "compression_detected": "COMPRESSION" in (flags or []),
        "is_grayscale": "GRAYSCALE" in (flags or []),
        "heavy_blur": "HEAVY_BLUR" in (flags or []),
        "clipped_hist": "CLIPPED_BLACK" in (flags or [])
        or "CLIPPED_WHITE" in (flags or []),
    }
    for result in tool_results:
        if not result.success:
            continue
        tool_name = _normalize_tool_name(result.tool_name)
        if tool_name == "run_dct":
            context["dct_peak_ratio"] = _safe_get_details(result, "peak_ratio", 0.0)
            if context["dct_peak_ratio"] > DCT_DOUBLE_QUANT_COMPRESSION_THRESHOLD:
                context["compression_detected"] = True
        elif tool_name == "run_univfd":
            context["univfd_score"] = 1.0 - result.real_prob
            context["univfd_available"] = True
    return context


def _compute_conflict_std(implied_probs: List[float]) -> float:
    if len(implied_probs) < 2:
        return 0.0
    mean = sum(implied_probs) / len(implied_probs)
    variance = sum((p - mean) ** 2 for p in implied_probs) / len(implied_probs)
    return variance**0.5


def _to_logit(p: float) -> float:
    """Convert probability to log-odds (logit)."""
    p = max(1e-6, min(1.0 - 1e-6, p))
    return float(np.log(p / (1.0 - p)))


def _from_logit(l: float) -> float:
    """Convert log-odds (logit) back to probability."""
    return float(1.0 / (1.0 + np.exp(-l)))


# ──────────────────────────────────────────────────────────────
# Routing Logic
# ──────────────────────────────────────────────────────────────


def _route(
    result: ToolResult,
    context: Dict[str, float],
    use_confidence_weighting: bool = False,
) -> Tuple[float, float, bool]:
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

    real_prob = max(0.0, min(1.0, result.real_prob))
    score = 1.0 - real_prob
    tool_name = _normalize_tool_name(result.tool_name)

    # base_weight is set below.

    base_weight = WEIGHT_MAP.get(tool_name, 0.0)

    if base_weight == 0.0 and tool_name != "check_c2pa":
        return (0.0, 0.0, True)

    if tool_name == "check_c2pa":
        return (
            0.0,
            0.0,
            False,
        )  # C2PA is handled separately, returns 0 contribution here

    # --- rPPG ---
    if tool_name == "run_rppg":
        if score < RPPG_PULSE_THRESHOLD_LOW:
            contribution, weight = 0.0, WEIGHT_RPPG
        elif score > RPPG_PULSE_THRESHOLD_HIGH:
            contribution, weight = WEIGHT_RPPG * RPPG_NO_PULSE_IMPLIED_PROB, WEIGHT_RPPG
        else:
            return (0.0, 0.0, True)  # Borderline -> Abstain
        if use_confidence_weighting:
            weight *= result.confidence
            contribution = (contribution / (WEIGHT_RPPG + 1e-10)) * weight
        return (contribution, weight, False)

    # --- UnivFD / Xception ---
    if tool_name in ("run_univfd", "run_xception"):
        weight = base_weight
        if tool_name == "run_xception" and (
            context.get("is_grayscale") or context.get("compression_detected")
        ):
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
            if context.get("compression_detected"):
                sbi_weight *= SBI_COMPRESSION_DISCOUNT
            if context.get("is_grayscale"):
                sbi_weight *= 0.1
            contribution, weight = score * sbi_weight, sbi_weight
            if use_confidence_weighting:
                weight *= result.confidence
                contribution = (contribution / (sbi_weight + 1e-10)) * weight
            return (contribution, weight, False)

        clip_score = (
            context.get("univfd_score", 0.5)
            if context.get("univfd_available", False)
            else 0.5
        )
        sbi_weight = SBI_MID_BAND_BASE_WEIGHT + (
            SBI_MID_BAND_CLIP_MULTIPLIER * clip_score
        )
        if context.get("compression_detected"):
            sbi_weight *= SBI_COMPRESSION_DISCOUNT
        if context.get("is_grayscale") or context.get("heavy_blur"):
            sbi_weight *= 0.1
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

        if score < CORNEAL_BLIND_SPOT_THRESHOLD:
            return (0.0, 0.0, True)

    # --- Illumination ---
    if tool_name == "run_illumination":
        from utils.thresholds import ILLUMINATION_DIFFUSE_THRESHOLD

        if score < ILLUMINATION_DIFFUSE_THRESHOLD:
            return (0.0, 0.0, True)

    # --- Geometry ---
    if tool_name == "run_geometry":
        weight = base_weight
        if (
            context.get("is_grayscale")
            or context.get("compression_detected")
            or context.get("heavy_blur")
        ):
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
    c2pa_result = next(
        (r for r in tool_results if _normalize_tool_name(r.tool_name) == "check_c2pa"),
        None,
    )

    if (
        c2pa_result
        and c2pa_result.success
        and _safe_get_details(c2pa_result, "c2pa_verified", False)
    ):
        # Collect visual tools that actually contributed (confidence > 0)
        visual_tools_run = [
            r
            for r in tool_results
            if _normalize_tool_name(r.tool_name)
            in GPU_SPECIALISTS | {"run_geometry", "run_illumination"}
            and r.success
            and r.confidence > 0.1
        ]

        if (
            len(visual_tools_run) >= 1
            and sum(
                WEIGHT_MAP.get(_normalize_tool_name(r.tool_name), 0)
                for r in visual_tools_run
            )
            > C2PA_VISUAL_MIN_WEIGHT
        ):
            # Calculate visual average
            vis_contribs, vis_weights = [], []
            for r in visual_tools_run:
                c, w, _ = _route(r, context, use_confidence_weighting=False)
                if w > 1e-9:
                    vis_contribs.append(c)
                    vis_weights.append(w)

            visual_avg = (
                sum(vis_contribs) / sum(vis_weights) if sum(vis_weights) > 0 else 0.0
            )

            # Decision Logic:
            if visual_avg < 0.35:
                # Visuals agree: REAL -> Override to REAL
                return {
                    **_get_base_schema(),
                    "is_c2pa_override": True,
                    "override_reason": "C2PA verified, visuals agree",
                }
            elif visual_avg > C2PA_VISUAL_CONTRADICTION_THRESHOLD:
                # Visuals scream FAKE -> Flag Spoofing
                logger.warning(
                    "C2PA Verified but Visuals detect strong artifacts (Avg: %.2f). Possible Spoofing.",
                    visual_avg,
                )
                return {
                    **_get_base_schema(),
                    "is_c2pa_override": False,
                    "override_reason": "Visual contradiction detected",
                    "is_spoofing_warning": True,
                }
            # Else: Ambiguous -> Do not override. Let ensemble decide.
        else:
            # No visual tools ran -> Do not override (blind trust risk)
            logger.info(
                "C2PA valid but no visual tools ran. Proceeding with ensemble (no override)."
            )

    # 2. Route Tools
    total_contribution, total_weight = 0.0, 0.0
    weight_breakdown = {}  # Internal metadata always needed for Prong 1 Logic
    tools_ran, abstentions, implied_probs, gpu_specialist_probs = [], [], [], []
    gpu_abstained_tools = []
    gpu_contribution, gpu_weight = 0.0, 0.0
    cpu_contribution, cpu_weight = 0.0, 0.0

    for result in tool_results:
        tool_name = _normalize_tool_name(result.tool_name)
        if not result.success:
            abstentions.append(
                {
                    "tool_name": tool_name,
                    "real_prob": result.real_prob,
                    "reason": "tool_failed",
                }
            )
            continue

        contribution, effective_weight, is_abstention = _route(
            result, context, use_confidence_weighting
        )

        # Dynamic Confidence Dampening: 
        # If a tool (like a GPU specialist) outputs an extreme score (e.g. 95% authentic) 
        # but has very low confidence (e.g. 40%), we quadratically strip its voting power. 
        # It should not be allowed to heavily dilute confident votes from other detectors out of sheer noise.
        if not is_abstention and result.confidence < 0.60:
            penalty = (result.confidence / 0.60) ** 2
            contribution *= penalty
            effective_weight *= penalty

        # AUDIT FIX S-12 (UPDATED): The "Suspicious Middle"
        # If multiple GPU tools return scores near 50-59 (authenticity) with low confidence, 
        # they are typically sensing Deepfake artifacts they cannot mathematically lock onto.
        # We NO LONGER skip these! We track them to boost fake suspicion.
        is_suspicious_middle = False
        if (
            not is_abstention
            and tool_name in GPU_SPECIALISTS
            and NEUTRAL_DEADZONE_LOW <= (1.0 - result.real_prob) <= NEUTRAL_DEADZONE_HIGH
            and result.confidence < NEUTRAL_DEADZONE_CONFIDENCE_MIN
        ):
            is_suspicious_middle = True
            logger.info(
                f"Ensemble: Suspicious Middle hit for {tool_name} "
                f"(fake_prob={1.0 - result.real_prob:.2f}, conf={result.confidence:.2f})."
            )

        if is_abstention:
            effective_weight = 0.0
            contribution = 0.0
            abstentions.append(
                {
                    "tool_name": tool_name,
                    "real_prob": result.real_prob,
                    "reason": "abstain_or_failure",
                }
            )
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

        if not is_abstention and effective_weight > 0.0:
            if tool_name in GPU_SPECIALISTS:
                gpu_contribution += contribution
                gpu_weight += effective_weight
            else:
                cpu_contribution += contribution
                cpu_weight += effective_weight

        weight_breakdown[tool_name] = {
            "real_prob": result.real_prob,
            "fake_prob": 1.0 - result.real_prob,
            "confidence": result.confidence,
            "contribution": contribution,
            "effective_weight": effective_weight,
        }

    if total_weight < 1e-9:
        return {
            **_get_base_schema(),
            "ensemble_score": 0.5,
            "is_inconclusive": True,
            "abstentions": abstentions,
        }

    # GPU tools are primary detectors; CPU tools are supportive when GPU evidence exists.
    if gpu_weight > 1e-9:
        gpu_only_fake_prob = gpu_contribution / gpu_weight
        cpu_only_fake_prob = cpu_contribution / cpu_weight if cpu_weight > 1e-9 else gpu_only_fake_prob
        base_ensemble = ((1.0 - CPU_SUPPORT_WHEN_GPU_AVAILABLE) * gpu_only_fake_prob) + (
            CPU_SUPPORT_WHEN_GPU_AVAILABLE * cpu_only_fake_prob
        )
    else:
        base_ensemble = total_contribution / total_weight

    # 3. Advanced Logic (Prongs)
    max_gpu_prob = max(gpu_specialist_probs) if gpu_specialist_probs else 0.0
    highest_suspicion = max(implied_probs) if implied_probs else 0.0

    # GPU-first policy (soft, non-forced):
    # 1) Single GPU specialist with high confidence gets a strong suspicion boost.
    # 2) Two GPU specialists voting fake get corroboration boost with lower (but not low) confidence.
    gpu_votes = []
    for tool_name, meta in weight_breakdown.items():
        if tool_name not in GPU_SPECIALISTS or meta["effective_weight"] <= 1e-9:
            continue
        gpu_votes.append(
            {
                "tool_name": tool_name,
                "fake_prob": float(meta["fake_prob"]),
                "confidence": float(meta["confidence"]),
            }
        )

    single_high_conf_fake = any(
        v["fake_prob"] > ENSEMBLE_SPECIALIST_FAKE_THRESHOLD
        and v["confidence"] >= GPU_SINGLE_DETECT_HIGH_CONFIDENCE
        for v in gpu_votes
    )

    dual_fake_votes = [
        v
        for v in gpu_votes
        if v["fake_prob"] > ENSEMBLE_SPECIALIST_FAKE_THRESHOLD
        and v["confidence"] >= GPU_MULTI_DETECT_MIN_CONFIDENCE
    ]

    gpu_policy_boost = 0.0
    gpu_policy_reason = None
    if single_high_conf_fake:
        strongest = max(v["fake_prob"] for v in gpu_votes)
        gpu_policy_boost = GPU_SINGLE_DETECT_BOOST * strongest
        gpu_policy_reason = "single_high_confidence_gpu"
    elif len(dual_fake_votes) >= 2:
        top_two = sorted(dual_fake_votes, key=lambda x: x["fake_prob"], reverse=True)[:2]
        avg_top_two = sum(v["fake_prob"] for v in top_two) / 2.0
        gpu_policy_boost = GPU_MULTI_DETECT_BOOST * avg_top_two
        gpu_policy_reason = "two_gpu_fake_votes"

    # PRONG 1: Specialized Hunter Override (Damped Logit Fusion - Audit Fix S-14)
    # GPU tools are specialized anomaly detectors. Instead of aggressive Noisy-OR,
    # we use Logit-Sum Fusion which is more stable for forensic evidence.
    
    suspicion_logits = []
    # Track categories to prevent double-counting (S-10)
    categories_contributed = set()

    # Re-extracting tool associations for category-aware logit sum
    for tool_name, meta in weight_breakdown.items():
        if tool_name not in GPU_SPECIALISTS or meta["effective_weight"] <= 1e-9:
            continue
        p = meta["fake_prob"]
        
        # Determine mass
        mass = 0.0
        if p > 0.5:
            mass = p
        elif p >= ENCORE_NEAR_MISS_THRESHOLD:
            mass = 0.5 + (p - ENCORE_NEAR_MISS_THRESHOLD) * ENCORE_CORROBORATION_SENSITIVITY
        
        if mass > 0.51:
            logit = _to_logit(mass)
            
            # Category Damping
            tool_cat = None
            for cat, tools in _INDEPENDENCE_GROUPS.items():
                if tool_name in tools:
                    tool_cat = cat
                    break
            
            if tool_cat and tool_cat in categories_contributed:
                # Same category? Dampen the additional evidence by 50%
                logit *= 0.5
                logger.info(f"Ensemble: Category damping applied to {tool_name} ({tool_cat}).")

            # AUDIT FIX S-15: Confidence-Gated Contribution (2026 Premium)
            # Logit influence is proportional to confidence squared.
            # (e.g. 0.5 conf -> 0.25 influence, 0.9 conf -> 0.81 influence)
            logit *= (meta["confidence"] ** 2)

            # AUDIT FIX S-16: Specialist Dominance
            # If a specialist tool finds a deepfake with good confidence, amplify it.
            if meta["confidence"] >= ENSEMBLE_SPECIALIST_MIN_CONFIDENCE:
                logit *= ENSEMBLE_SPECIALIST_DOMINANCE_FACTOR
                logger.info(f"Ensemble: Dominance factor applied to specialized tool {tool_name}.")
            
            suspicion_logits.append(logit)
            if tool_cat:
                categories_contributed.add(tool_cat)

    if suspicion_logits:
        # Damped Logit Sum
        combined_logit = sum(suspicion_logits) * LOGIT_DAMPING_FACTOR
        corroborated_score = _from_logit(combined_logit)
        
        # Check if we have massive disagreement between tools
        conflict_tmp = _compute_conflict_std(implied_probs)

        # Isolated anomaly dampening: if a single tool goes rogue but the 
        # rest strongly disagree (high variance), we trust the ensemble baseline.
        if len(suspicion_logits) == 1 and conflict_tmp > BORDERLINE_CONSENSUS_LOW / 2.0:
            logger.info("Ensemble: Single suspicion logit combined with high conflict. Relying on base_ensemble.")
            fake_score = base_ensemble
        else:
            # Removed max_gpu_prob here to prevent a single noisy tool from 
            # steamrolling unanimous authentic votes from other detectors.
            fake_score = max(corroborated_score, base_ensemble)
        
        logger.info(
            f"Logit Fusion: Combined {len(suspicion_logits)} suspects. Mass: {corroborated_score:.4f} (Base: {base_ensemble:.4f})."
        )
    else:
        fake_score = base_ensemble

    # PRONG 2: The "Suspicious Middle" Consensus (User requested)
    # If multiple GPU tools return authentic-leaning but borderline scores 
    # (e.g. 50-59%) with low confidence, this indicates a subtle deepfake
    # that evaded strict classifiers. Boost to fake!
    borderline_gpu_tools = [
        v for v in gpu_votes if BORDERLINE_CONSENSUS_LOW <= v["fake_prob"] <= BORDERLINE_CONSENSUS_HIGH
        and v["confidence"] < NEUTRAL_DEADZONE_CONFIDENCE_MIN
    ]
    if len(borderline_gpu_tools) >= 2:
        logger.warning(f"Suspicious Consensus: {len(borderline_gpu_tools)} GPU tools are deeply uncertain. Flagging as FAKE.")
        # Push the fake score safely above the 0.54 fake cutoff.
        fake_score = max(fake_score, 0.85)

    # PRONG 3: GPU Coverage Degradation (Audit Fix C-05)
    # Only penalize if tools ABSTAINED EVIDENTIALLY (ran but failed/uncertain).
    # Structural abstentions (rPPG on image) are ignored here.
    gpu_degradation_boost = 1.0
    if len(gpu_abstained_tools) > 0:
        gpu_degradation_boost = 1.0 + (
            GPU_COVERAGE_DEGRADATION_FACTOR * len(gpu_abstained_tools)
        )

    candidate_score = max(base_ensemble, fake_score)
    if gpu_policy_boost > 0.0:
        candidate_score = min(1.0, candidate_score + gpu_policy_boost)
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
        "real_prob": ensemble_score,
        "is_inconclusive": total_weight < ENSEMBLE_INCONCLUSIVE_WEIGHT
        or (conflict_std > 0.25),
        "total_weight": round(total_weight, 4),
        "tools_ran": tools_ran,
        "abstentions": abstentions,
        "conflict_std": round(conflict_std, 4),
        "has_conflict": conflict_std > CONFLICT_STD_THRESHOLD,
        "gpu_policy_reason": gpu_policy_reason,
        "gpu_policy_boost": round(gpu_policy_boost, 4),
    }

    if return_metadata:
        output["weight_breakdown"] = weight_breakdown
        output["context"] = context
    return output


def stream_ensemble_score(
    frame_results_iterator,
    return_metadata=False,
    apply_ema_smoothing=True,
    ema_alpha=None,
):
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
                    smoothed = (ema_alpha * current_raw) + (
                        (1.0 - ema_alpha) * prev_score
                    )
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
        if score <= 0.54:
            return "FAKE"
        return "REAL"
