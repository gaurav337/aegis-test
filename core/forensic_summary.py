"""Forensic Summary — Tool outputs → Structured Phi-3 prompt.

Converts ensemble results into a natural language prompt that grounds
every claim in specific tool evidence. The LLM never sees raw pixels.
"""

from typing import Dict
from core.data_types import ToolResult
from utils.thresholds import REAL_THRESHOLD, FAKE_THRESHOLD


def build_phi3_prompt(
    ensemble_score: float,
    tool_results: Dict[str, ToolResult],
    verdict: str,
) -> str:
    """
    Build structured prompt for Phi-3 Mini reasoning.
    
    All scores are converted to REAL PROBABILITY (0.0 = definitely fake, 1.0 = definitely real).
    Per Spec Section 5: LLM receives only structured text, never raw images.
    """
    prompt_parts = []
    
    # ── CRITICAL SCORE DIRECTION BLOCK ──
    prompt_parts.append("=== CRITICAL: SCORE INTERPRETATION ===")
    prompt_parts.append("ALL SCORES BELOW ARE 'AUTHENTICITY SCORES' (Real Probability):")
    prompt_parts.append("  100% = Definitely REAL/AUTHENTIC")
    prompt_parts.append("  50%  = Uncertain / Inconclusive")
    prompt_parts.append("  0%   = Definitely FAKE/MANIPULATED")
    prompt_parts.append("A LOW score means this tool DETECTED manipulation artifacts.")
    prompt_parts.append("A HIGH score means this tool found NO manipulation artifacts.")
    prompt_parts.append("")
    
    # ── HEADER ──
    prompt_parts.append("=== AEGIS-X FORENSIC ANALYSIS ===\n")
    prompt_parts.append(f"Overall Authenticity Score: {ensemble_score:.1%}")
    prompt_parts.append(f"Verdict: {verdict}\n")
    
    # ── TOOL EVIDENCE (Categorized) ──
    prompt_parts.append("=== SECONDARY HELPERS (CPU heuristics) ===")
    prompt_parts.append("These are classical physics/geometry heuristics that act as supporters.")
    prompt_parts.append("They are noisy and should NEVER override a unanimous Authentic read from Primary AI Detectors.\n")
    
    # Helper to interpret real probability directly
    def _interpret(real_prob: float, tool_desc: str, high_label: str, low_label: str, confidence: float = 1.0) -> str:
        if confidence == 0.0:
            return f"{tool_desc}: [ABSTAINED] Insufficient data. Do NOT assume authenticity."
        pct = f"{real_prob:.0%}"
        if real_prob >= 0.70:
            interpretation = f"CLEAR — {high_label}"
        elif real_prob >= 0.45:
            interpretation = f"INCONCLUSIVE — borderline result"
        else:
            interpretation = f"SUSPICIOUS — {low_label}"
        return f"{tool_desc}: Authenticity {pct} → {interpretation}"
    
    # C2PA
    if "check_c2pa" in tool_results:
        r = tool_results["check_c2pa"]
        if r.details.get("c2pa_verified"):
            prompt_parts.append(f"✅ C2PA: Cryptographically verified (signer: {r.details.get('signer', 'Unknown')})")
        else:
            prompt_parts.append("⚪ C2PA: No provenance data found (unsigned media)")
    
    # rPPG
    if "run_rppg" in tool_results:
        r = tool_results["run_rppg"]
        label = r.details.get("liveness_label", "UNKNOWN")
        prompt_parts.append(f"💓 rPPG (Pulse Detection Helper): {label} — {r.evidence_summary}")
        if r.confidence == 0.0 or label == "UNKNOWN":
            prompt_parts.append("  → CRITICAL INSTRUCTION: If rPPG failed to track a human pulse, this STRONGLY suggests the video is an AI generation (lacking real micro-blood flow) - unless the video is simply too short to measure. Explicitly mention this in your summary!")
    
    # DCT
    if "run_dct" in tool_results:
        r = tool_results["run_dct"]
        prompt_parts.append(f"📐 DCT (Compression Analysis): peak_ratio={r.details.get('peak_ratio', 0):.3f}")
        if r.details.get("grid_artifacts"):
            prompt_parts.append("  → Double-quantization artifacts detected (compression or tampering)")
    
    # GEOMETRY
    if "run_geometry" in tool_results:
        r = tool_results["run_geometry"]
        violations = r.details.get("violations", [])
        prompt_parts.append(_interpret(r.real_prob, "📏 Geometry (Facial Proportion Helper)", 
                                       "all anatomical ratios normal",
                                       "impossible facial proportions detected", r.confidence))
        if violations:
            prompt_parts.append(f"  → Failed checks: {', '.join(violations)}")
    
    # Illumination
    if "run_illumination" in tool_results:
        r = tool_results["run_illumination"]
        prompt_parts.append(_interpret(r.real_prob, "💡 Illumination (Lighting Consistency Helper)",
                                       "consistent lighting across face",
                                       "lighting inconsistencies detected", r.confidence))
        if r.confidence == 0.0:
            prompt_parts.append("  → ABSTAINED: insufficient directional lighting data")
    
    # Corneal
    if "run_corneal" in tool_results:
        r = tool_results["run_corneal"]
        prompt_parts.append(_interpret(r.real_prob, "👁️ Corneal (Eye Reflection Helper)",
                                       "consistent corneal reflections",
                                       "mismatched eye reflections detected", r.confidence))
        if r.confidence == 0.0:
            prompt_parts.append("  → ABSTAINED: insufficient reflection data")
    
    prompt_parts.append("\n=== PRIMARY AI DETECTORS (GPU SPECIALISTS) ===")
    prompt_parts.append("These are deep-learning experts trained on millions of images. They are the true authorities.")
    prompt_parts.append("If one of these Primary Specialists finds manipulation, it is a STRONG fake signal.\n")
    
    # UnivFD
    if "run_univfd" in tool_results:
        r = tool_results["run_univfd"]
        prompt_parts.append(_interpret(r.real_prob, "🧠 UnivFD (AI-Generated Image Specialist)",
                                       "no GAN/diffusion fingerprints found",
                                       "AI-generation signatures detected", r.confidence))

    # Xception
    if "run_xception" in tool_results:
        r = tool_results["run_xception"]
        prompt_parts.append(_interpret(r.real_prob, "🔬 XceptionNet (Face-Swap Specialist)",
                                       "natural facial blending patterns",
                                       "face-swap blending artifacts detected", r.confidence))
    
    # SBI
    if "run_sbi" in tool_results:
        r = tool_results["run_sbi"]
        if r.details.get("boundary_detected"):
            prompt_parts.append(_interpret(r.real_prob, "🪡 SBI (Blend Boundary Specialist)",
                                           "no composite boundaries",
                                           f"blend boundary at {r.details.get('boundary_region', 'unknown')}", r.confidence))
        else:
            prompt_parts.append(_interpret(r.real_prob, "🪡 SBI (Blend Boundary Specialist)",
                                           "no blend boundaries detected",
                                           "blend boundaries detected", r.confidence))
    
    # FreqNet
    if "run_freqnet" in tool_results:
        r = tool_results["run_freqnet"]
        prompt_parts.append(_interpret(r.real_prob, "📡 FreqNet (Frequency Fingerprint Specialist)",
                                       "normal frequency spectrum",
                                       "anomalous frequency patterns detected", r.confidence))
    
    # ── REASONING RULES ──
    prompt_parts.append("\n=== REASONING RULES ===")
    prompt_parts.append("1. ALL scores above are AUTHENTICITY scores: high% = likely real, low% = likely fake.")
    prompt_parts.append("2. Trust the PRIMARY AI DETECTORS heavily. If a Primary Detector spots manipulation, it's highly suspicious.")
    prompt_parts.append("3. Treat SECONDARY HELPERS with skepticism on real images. They are noisy and cannot override Primary Detectors.")
    prompt_parts.append("4. Ground every claim in specific tool results, explicitly using their names.")
    prompt_parts.append("5. Use probabilistic language ('suggests', 'consistent with') — avoid absolute certainty.")
    prompt_parts.append("6. Keep explanation under 150 words in plain language.")
    
    # ── OUTPUT FORMAT ──
    prompt_parts.append("\n=== OUTPUT FORMAT ===")
    prompt_parts.append("Return a single paragraph of plain text containing your explanation. DO NOT use JSON, markdown, or any structured format. Just write the natural language explanation directly.")
    
    return "\n".join(prompt_parts)


async def generate_verdict(
    ensemble_score: float,
    tool_results: Dict[str, ToolResult],
    verdict: str,
) -> str:
    """
    Generate LLM verdict with streaming support.
    
    Returns:
        str: Natural language explanation from Phi-3
    """
    from core.llm import stream_completion
    
    prompt = build_phi3_prompt(ensemble_score, tool_results, verdict)
    
    explanation = ""
    async for token in stream_completion(prompt, temperature=0.1, max_tokens=512):
        explanation += token
        # Yield token for UI streaming (handled by agent.py)
    
    return explanation